"""Continuous-signal preprocessing: segment handling, band-pass filtering, re-referencing.

Why filter the *continuous* recording and not each short epoch?
    IIR filters have start-up transients. A 2-second epoch would be dominated by edge effects, so we
    filter long, gap-free stretches (the recording is split at the documented missing segments) and
    cut the epochs afterwards. The filter has no data-dependent parameters, so doing this for
    training and test recordings alike cannot leak information between them.

Filter used: Butterworth band-pass, applied forward and backward (``sosfiltfilt``).
    * zero phase (no time shift of the signal), effective order = 2 x ``order``
    * non-causal: it uses future samples. That is fine for offline analysis, but a real-time BCI
      would need a causal filter (the original competition required causality; this project does not
      claim to meet that requirement).
    * ``axis=-1`` is time. Input is always (channels, samples).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from src.data.dataset import Recording


class PreprocessingError(ValueError):
    """Raised for invalid preprocessing input or parameters."""


def finite_segments(data: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, stop) ranges where all channels are finite."""
    if data.ndim != 2:
        raise PreprocessingError(f"Expected 2-D (channels, samples) array, got shape {data.shape}")
    ok = np.isfinite(data).all(axis=0)
    if ok.all():
        return [(0, data.shape[1])]
    padded = np.concatenate([[0], ok.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def validate_band(low_hz: float, high_hz: float, sfreq: float) -> None:
    nyq = sfreq / 2.0
    if not (np.isfinite(low_hz) and np.isfinite(high_hz) and np.isfinite(sfreq)) or sfreq <= 0:
        raise PreprocessingError("Band edges and sampling frequency must be finite and sfreq > 0")
    if low_hz <= 0:
        raise PreprocessingError(f"Low cut-off must be > 0 Hz (got {low_hz})")
    if high_hz <= low_hz:
        raise PreprocessingError(f"High cut-off ({high_hz}) must exceed low cut-off ({low_hz})")
    if high_hz >= nyq:
        raise PreprocessingError(f"High cut-off ({high_hz} Hz) must be below the Nyquist frequency ({nyq} Hz)")


def bandpass_filter(data: np.ndarray, sfreq: float, low_hz: float, high_hz: float, order: int = 4,
                    causal: bool = False) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the time axis, segment by segment.

    ``causal=True`` uses a forward-only filter (``sosfilt``): the output at time t depends only on samples <= t, so no
    information from LATER samples can enter an earlier window (used by the pre-cue control). It has phase delay and a start-up
    transient at each segment start. The default zero-phase filter is non-causal: its impulse response spreads backwards in time.

    NaN samples (documented missing data) are preserved as NaN and never mixed into the filter.
    Raises ``PreprocessingError`` for empty/short input or unexpected NaNs inside a channel.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise PreprocessingError("data must be a 2-D numpy array (channels, samples)")
    if data.shape[1] == 0 or data.shape[0] == 0:
        raise PreprocessingError("Cannot filter an empty signal")
    if order < 1:
        raise PreprocessingError("Filter order must be >= 1")
    validate_band(low_hz, high_hz, sfreq)
    nan_cols = np.isnan(data)
    if np.isinf(data).any() or (nan_cols.any(axis=0) != nan_cols.all(axis=0)).any():
        raise PreprocessingError("Non-finite values found that are not whole-sample gaps; refusing to filter")

    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sfreq, output="sos")
    # sosfiltfilt needs more samples than its padding length (3 * (2 * n_sections + 1) by default)
    min_len = 3 * (2 * sos.shape[0] + 1) + 1
    out = np.full(data.shape, np.nan, dtype=np.float64)
    segments = finite_segments(data)
    if not segments or all(b - a == 0 for a, b in segments):
        raise PreprocessingError("No finite samples to filter")
    for a, b in segments:
        if b - a < min_len:
            raise PreprocessingError(
                f"Segment of {b - a} samples is too short to filter (need >= {min_len}). "
                "Use a longer recording or a lower filter order."
            )
        out[:, a:b] = sosfilt(sos, data[:, a:b], axis=-1) if causal else sosfiltfilt(sos, data[:, a:b], axis=-1)
    return out


def rereference(data: np.ndarray, mode: str = "none") -> np.ndarray:
    """Optional re-referencing. ``average`` subtracts the across-channel mean at each time sample.

    Recordings are monopolar against the left mastoid. With only 22 channels and no coverage of the
    lower head, an average reference is a rough approximation; the default is therefore ``none``.
    """
    if mode == "none":
        return data
    if mode == "average":
        return data - np.nanmean(data, axis=0, keepdims=True)
    raise PreprocessingError(f"Unknown rereference mode '{mode}' (use 'none' or 'average')")


def preprocess_recording(rec: Recording, cfg, band_hz: tuple[float, float] | None = None, causal: bool = False) -> np.ndarray:
    """Apply the configured preprocessing to a whole recording. Returns (channels, samples) in uV.

    The same function is used for training, evaluation and inference (consistency requirement).
    """
    pre = cfg.preprocessing
    low, high = band_hz if band_hz is not None else pre["bandpass_hz"]
    x = rereference(rec.data, pre["rereference"])
    return bandpass_filter(x, rec.sfreq, low, high, pre["filter_order"], causal=causal)
