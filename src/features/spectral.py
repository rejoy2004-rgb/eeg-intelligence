"""Spectral features: Welch power spectral density and band power.

Shapes (n = trials, C = channels, T = time samples, B = bands, F = frequency bins)
    input  X          : (n, C, T)  microvolts, finite
    psd               : (n, C, F)  uV^2 / Hz
    band_power        : (n, B, C)  mean PSD inside each band (uV^2 / Hz)
    log_band_power    : (n, B, C)  natural log of the above

"Band-major" ordering (B before C) mirrors EEGLearn, whose feature vector was
(theta_1..theta_64, alpha_1..alpha_64, beta_1..beta_64). Each band becomes one image channel later.

Band edges are conventions, not biological laws: individual mu/beta peaks vary between people.
We use the *mean* PSD in a band (not the sum) so wide and narrow bands are comparable.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.signal import welch


class FeatureError(ValueError):
    """Raised for invalid feature-extraction input."""


def _check_epochs(X: np.ndarray, sfreq: float) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 3:
        raise FeatureError(f"X must be a 3-D array (trials, channels, time); got {getattr(X, 'shape', type(X))}")
    if X.shape[0] == 0 or X.shape[1] == 0 or X.shape[2] == 0:
        raise FeatureError(f"X has an empty dimension: {X.shape}")
    if not (np.isfinite(sfreq) and sfreq > 0):
        raise FeatureError(f"Invalid sampling frequency {sfreq}")
    if not np.isfinite(X).all():
        raise FeatureError("X contains NaN or infinite values")


def welch_psd(X: np.ndarray, sfreq: float, nperseg: int = 250, overlap: float = 0.5):
    """Welch PSD along the time axis. Returns (freqs, psd)."""
    _check_epochs(X, sfreq)
    if nperseg > X.shape[2]:
        raise FeatureError(f"Trial has {X.shape[2]} samples, fewer than Welch nperseg={nperseg} (short trial)")
    if not 0 <= overlap < 1:
        raise FeatureError("overlap must be in [0, 1)")
    freqs, psd = welch(X, fs=sfreq, nperseg=nperseg, noverlap=int(nperseg * overlap), axis=-1, detrend="constant")
    return freqs, psd


def band_power(X: np.ndarray, sfreq: float, bands: Mapping[str, Sequence[float]],
               nperseg: int = 250, overlap: float = 0.5) -> np.ndarray:
    """Mean PSD per band. Output (n, n_bands, C); band order follows ``bands`` (dict order)."""
    if not bands:
        raise FeatureError("At least one band is required")
    nyq = sfreq / 2.0
    freqs, psd = welch_psd(X, sfreq, nperseg, overlap)
    out = np.empty((X.shape[0], len(bands), X.shape[1]))
    for b, (name, (lo, hi)) in enumerate(bands.items()):
        if not (0 <= lo < hi < nyq):
            raise FeatureError(f"Band '{name}' = ({lo}, {hi}) Hz is invalid for Nyquist {nyq} Hz")
        sel = (freqs >= lo) & (freqs <= hi)
        if not sel.any():
            raise FeatureError(
                f"Band '{name}' ({lo}-{hi} Hz) contains no frequency bin; resolution is {freqs[1] - freqs[0]:.2f} Hz"
            )
        out[:, b, :] = psd[:, :, sel].mean(axis=-1)
    return out


def log_band_power(X: np.ndarray, sfreq: float, bands: Mapping[str, Sequence[float]],
                   nperseg: int = 250, overlap: float = 0.5, floor: float = 1e-12) -> np.ndarray:
    """Natural log of band power. A flat (all-zero) channel would give log(0); we floor and keep finite."""
    bp = band_power(X, sfreq, bands, nperseg, overlap)
    return np.log(np.maximum(bp, floor))


def band_power_from_config(X: np.ndarray, sfreq: float, cfg) -> np.ndarray:
    """Convenience wrapper: log band power with the project's configured bands. Shape (n, B, C)."""
    f = cfg.features
    return log_band_power(X, sfreq, f["bands_hz"], f["welch_nperseg"], f["welch_overlap"])


def flatten_band_major(feats: np.ndarray) -> np.ndarray:
    """(n, B, C) -> (n, B*C) in band-major order (band_1 channels..., band_2 channels..., ...)."""
    if feats.ndim != 3:
        raise FeatureError(f"Expected (n, bands, channels); got {feats.shape}")
    return feats.reshape(feats.shape[0], -1)
