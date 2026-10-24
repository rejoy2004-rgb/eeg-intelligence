"""EOG handling for ocular-confound analysis: EOG loading, a calibration-block regression, and EOG-only features.

Why this module exists
    BCI IV 2a is cued by a left/right ARROW shown for 1.25 s. Gaze shifts or blinks time-locked to the cue could carry the class
    into frontal EEG channels. The dataset provides 3 monopolar EOG channels (left, central, right) *so that this can be checked*,
    and states that they "must not be used for classification". Consequently:
      * EOG is NEVER given to any classifier of this project as a feature.
      * EOG-only decoding (``eog_features``) is a DIAGNOSTIC of "does the eye signal carry class information?".
      * EOG is used a second way: as the regressor of a linear correction (``fit_eog_regression``) whose coefficients are
        estimated on the eye-movement calibration block (event 1072) that the data providers recorded for exactly this purpose.

What these tools can and cannot establish
    * EOG-only accuracy above chance shows that the eye channels contain class information; it does not show that the EEG
      classifier uses it.
    * A performance change after removing frontal channels or regressing out EOG is *compatible with* ocular reliance but does
      not prove it: frontal channels also carry neural signal and the EOG electrodes also pick up brain activity, so regression
      can remove neural signal (over-correction). Unchanged performance does not prove absence of ocular contribution either.
"""
from __future__ import annotations

import numpy as np

from src.data.dataset import Recording
from src.data.validation import DataValidationError
from src.signal.preprocessing import PreprocessingError, bandpass_filter

CALIBRATION_CODE = 1072          # "eye movements" block (official event table)
MIN_CALIBRATION_S = 30.0


def extract_eog(raw, rec: Recording) -> np.ndarray:
    """EOG channels (3, N) in microvolts from an in-memory ``mne`` Raw, NaN at the recording's documented missing segments."""
    idx = [i for i, n in enumerate(raw.ch_names) if n.upper().startswith("EOG")]
    if len(idx) != 3:
        raise DataValidationError(f"Expected 3 EOG channels, found {len(idx)}")
    eog = raw.get_data(picks=idx) * 1e6
    if eog.shape[1] != rec.n_samples:
        raise DataValidationError("EOG and EEG have different lengths")
    for a, b in rec.missing_segments:
        eog[:, a:b] = np.nan
    return eog


def load_eog(rec: Recording) -> np.ndarray:
    """Read the EOG channels belonging to ``rec`` from its GDF file (the recording loader deliberately drops them)."""
    import mne

    raw = mne.io.read_raw_gdf(rec.source_path, preload=True, verbose="ERROR")
    return extract_eog(raw, rec)


def calibration_slice(rec: Recording, code: int = CALIBRATION_CODE) -> tuple[int, int]:
    """[start, stop) samples of the eye-movement calibration block. Must exist once, be long enough, and END BEFORE the first trial."""
    ev = rec.events[rec.events["code"] == code]
    if len(ev) != 1:
        raise DataValidationError(f"Expected exactly one calibration block (event {code}); found {len(ev)}")
    start, stop = int(ev["sample"].iloc[0]), int(ev["sample"].iloc[0] + ev["duration"].iloc[0])
    if (stop - start) < MIN_CALIBRATION_S * rec.sfreq:
        raise DataValidationError(f"Calibration block shorter than {MIN_CALIBRATION_S:.0f} s")
    first_trial = int(rec.events.loc[rec.events["code"] == 768, "sample"].min())
    if stop > first_trial:
        raise DataValidationError("Calibration block overlaps the trial period; it would not be separate from evaluation data")
    return start, min(stop, rec.n_samples)


def fit_eog_regression(eeg_f: np.ndarray, eog_f: np.ndarray, block: tuple[int, int]):
    """Ordinary least squares from 3 EOG channels to every EEG channel on the calibration block.

    ``eeg_f`` (C, N) and ``eog_f`` (3, N) must be band-passed identically (NaN gaps allowed). Returns (B (C, 3), diagnostics).
    Diagnostics: in-sample R^2 per channel and split-half R^2 (fit on the first half of the block, evaluated on the second).
    No label and no trial sample is involved.
    """
    a, b = block
    x, y = eog_f[:, a:b], eeg_f[:, a:b]
    ok = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0)
    if ok.sum() < 3 * 250:
        raise PreprocessingError("Too few finite calibration samples for the regression")
    x, y = x[:, ok], y[:, ok]
    B = np.linalg.lstsq(x.T, y.T, rcond=None)[0].T                      # (C, 3)

    def r2(Bm, xs, ys):
        res = ys - Bm @ xs
        return 1.0 - res.var(axis=1) / ys.var(axis=1)

    h = x.shape[1] // 2
    B1 = np.linalg.lstsq(x[:, :h].T, y[:, :h].T, rcond=None)[0].T
    return B, {"r2_in_sample": r2(B, x, y), "r2_split_half": r2(B1, x[:, h:], y[:, h:]), "n_samples": int(x.shape[1])}


def apply_eog_regression(eeg_f: np.ndarray, eog_f: np.ndarray, B: np.ndarray) -> np.ndarray:
    """EEG minus the EOG-explained part. NaN gaps stay NaN. Shapes: eeg (C, N), eog (3, N), B (C, 3)."""
    if B.shape != (eeg_f.shape[0], eog_f.shape[0]):
        raise PreprocessingError(f"Coefficient matrix {B.shape} does not match EEG {eeg_f.shape} and EOG {eog_f.shape}")
    return eeg_f - B @ eog_f


def eog_features(eog_raw: np.ndarray, cue_samples: np.ndarray, sfreq: float, tmin: float, tmax: float,
                 baseline: tuple[float, float] = (-0.5, 0.0)) -> np.ndarray:
    """Six features per trial from the RAW EOG (3, N): [mean shift of each channel (task window minus pre-cue baseline),
    log variance of each channel in the task window]. Trials whose windows contain NaN yield NaN rows (caller checks)."""
    a, b = int(round(tmin * sfreq)), int(round(tmax * sfreq))
    b0, b1 = int(round(baseline[0] * sfreq)), int(round(baseline[1] * sfreq))
    out = np.full((len(cue_samples), 6), np.nan)
    for i, cue in enumerate(np.asarray(cue_samples, dtype=int)):
        if cue + b0 < 0 or cue + b > eog_raw.shape[1]:
            continue
        w, base = eog_raw[:, cue + a: cue + b], eog_raw[:, cue + b0: cue + b1]
        if np.isfinite(w).all() and np.isfinite(base).all():
            out[i, :3] = (w - base.mean(axis=1, keepdims=True)).mean(axis=1)
            out[i, 3:] = np.log(w.var(axis=1) + 1e-9)
    return out


def filter_eog(eog: np.ndarray, sfreq: float, band, order: int) -> np.ndarray:
    """Band-pass the EOG exactly like the EEG (same band, order, zero-phase, per gap-free segment)."""
    return bandpass_filter(eog, sfreq, band[0], band[1], order)
