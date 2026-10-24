"""Signal-quality indicators and simple, transparent artifact checks.

These are *screening* tools. A trial that passes them is not guaranteed to be artifact free, and a
rejected trial is not guaranteed to be bad. We report how many trials each rule removes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset import Recording


def epoch_peak_to_peak(epochs: np.ndarray) -> np.ndarray:
    """Per-epoch, per-channel peak-to-peak amplitude. Input (n, channels, time) -> (n, channels)."""
    if epochs.ndim != 3:
        raise ValueError(f"epochs must be 3-D (n, channels, time); got {epochs.shape}")
    return np.nanmax(epochs, axis=-1) - np.nanmin(epochs, axis=-1)


def flag_high_amplitude(epochs: np.ndarray, limit_uv: float) -> np.ndarray:
    """Boolean per epoch: True if ANY channel's peak-to-peak exceeds ``limit_uv`` (or data non-finite)."""
    if limit_uv <= 0:
        raise ValueError("limit_uv must be positive")
    if epochs.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    non_finite = ~np.isfinite(epochs).all(axis=(1, 2))
    with np.errstate(invalid="ignore"):
        ptp = epoch_peak_to_peak(np.where(np.isfinite(epochs), epochs, 0.0))
    return non_finite | (ptp > limit_uv).any(axis=1)


def recording_quality(rec: Recording) -> pd.DataFrame:
    """Per-channel quality table: RMS, std, max |amplitude|, flat-channel flag, fraction non-finite."""
    x = rec.data
    finite = np.isfinite(x)
    rows = []
    for i, name in enumerate(rec.ch_names):
        v = x[i][finite[i]]
        std = float(np.std(v)) if v.size else float("nan")
        rows.append(
            {
                "channel": name,
                "rms_uV": float(np.sqrt(np.mean(v**2))) if v.size else float("nan"),
                "std_uV": std,
                "max_abs_uV": float(np.max(np.abs(v))) if v.size else float("nan"),
                "flat": bool(v.size == 0 or std < 0.1),
                "fraction_missing": float(1.0 - finite[i].mean()),
            }
        )
    return pd.DataFrame(rows)
