"""Simple time-domain statistics per channel.

Input is always (n, C, T) finite microvolts. Output shapes are (n, C) or (n, 3, C).
These are label-free: they use only the trial's own samples (no training-set statistics),
so they cannot leak information between data partitions.
"""
from __future__ import annotations

import numpy as np

from src.features.spectral import FeatureError, _check_epochs


def log_variance(X: np.ndarray, sfreq: float = 1.0, floor: float = 1e-12) -> np.ndarray:
    """Log of the signal variance per channel. For band-passed EEG this is a log band-power proxy."""
    _check_epochs(X, sfreq)
    return np.log(np.maximum(X.var(axis=-1), floor))


def hjorth(X: np.ndarray, sfreq: float = 1.0, floor: float = 1e-12) -> np.ndarray:
    """Hjorth parameters (activity, mobility, complexity) -> (n, 3, C).

    activity   = var(x)
    mobility   = sqrt(var(dx) / var(x))
    complexity = mobility(dx) / mobility(x)
    """
    _check_epochs(X, sfreq)
    if X.shape[2] < 3:
        raise FeatureError("Hjorth parameters need at least 3 samples per trial")
    dx = np.diff(X, axis=-1)
    ddx = np.diff(dx, axis=-1)
    v0, v1, v2 = (np.maximum(a.var(axis=-1), floor) for a in (X, dx, ddx))
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / mobility
    return np.stack([v0, mobility, complexity], axis=1)
