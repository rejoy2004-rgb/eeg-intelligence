"""Euclidean alignment (He & Wu, 2019) - an UNSUPERVISED, target-calibrated domain-adaptation step.

For each group g (here: one participant in one session) the mean spatial covariance of its trials
    R_g = mean_i (X_i X_i^T / T)
is estimated and every trial is whitened with it:  X_i <- R_g^{-1/2} X_i.
After alignment every group has an identity mean covariance, which removes much of the between-participant
(and between-session) difference in spatial mixing and overall amplitude.

IMPORTANT - what it does and does not use
    * It uses NO labels.
    * It DOES use the unlabeled EEG of the group it aligns, including the held-out participant / session at test
      time. So a model that uses it is NOT zero-shot: it needs a block of unlabeled data from the target
      participant (here: all of their trials) before predicting. Results with this step must never be described
      as fully unseen-participant generalisation without that qualification.
"""
from __future__ import annotations

import numpy as np


def euclidean_alignment(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Align trials (n, channels, time) of ONE group. Returns float64 (n, channels, time)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or X.shape[0] == 0:
        raise ValueError(f"X must be a non-empty (n, channels, time) array; got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    cov = np.einsum("nct,ndt->cd", X, X) / (X.shape[0] * X.shape[2])
    vals, vecs = np.linalg.eigh(cov)
    if vals.min() <= eps * vals.max():
        raise ValueError("Mean covariance is (numerically) singular; cannot align")
    inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T
    return np.einsum("cd,ndt->nct", inv_sqrt, X)


def align_by_group(X: np.ndarray, groups) -> np.ndarray:
    """Apply ``euclidean_alignment`` separately to each distinct group label (rows of X keep their order)."""
    groups = np.asarray(groups)
    if len(groups) != len(X):
        raise ValueError("groups must have one entry per trial")
    out = np.empty(X.shape, dtype=np.float64)
    for g in np.unique(groups, axis=0) if groups.ndim > 1 else np.unique(groups):
        m = np.all(groups == g, axis=1) if groups.ndim > 1 else groups == g
        out[m] = euclidean_alignment(X[m])
    return out
