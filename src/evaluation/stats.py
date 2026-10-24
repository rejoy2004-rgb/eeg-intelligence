"""Participant-level statistics (the unit of analysis is the participant, never the trial).

With 9 participants the smallest two-sided exact Wilcoxon p-value is 2/2^9 = 0.0039; bootstrap intervals over 9 participants are coarse.
Results should be read as indicative, not as population-level inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class PairingError(ValueError):
    """Raised when two sets of predictions are not scored on exactly the same test trials."""


def assert_same_test_trials(*prediction_tables: pd.DataFrame, protocol: str | None = None) -> int:
    """All tables (one per compared model/strategy) must contain exactly the same (participant, session, trial) test keys.

    Returns the number of shared test trials. Raises ``PairingError`` otherwise - a paired comparison on different trials
    would confound the method with the trial set.
    """
    if len(prediction_tables) < 2:
        raise PairingError("Need at least two prediction tables")
    key_sets = []
    for t in prediction_tables:
        d = t if protocol is None else t[t["protocol"] == protocol]
        if d.empty:
            raise PairingError("Empty prediction table")
        keys = list(zip(d["test_subject"], d["session"], d["trial"]))
        if len(keys) != len(set(keys)):
            raise PairingError("Duplicate test trials inside one prediction table (several models mixed?); filter to one model first")
        key_sets.append(frozenset(keys))
    if len(set(key_sets)) != 1:
        raise PairingError("Prediction tables do not cover identical test trials")
    return len(key_sets[0])


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN stays NaN and is ignored in the family size)."""
    p = np.asarray(pvalues, dtype=float)
    adj = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(order) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bootstrap_mean_ci(values, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling PARTICIPANTS."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo = (1 - level) / 2
    return float(np.quantile(means, lo)), float(np.quantile(means, 1 - lo))


def paired_comparison(a, b, seed: int = 0) -> dict:
    """Paired participant-level comparison of a vs b (aligned on the index = participant id)."""
    a, b = pd.Series(a), pd.Series(b)
    if set(a.index) != set(b.index):
        raise PairingError("The two result series are not indexed by the same participants")
    d = (a - b.reindex(a.index)).dropna()
    p = float(stats.wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else float("nan")
    lo, hi = bootstrap_mean_ci(d.to_numpy(), seed=seed) if len(d) else (float("nan"),) * 2
    return {"mean_diff": float(d.mean()) if len(d) else float("nan"), "ci_low": lo, "ci_high": hi, "p_wilcoxon": p,
            "n_participants": int(len(d)), "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum())}


def one_sample_vs_chance(values, chance: float = 0.5) -> dict:
    """Exact Wilcoxon of participant scores against chance (two-sided)."""
    v = pd.Series(values).dropna().to_numpy() - chance
    p = float(stats.wilcoxon(v).pvalue) if len(v) >= 6 and np.any(v != 0) else float("nan")
    return {"mean": float(np.mean(v) + chance), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), "p_wilcoxon": p,
            "n_participants": int(len(v))}


def within_group_permutation_test(y_true, y_pred, groups, participants, n_perm: int = 5000, seed: int = 0) -> dict:
    """Does the MEAN-over-participants balanced accuracy exceed what label exchangeability within groups (runs) allows?

    Predictions are held fixed and the true labels are permuted within each (participant, run) group, which preserves any
    temporal structure shared by predictions and labels at the run level. One-sided p = P(null mean >= observed mean).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    groups, participants = np.asarray(groups), np.asarray(participants)
    parts = np.unique(participants)

    def mean_ba(yt):
        v = []
        for s in parts:
            m = participants == s
            t, p = yt[m], y_pred[m]
            v.append(np.mean([(p[t == c] == c).mean() for c in (0, 1) if (t == c).any()]))
        return float(np.mean(v))

    observed = mean_ba(y_true)
    rng = np.random.default_rng(seed)
    idx_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    null = np.empty(n_perm)
    for k in range(n_perm):
        yt = y_true.copy()
        for idx in idx_by_group:
            yt[idx] = y_true[rng.permutation(idx)]
        null[k] = mean_ba(yt)
    return {"observed_mean_ba": observed, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "p_one_sided": float((1 + (null >= observed).sum()) / (n_perm + 1)), "n_perm": n_perm}
