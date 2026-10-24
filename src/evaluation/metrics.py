"""Binary-classification metrics with explicit handling of undefined cases.

Convention: class 1 = right hand = "positive". Class 0 = left hand = "negative".
A metric that cannot be computed (e.g. precision when nothing was predicted positive) is returned as
NaN and the reason is appended to ``result["notes"]`` - never silently as 0 or 1.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


class MetricsError(ValueError):
    """Raised for invalid metric inputs (empty data, wrong labels, mismatched lengths)."""


def _validate(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise MetricsError(f"y_true and y_pred must be 1-D with equal length; got {y_true.shape} and {y_pred.shape}")
    if y_true.size == 0:
        raise MetricsError("Cannot compute metrics on empty data")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        bad = set(np.unique(arr).tolist()) - {0, 1}
        if bad:
            raise MetricsError(f"{name} contains labels other than 0/1: {sorted(bad)}")
    return y_true.astype(int), y_pred.astype(int)


def balanced_accuracy_simple(y_true, y_pred) -> float:
    """Mean recall over the classes present in y_true (used for validation during CNN training)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = [(y_pred[y_true == c] == c).mean() for c in (0, 1) if (y_true == c).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def compute_metrics(y_true, y_pred, scores=None) -> dict:
    """Accuracy, balanced accuracy, precision, recall, specificity, F1, confusion matrix, ROC-AUC."""
    y_true, y_pred = _validate(y_true, y_pred)
    notes: list[str] = []
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n_pos, n_neg = tp + fn, tn + fp
    nan = float("nan")

    def ratio(num, den, what):
        if den == 0:
            notes.append(f"{what} undefined (denominator is 0)")
            return nan
        return num / den

    recall = ratio(tp, n_pos, "recall (no positive trials)")
    specificity = ratio(tn, n_neg, "specificity (no negative trials)")
    precision = ratio(tp, tp + fp, "precision (no positive predictions)")
    if np.isnan(precision) or np.isnan(recall):
        f1 = nan
        notes.append("F1 undefined")
    else:
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    if n_pos == 0 or n_neg == 0:
        balanced = nan
        notes.append("balanced accuracy undefined (only one class present in y_true)")
    else:
        balanced = (recall + specificity) / 2

    auc = nan
    if scores is not None:
        s = np.asarray(scores, dtype=float)
        if s.shape != y_true.shape:
            raise MetricsError("scores must have the same shape as y_true")
        if not np.isfinite(s).all():
            raise MetricsError("scores contain non-finite values")
        if n_pos == 0 or n_neg == 0:
            notes.append("ROC-AUC undefined (only one class present)")
        else:
            auc = float(roc_auc_score(y_true, s))
    else:
        notes.append("ROC-AUC not computed (no scores supplied)")

    return {
        "n": int(y_true.size), "n_positive": n_pos, "n_negative": n_neg,
        "accuracy": (tp + tn) / y_true.size, "balanced_accuracy": balanced,
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
        "roc_auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],     # rows: true (left, right); cols: predicted
        "notes": notes,
    }


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (accuracy on n independent trials)."""
    if n <= 0:
        raise MetricsError("n must be positive")
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def chance_threshold(n: int, alpha: float = 0.05) -> float:
    """Smallest accuracy that is significantly above 50% (one-sided binomial test) for n trials.

    Useful for judging whether a small test set could produce the observed accuracy by luck.
    """
    if n <= 0:
        raise MetricsError("n must be positive")
    k = int(stats.binom.isf(alpha, n, 0.5)) + 1   # P(X >= k) <= alpha
    return k / n


def summarize(values) -> dict:
    """Mean / sample std / min / max over participants, ignoring NaNs (reported with the count)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}
