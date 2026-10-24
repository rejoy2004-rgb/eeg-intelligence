"""Metric tests, including undefined / edge cases."""
import numpy as np
import pytest

from src.evaluation.metrics import (MetricsError, balanced_accuracy_simple, chance_threshold, compute_metrics,
                                    summarize, wilson_interval)


def test_known_confusion_matrix():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    m = compute_metrics(y_true, y_pred)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (3, 1, 4, 2)
    assert m["confusion_matrix"] == [[4, 2], [1, 3]]
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["recall"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(0.6)
    assert m["f1"] == pytest.approx(2 * 0.6 * 0.75 / 1.35)
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 4 / 6) / 2)


def test_perfect_and_worst():
    m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], scores=[-2, 2, -1, 1])
    assert m["accuracy"] == m["balanced_accuracy"] == m["f1"] == m["roc_auc"] == 1.0
    w = compute_metrics([0, 1, 0, 1], [1, 0, 1, 0], scores=[2, -2, 1, -1])
    assert w["accuracy"] == 0.0 and w["roc_auc"] == 0.0


def test_single_class_edge_cases_are_nan_with_notes():
    m = compute_metrics([1, 1, 1], [1, 0, 1], scores=[1.0, -1.0, 0.5])
    assert m["recall"] == pytest.approx(2 / 3)
    assert np.isnan(m["specificity"]) and np.isnan(m["balanced_accuracy"]) and np.isnan(m["roc_auc"])
    assert any("only one class" in n for n in m["notes"])
    neg = compute_metrics([0, 0], [0, 0])
    assert np.isnan(neg["precision"]) and np.isnan(neg["f1"]) and np.isnan(neg["recall"])
    assert neg["accuracy"] == 1.0


def test_no_positive_predictions_precision_undefined_not_zero():
    m = compute_metrics([0, 1, 0, 1], [0, 0, 0, 0])
    assert np.isnan(m["precision"]) and m["recall"] == 0.0
    assert any("precision" in n for n in m["notes"])


def test_f1_zero_when_precision_and_recall_zero():
    m = compute_metrics([1, 1, 0, 0], [0, 0, 1, 1])
    assert m["f1"] == 0.0


def test_auc_not_computed_without_scores():
    m = compute_metrics([0, 1], [0, 1])
    assert np.isnan(m["roc_auc"]) and any("not computed" in n for n in m["notes"])


def test_invalid_inputs():
    with pytest.raises(MetricsError, match="empty"):
        compute_metrics([], [])
    with pytest.raises(MetricsError, match="equal length"):
        compute_metrics([0, 1], [0])
    with pytest.raises(MetricsError, match="other than 0/1"):
        compute_metrics([0, 2], [0, 1])
    with pytest.raises(MetricsError, match="other than 0/1"):
        compute_metrics([0, 1], [0, -1])
    with pytest.raises(MetricsError, match="non-finite"):
        compute_metrics([0, 1], [0, 1], scores=[0.0, np.nan])
    with pytest.raises(MetricsError, match="same shape"):
        compute_metrics([0, 1], [0, 1], scores=[0.0])
    with pytest.raises(MetricsError):
        compute_metrics(np.zeros((2, 2)), np.zeros((2, 2)))


def test_balanced_accuracy_simple():
    assert balanced_accuracy_simple([0, 0, 0, 1], [0, 0, 0, 0]) == pytest.approx(0.5)
    assert balanced_accuracy_simple([1, 1], [1, 0]) == pytest.approx(0.5)
    assert np.isnan(balanced_accuracy_simple([], []))


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 or lo < 1e-9
    with pytest.raises(MetricsError):
        wilson_interval(1, 0)


def test_chance_threshold_decreases_with_more_trials():
    assert chance_threshold(100) == pytest.approx(0.59, abs=0.011)
    assert chance_threshold(1000) < chance_threshold(100) < chance_threshold(20) <= 1.0
    with pytest.raises(MetricsError):
        chance_threshold(0)


def test_summarize_ignores_nan():
    s = summarize([0.5, 0.7, float("nan")])
    assert s["n"] == 2 and s["mean"] == pytest.approx(0.6)
    assert np.isnan(summarize([])["mean"]) and summarize([])["n"] == 0
    assert np.isnan(summarize([0.5])["std"])
