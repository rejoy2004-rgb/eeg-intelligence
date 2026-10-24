"""Epoching tests (synthetic recordings; see conftest.py warning)."""
import numpy as np
import pandas as pd
import pytest

from src.data.validation import DataValidationError
from src.signal.epoching import (TrialSet, build_dual_band_trials, build_trial_set, extract_epochs, locate_trials)
from tests.conftest import make_synthetic_raw
from src.data.dataset import recording_from_raw


def test_locate_trials_structure(synth, cfg):
    rec, label4 = synth
    t = locate_trials(rec, cfg)
    assert len(t) == 32
    assert t["label4"].tolist() == label4.tolist()
    assert (t["cue_sample"] - t["trial_start"] == 500).all()
    assert sorted(t["run"].unique()) == [0, 1] and (t["run"].value_counts() == 16).all()
    assert not t["expert_artifact"].any()


def test_unknown_cue_needs_external_labels(cfg):
    raw, label4 = make_synthetic_raw(cfg, unknown_cue=True)
    rec = recording_from_raw(raw, cfg, 1, "E")
    assert locate_trials(rec, cfg)["label4"].isna().all()
    with pytest.raises(DataValidationError, match="unknown class"):
        build_trial_set(rec, cfg)                                          # labels required but absent
    ts = build_trial_set(rec, cfg, external_labels=label4)
    assert ts.n_trials == 16 and set(ts.y) == {0, 1}


def test_external_labels_wrong_length_or_conflicting(synth, cfg):
    rec, label4 = synth
    with pytest.raises(DataValidationError, match="external labels"):
        locate_trials(rec, cfg, label4[:-1])
    wrong = label4.copy()
    wrong[0] = 2 if wrong[0] == 1 else 1
    with pytest.raises(DataValidationError, match="disagree"):
        locate_trials(rec, cfg, wrong)
    with pytest.raises(DataValidationError, match="unknown"):
        locate_trials(rec, cfg, np.full(32, 9))


def test_event_structure_errors(cfg):
    raw, _ = make_synthetic_raw(cfg)
    raw.set_annotations(raw.annotations[[i for i, d in enumerate(raw.annotations.description) if d != "769"]])
    rec = recording_from_raw(raw, cfg, 1, "T")
    with pytest.raises(DataValidationError, match="cue events"):
        locate_trials(rec, cfg)
    raw, _ = make_synthetic_raw(cfg)
    raw.set_annotations(raw.annotations[[i for i, d in enumerate(raw.annotations.description) if d != "768"]])
    with pytest.raises(DataValidationError, match="No trial-start"):
        locate_trials(recording_from_raw(raw, cfg, 1, "T"), cfg)


def test_extract_epochs_shape_and_boundaries():
    sig = np.arange(2 * 1000, dtype=float).reshape(2, 1000)
    sig[:, 400:420] = np.nan
    ep, reasons = extract_epochs(sig, 250.0, np.array([100, 330, 900, 5]), 0.2, 0.6)   # 100-sample window
    assert ep.shape == (4, 2, 100)
    assert reasons[0] is None                                   # samples 150..250 are fine
    assert reasons[1] == "epoch overlaps missing data"          # samples 380..480 contain the NaN gap 400..420
    assert reasons[2] == "epoch extends beyond recording"       # samples 950..1050 run past the end
    assert reasons[3] is None
    assert np.isfinite(ep[0]).all() and np.isnan(ep[2]).all()
    assert np.array_equal(ep[0], sig[:, 150:250])


def test_extract_epochs_invalid_arguments():
    sig = np.zeros((2, 100))
    with pytest.raises(DataValidationError):
        extract_epochs(sig, 250.0, np.array([10]), 1.0, 0.5)
    with pytest.raises(DataValidationError):
        extract_epochs(np.zeros(100), 250.0, np.array([10]), 0.0, 0.2)
    with pytest.raises(DataValidationError):
        extract_epochs(sig, 250.0, np.array([10]), 0.0, 0.001)
    ep, reasons = extract_epochs(sig, 250.0, np.array([], dtype=int), 0.0, 0.2)
    assert ep.shape[0] == 0 and reasons == []


def test_trial_set_selects_binary_task_and_counts_other_classes(synth, cfg):
    rec, label4 = synth
    ts = build_trial_set(rec, cfg)
    assert ts.X.shape == (16, 22, 500)
    assert ts.summary["n_other_class"] == 16
    assert ts.summary["n_trials_in_file"] == 32
    assert set(ts.meta["label4"]) == {1.0, 2.0}
    assert ts.class_counts() == {0: 8, 1: 8}


def test_labels_stay_aligned_with_trials(synth, cfg):
    """Planted physiology: left trials have LOW 10 Hz variance at C4, right trials at C3."""
    rec, _ = synth
    ts = build_trial_set(rec, cfg)
    c3, c4 = ts.ch_names.index("C3"), ts.ch_names.index("C4")
    var = ts.X.var(axis=-1)
    left, right = ts.y == 0, ts.y == 1
    assert var[left, c4].mean() < 0.5 * var[right, c4].mean()
    assert var[right, c3].mean() < 0.5 * var[left, c3].mean()
    # meta label4 agrees with y
    assert ((ts.meta["label4"] == 2) == (ts.y == 1)).all()


def test_expert_artifact_rejection_is_recorded(cfg):
    raw, label4 = make_synthetic_raw(cfg, expert_artifact_trials=(0, 1, 2))   # classes 1,2,3
    rec = recording_from_raw(raw, cfg, 1, "T")
    ts = build_trial_set(rec, cfg)
    assert ts.n_trials == 14                                    # trials 0 and 1 are left/right
    assert ts.summary["rejection_counts"] == {"expert artifact marker (1023)": 2}
    assert len(ts.rejections) == 2 and set(ts.rejections["reason"]) == {"expert artifact marker (1023)"}
    assert ts.summary["n_kept"] + ts.summary["n_rejected"] == ts.summary["n_task_trials_before_rejection"]
    cfg["preprocessing"]["reject_expert_artifacts"] = False
    assert build_trial_set(rec, cfg).n_trials == 16


def test_amplitude_rejection(cfg):
    raw, _ = make_synthetic_raw(cfg, spike_trials=(0,))
    rec = recording_from_raw(raw, cfg, 1, "T")
    ts = build_trial_set(rec, cfg)
    assert ts.n_trials == 15
    assert any("peak-to-peak" in r for r in ts.rejections["reason"])
    assert ts.rejections["trial"].tolist() == [0]


def test_epoch_overlapping_gap_is_rejected_not_silently_dropped(cfg):
    raw, _ = make_synthetic_raw(cfg)
    rec = recording_from_raw(raw, cfg, 1, "T")
    # move the single gap into the epoch window of trial 0 (cue at 350+500=850; window 975..1475)
    a, b = rec.missing_segments[0]
    rec.data[:, a:b] = 0.0
    rec.data[:, 1000:1100] = np.nan
    rec.missing_segments = [(1000, 1100)]
    ts = build_trial_set(rec, cfg)
    assert "epoch overlaps missing data" in set(ts.rejections["reason"])
    assert ts.summary["n_rejected"] == len(ts.rejections)


def test_dual_band_sets_contain_identical_trials(synth, cfg):
    rec, _ = synth
    bp, csp = build_dual_band_trials(rec, cfg)
    assert bp.meta["trial"].tolist() == csp.meta["trial"].tolist()
    assert bp.y.tolist() == csp.y.tolist()
    assert bp.summary["band_hz"] != csp.summary["band_hz"]


def test_dual_band_sets_stay_aligned_under_a_tight_amplitude_limit(synth, cfg):
    rec, _ = synth
    # A bump that the two filter bands treat differently, plus a tight limit: whichever band rejects a
    # trial, the other must drop it too so that all models see identical trials.
    rec.data[3, 850 + 200:850 + 300] += 400.0 * np.hanning(100)       # broadband bump in trial 0 window
    cfg["preprocessing"]["reject_peak_to_peak_uv"] = 120.0
    bp, csp = build_dual_band_trials(rec, cfg)
    assert bp.meta["trial"].tolist() == csp.meta["trial"].tolist()


def test_trial_set_alignment_enforced():
    with pytest.raises(DataValidationError, match="Misaligned"):
        TrialSet(np.zeros((3, 2, 5)), np.zeros(2, dtype=int), pd.DataFrame({"a": [1, 2, 3]}),
                 pd.DataFrame(), 250.0, ["a", "b"])


def test_trial_set_concat_and_select(synth, cfg):
    rec, _ = synth
    ts = build_trial_set(rec, cfg)
    both = TrialSet.concat([ts, ts])
    assert both.n_trials == 32
    sel = ts.select(ts.y == 1)
    assert sel.n_trials == 8 and set(sel.y) == {1}
    with pytest.raises(DataValidationError, match="empty list"):
        TrialSet.concat([])


def test_empty_trial_set_after_total_rejection(cfg):
    raw, _ = make_synthetic_raw(cfg, expert_artifact_trials=tuple(range(32)))
    rec = recording_from_raw(raw, cfg, 1, "T")
    ts = build_trial_set(rec, cfg)
    assert ts.n_trials == 0 and ts.summary["n_rejected"] == 16
    assert len(ts.rejections) == 16
