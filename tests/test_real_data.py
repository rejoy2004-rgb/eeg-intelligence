"""Checks against the REAL BCI Competition IV 2a files. Skipped automatically if they are not downloaded.

These verify facts documented in docs/DATASET.md (trial counts, event structure, label files).
They do not evaluate any model.
"""
import numpy as np
import pytest

from src.config import load_config
from src.data.dataset import load_external_labels, load_recording, recording_path
from src.signal.epoching import build_trial_set, locate_trials

cfg = load_config()
pytestmark = pytest.mark.realdata
needs_data = pytest.mark.skipif(not recording_path(cfg, 1, "T").is_file(), reason="BCI IV 2a data not downloaded")


@pytest.fixture(scope="module")
def a01t():
    return load_recording(cfg, 1, "T")


@needs_data
def test_a01t_structure(a01t):
    assert a01t.sfreq == 250.0 and a01t.data.shape[0] == 22
    assert a01t.ch_names[0] == "Fz" and a01t.ch_names[9] == "Cz"
    assert len(a01t.missing_segments) == 8 and all(b - a == 100 for a, b in a01t.missing_segments)
    ev = a01t.events
    assert (ev["code"] == 768).sum() == 288
    assert {769, 770, 771, 772} == set(ev["code"]) & {769, 770, 771, 772}


@needs_data
def test_a01t_labels_match_event_codes_and_counts(a01t):
    labels = load_external_labels(cfg, 1, "T")
    t = locate_trials(a01t, cfg, labels)
    assert len(t) == 288 and np.bincount(t["label4"].astype(int))[1:].tolist() == [72, 72, 72, 72]
    assert t["expert_artifact"].sum() == 15


@needs_data
def test_a01t_binary_trial_set(a01t):
    ts = build_trial_set(a01t, cfg, load_external_labels(cfg, 1, "T"))
    assert ts.X.shape[1:] == (22, 500)
    assert ts.summary["n_task_trials_before_rejection"] == 144
    assert ts.n_trials + ts.summary["n_rejected"] == 144
    assert np.isfinite(ts.X).all() and set(ts.y) == {0, 1}


@needs_data
def test_evaluation_session_has_unknown_cues_and_external_labels():
    rec = load_recording(cfg, 1, "E")
    assert (rec.events["code"] == 783).sum() == 288 and not rec.events["code"].isin([769, 770, 771, 772]).any()
    t = locate_trials(rec, cfg, load_external_labels(cfg, 1, "E"))
    assert t["label4"].notna().all()


@needs_data
def test_all_nine_participants_labels_counts_and_cue_timing():
    """Audit regression: label/event consistency for every participant and session (not just A01)."""
    for s in cfg.dataset["subjects"]:
        for sess in ("T", "E"):
            rec = load_recording(cfg, s, sess)
            labels = load_external_labels(cfg, s, sess)
            t = locate_trials(rec, cfg, labels)                     # raises if events and labels disagree
            assert len(t) == 288 and (t["cue_sample"] - t["trial_start"] == 500).all()
            assert np.bincount(labels)[1:].tolist() == [72, 72, 72, 72]
            assert len(rec.missing_segments) in (6, 8) and all(b - a == 100 for a, b in rec.missing_segments)
