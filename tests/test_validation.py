"""Data loading and validation tests (synthetic fixtures; see conftest.py warning)."""
import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from src.data.dataset import (OVERFLOW_UV, find_missing_segments, load_external_labels, load_gdf_file,
                              recording_from_raw, recording_path)
from src.data.validation import (DataValidationError, ValidationReport, check_finite, validate_channels,
                                 validate_events, validate_labels, validate_sfreq, validate_signal_array)
from tests.conftest import make_synthetic_raw


# ---- generic validators -------------------------------------------------------------
def test_signal_array_rejects_bad_inputs():
    with pytest.raises(DataValidationError, match="numpy array"):
        validate_signal_array([[1.0]], 1)
    with pytest.raises(DataValidationError, match="2-D"):
        validate_signal_array(np.zeros(5), 1)
    with pytest.raises(DataValidationError, match="channels"):
        validate_signal_array(np.zeros((3, 10)), 22)
    with pytest.raises(DataValidationError, match="empty"):
        validate_signal_array(np.zeros((22, 0)), 22)
    with pytest.raises(DataValidationError, match="floating"):
        validate_signal_array(np.zeros((22, 5), dtype=int), 22)
    validate_signal_array(np.zeros((22, 5)), 22)


def test_check_finite_flags_unexpected_nan_but_allows_gaps():
    x = np.ones((3, 10))
    x[:, 4] = np.nan
    gap = np.zeros(10, dtype=bool)
    gap[4] = True
    assert check_finite(x, allowed_missing=gap) == 1
    with pytest.raises(DataValidationError, match="non-finite"):
        check_finite(x)
    x2 = np.ones((3, 10))
    x2[1, 7] = np.inf
    with pytest.raises(DataValidationError):
        check_finite(x2, allowed_missing=gap)


def test_sfreq_validation():
    validate_sfreq(250.0, 250.0)
    with pytest.raises(DataValidationError, match="Unexpected sampling"):
        validate_sfreq(128.0, 250.0)
    for bad in (0, -5, float("nan")):
        with pytest.raises(DataValidationError):
            validate_sfreq(bad)


def test_channel_validation():
    expected = ["a", "b", "c"]
    validate_channels(["a", "b", "c"], expected, {0: "a"})
    with pytest.raises(DataValidationError, match="Expected 3"):
        validate_channels(["a", "b"], expected)
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_channels(["a", "a", "c"], expected)
    with pytest.raises(DataValidationError, match="position 0"):
        validate_channels(["x", "b", "c"], expected, {0: "a"})


def _events(rows):
    return pd.DataFrame(rows, columns=["sample", "code", "duration"])


def test_event_validation():
    ok = _events([(0, 768, 0), (500, 769, 0)])
    validate_events(ok, 1000, {768, 769})
    with pytest.raises(DataValidationError, match="outside"):
        validate_events(_events([(0, 768, 0), (5000, 769, 0)]), 1000, {768, 769})
    with pytest.raises(DataValidationError, match="outside"):
        validate_events(_events([(-1, 768, 0)]), 1000, {768})
    with pytest.raises(DataValidationError, match="Unknown event"):
        validate_events(_events([(0, 999, 0)]), 1000, {768})
    with pytest.raises(DataValidationError, match="chronological"):
        validate_events(_events([(10, 768, 0), (5, 769, 0)]), 1000, {768, 769})
    with pytest.raises(DataValidationError, match="no event"):
        validate_events(_events([]), 1000, {768})
    with pytest.raises(DataValidationError, match="columns"):
        validate_events(pd.DataFrame({"x": [1]}), 10, {1})


def test_label_validation():
    validate_labels(np.array([1, 2, 3]), [1, 2, 3, 4])
    with pytest.raises(DataValidationError, match="unknown"):
        validate_labels(np.array([1, 5]), [1, 2, 3, 4])
    with pytest.raises(DataValidationError, match="empty"):
        validate_labels(np.array([]), [1, 2])
    with pytest.raises(DataValidationError, match="non-finite"):
        validate_labels(np.array([1.0, np.nan]), [1, 2])


def test_report_counts_rejections():
    r = ValidationReport()
    r.reject(trial=1, reason="a")
    r.reject(trial=2, reason="a")
    r.reject(trial=3, reason="b")
    assert r.rejection_counts() == {"a": 2, "b": 1}


# ---- file loading errors -----------------------------------------------------------
def test_missing_file(cfg, tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        load_gdf_file(tmp_path / "A01T.gdf", cfg)


def test_unsupported_format(cfg, tmp_path):
    p = tmp_path / "A01T.edf"
    p.write_bytes(b"123")
    with pytest.raises(DataValidationError, match="Unsupported"):
        load_gdf_file(p, cfg)


def test_empty_file(cfg, tmp_path):
    p = tmp_path / "A01T.gdf"
    p.write_bytes(b"")
    with pytest.raises(DataValidationError, match="empty"):
        load_gdf_file(p, cfg)


def test_corrupted_file(cfg, tmp_path):
    p = tmp_path / "A01T.gdf"
    p.write_bytes(b"this is definitely not a GDF file" * 100)
    with pytest.raises(DataValidationError, match="Could not read"):
        load_gdf_file(p, cfg)


def test_unknown_subject_or_session(cfg):
    with pytest.raises(DataValidationError, match="Unknown subject"):
        recording_path(cfg, 99, "T")
    with pytest.raises(DataValidationError, match="Unknown session"):
        recording_path(cfg, 1, "X")


# ---- conversion of an in-memory Raw ---------------------------------------------------
def test_recording_structure(synth):
    rec, _ = synth
    assert rec.data.shape[0] == 22 and rec.sfreq == 250.0
    assert rec.ch_names[7] == "C3" and rec.ch_names[11] == "C4"
    assert rec.subject == 1 and rec.session == "T"
    assert rec.duration_s == pytest.approx(rec.n_samples / 250.0)


def test_missing_segments_become_nan_and_are_reported(synth):
    rec, _ = synth
    assert len(rec.missing_segments) == 1                       # 2 runs -> 1 gap
    a, b = rec.missing_segments[0]
    assert b - a == 100
    assert np.isnan(rec.data[:, a:b]).all()
    assert np.isfinite(np.delete(rec.data, np.s_[a:b], axis=1)).all()
    assert any("missing segment" in w for w in rec.report.warnings)


def test_find_missing_segments_simple():
    x = np.zeros((2, 20))
    x[:, 5:8] = OVERFLOW_UV
    x[0, 12] = OVERFLOW_UV          # only one channel: not a documented gap
    assert find_missing_segments(x) == [(5, 8)]
    assert find_missing_segments(np.zeros((2, 5))) == []


def test_wrong_sampling_rate_rejected(cfg):
    raw, _ = make_synthetic_raw(cfg)
    raw.resample(125.0)
    with pytest.raises(DataValidationError, match="Unexpected sampling"):
        recording_from_raw(raw, cfg, 1, "T")


def test_wrong_channel_count_rejected(cfg):
    raw, _ = make_synthetic_raw(cfg)
    raw.drop_channels(["EOG-left"])
    with pytest.raises(DataValidationError, match="Expected 25"):
        recording_from_raw(raw, cfg, 1, "T")


def test_anchor_channel_mismatch_rejected(cfg):
    raw, _ = make_synthetic_raw(cfg)
    raw.rename_channels({"EEG-C3": "EEG-Cx", "EEG-Cz": "EEG-C3"})
    raw.rename_channels({"EEG-Cx": "EEG-Cz"})          # swap so positions 7 and 9 are exchanged
    with pytest.raises(DataValidationError, match="position"):
        recording_from_raw(raw, cfg, 1, "T")


def test_unknown_event_code_rejected(cfg):
    import mne

    raw, _ = make_synthetic_raw(cfg)
    ann = raw.annotations
    raw.set_annotations(ann + mne.Annotations([1.0], [0.0], ["4242"]))
    with pytest.raises(DataValidationError, match="Unknown event"):
        recording_from_raw(raw, cfg, 1, "T")


def test_nonnumeric_event_rejected(cfg):
    import mne

    raw, _ = make_synthetic_raw(cfg)
    raw.set_annotations(raw.annotations + mne.Annotations([1.0], [0.0], ["BAD_BLINK"]))
    with pytest.raises(DataValidationError, match="Non-numeric"):
        recording_from_raw(raw, cfg, 1, "T")


def test_partial_nan_rejected(cfg):
    raw, _ = make_synthetic_raw(cfg)
    raw._data[2, 1000] = np.nan
    with pytest.raises(DataValidationError, match="non-finite"):
        recording_from_raw(raw, cfg, 1, "T")


def test_external_label_file_errors(cfg, tmp_path):
    cfg["dataset"]["labels_dir"] = str(tmp_path)
    with pytest.raises(DataValidationError, match="not found"):
        load_external_labels(cfg, 1, "E")
    (tmp_path / "A01E.mat").write_bytes(b"garbage")
    with pytest.raises(DataValidationError, match="Could not read"):
        load_external_labels(cfg, 1, "E")
    sio.savemat(tmp_path / "A02E.mat", {"other": np.ones(3)})
    with pytest.raises(DataValidationError, match="classlabel"):
        load_external_labels(cfg, 2, "E")
    sio.savemat(tmp_path / "A03E.mat", {"classlabel": np.array([[1], [2], [9]])})
    with pytest.raises(DataValidationError, match="unknown"):
        load_external_labels(cfg, 3, "E")
    sio.savemat(tmp_path / "A04E.mat", {"classlabel": np.array([[1], [2], [3], [4]])})
    assert load_external_labels(cfg, 4, "E").tolist() == [1, 2, 3, 4]
