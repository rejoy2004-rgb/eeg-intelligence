"""Split-integrity tests: no participant / trial overlap where the protocol forbids it."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.splits import SplitError, assert_no_overlap, cross_session_split, loso_splits, split_to_json


@pytest.fixture
def meta_y():
    rows = []
    for s in range(1, 6):
        for sess in ("T", "E"):
            for t in range(24):
                rows.append({"subject": s, "session": sess, "trial": t, "run": t // 8})
    meta = pd.DataFrame(rows)
    y = np.tile([0, 1], len(meta) // 2)
    return meta, y


def test_loso_partitions_are_disjoint_complete_and_subject_pure(meta_y):
    meta, y = meta_y
    splits = loso_splits(meta, y, n_val_subjects=1)
    assert len(splits) == 5
    for sp in splits:
        fit = np.setdiff1d(sp.train_idx, sp.val_idx)
        parts = [fit, sp.val_idx, sp.test_idx]
        assert sum(len(p) for p in parts) == len(meta)                 # complete
        assert len(set(np.concatenate(parts))) == len(meta)            # disjoint
        subj = [set(meta.iloc[p]["subject"]) for p in parts]
        assert subj[0].isdisjoint(subj[1]) and subj[0].isdisjoint(subj[2]) and subj[1].isdisjoint(subj[2])
        assert len(subj[2]) == 1                                       # exactly one held-out participant
        assert set(meta.iloc[sp.test_idx]["session"]) == {"T", "E"}     # all of its sessions are held out


def test_each_subject_is_held_out_exactly_once(meta_y):
    meta, y = meta_y
    held = [int(meta.iloc[sp.test_idx[0]]["subject"]) for sp in loso_splits(meta, y, 2)]
    assert sorted(held) == [1, 2, 3, 4, 5]


def test_loso_validation_rule_is_deterministic(meta_y):
    meta, y = meta_y
    a = loso_splits(meta, y, 2)
    b = loso_splits(meta, y, 2)
    assert all(np.array_equal(x.val_idx, z.val_idx) for x, z in zip(a, b))


def test_loso_argument_errors(meta_y):
    meta, y = meta_y
    with pytest.raises(SplitError, match="at least 3"):
        loso_splits(meta[meta["subject"] <= 2], y[:len(meta[meta["subject"] <= 2])])
    with pytest.raises(SplitError, match="n_val_subjects"):
        loso_splits(meta, y, n_val_subjects=4)
    with pytest.raises(SplitError, match="columns"):
        loso_splits(meta.drop(columns=["session"]), y)
    with pytest.raises(SplitError, match="No trials"):
        loso_splits(meta.iloc[:0], y[:0])


def test_cross_session_split_integrity(meta_y):
    meta, y = meta_y
    sp = cross_session_split(meta, y, subject=3)
    train, test = meta.iloc[sp.train_idx], meta.iloc[sp.test_idx]
    assert set(train["session"]) == {"T"} and set(test["session"]) == {"E"}
    assert set(train["subject"]) == {3} and set(test["subject"]) == {3}
    val = meta.iloc[sp.val_idx]
    assert set(val["run"]) == {2} and len(val) == 8                    # last run of the TRAINING session only
    assert set(sp.val_idx) <= set(sp.train_idx) and not set(sp.val_idx) & set(sp.test_idx)


def test_cross_session_missing_data(meta_y):
    meta, y = meta_y
    with pytest.raises(SplitError, match="lacks trials"):
        cross_session_split(meta, y, subject=99)
    one_run = meta[(meta["session"] == "E") | (meta["run"] == 0)]
    with pytest.raises(SplitError, match="single run"):
        cross_session_split(one_run.reset_index(drop=True), y[:len(one_run)], subject=1)


def test_overlap_detector_catches_violations(meta_y):
    meta, _ = meta_y
    a = np.flatnonzero(meta["subject"] <= 2)
    b = np.flatnonzero(meta["subject"] >= 2)                           # subject 2 is in both
    with pytest.raises(SplitError):
        assert_no_overlap(meta, a, b, by="subject")
    with pytest.raises(SplitError, match="Row indices"):
        assert_no_overlap(meta, [0, 1, 2], [2, 3], by="trial_id")
    # same (subject, session, trial) under different row numbers (e.g. duplicated rows)
    dup = pd.concat([meta, meta.iloc[:3]], ignore_index=True)
    with pytest.raises(SplitError, match="same"):
        assert_no_overlap(dup, [0, 1, 2], [len(meta), len(meta) + 1], by="trial_id")
    with pytest.raises(SplitError, match="Unknown overlap"):
        assert_no_overlap(meta, [0], [1], by="nonsense")
    assert_no_overlap(meta, np.flatnonzero(meta["subject"] == 1), np.flatnonzero(meta["subject"] == 2), by="subject")


def test_manifest_records_subject_ids_and_counts(meta_y):
    meta, y = meta_y
    sp = loso_splits(meta, y, 1)[0]
    m = split_to_json(sp)
    assert m["test"]["subjects"] == [1] and m["test"]["n_trials"] == 48
    assert m["cnn_validation_part"]["subjects"] == [2]
    assert m["train_all"]["n_trials"] == m["cnn_fit_part"]["n_trials"] + m["cnn_validation_part"]["n_trials"]
    assert sum(m["test"]["class_counts"].values()) == m["test"]["n_trials"]
