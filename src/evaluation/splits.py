"""Leakage-resistant data splits.

Three different scientific questions - never mix their results:

1. ``cross_session`` (within participant, unseen session)
      Train on a participant's session T, test on the same participant's session E (recorded on a
      different day). Asks: "does a model calibrated for ONE person work on that person later?"
2. ``loso`` (unseen participant)
      Leave-One-Subject-Out: train on 8 participants (both sessions), test on the 9th.
      Asks: "does a model transfer to a person it has never seen?"
3. Within-session random k-fold would answer a third (easier, optimistic) question. We do not
   report it as evidence for 1 or 2 and do not implement it as an evaluation protocol.

Every split is a ``Split`` object with the participant IDs and trial counts of each partition, and
``assert_no_overlap`` raises if any forbidden overlap exists (used in the code path AND in tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class SplitError(ValueError):
    """Raised when a split would violate the evaluation protocol."""


@dataclass
class Split:
    protocol: str
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    val_idx: np.ndarray | None = None       # optional validation partition (for the CNN)
    manifest: dict = field(default_factory=dict)


def _check_meta(meta: pd.DataFrame, cols=("subject", "session", "trial")) -> None:
    missing = [c for c in cols if c not in meta.columns]
    if missing:
        raise SplitError(f"Trial metadata lacks required columns: {missing}")
    if len(meta) == 0:
        raise SplitError("No trials to split")


def _describe(meta: pd.DataFrame, idx: np.ndarray, y: np.ndarray) -> dict:
    sub = meta.iloc[idx]
    return {
        "subjects": sorted(int(s) for s in sub["subject"].unique()),
        "sessions": sorted(str(s) for s in sub["session"].unique()),
        "n_trials": int(len(idx)),
        "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y[idx], return_counts=True))},
    }


def assert_no_overlap(meta: pd.DataFrame, a_idx, b_idx, by: str = "trial_id") -> None:
    """Raise SplitError if two index sets share a trial (``by='trial_id'``) or a participant (``by='subject'``)."""
    a_idx, b_idx = np.asarray(a_idx), np.asarray(b_idx)
    if set(a_idx.tolist()) & set(b_idx.tolist()):
        raise SplitError("Row indices overlap between partitions")
    if by == "subject":
        shared = set(meta.iloc[a_idx]["subject"]) & set(meta.iloc[b_idx]["subject"])
        if shared:
            raise SplitError(f"Participants appear in both partitions: {sorted(shared)}")
    elif by == "trial_id":
        key = lambda idx: set(zip(meta.iloc[idx]["subject"], meta.iloc[idx]["session"], meta.iloc[idx]["trial"]))
        if key(a_idx) & key(b_idx):
            raise SplitError("The same (subject, session, trial) appears in both partitions")
    else:
        raise SplitError(f"Unknown overlap criterion '{by}'")


def cross_session_split(meta: pd.DataFrame, y: np.ndarray, subject: int, train_session="T", test_session="E") -> Split:
    """Participant ``subject``: train on one session, test on the other. Needs an inner val split from runs."""
    _check_meta(meta)
    is_s = meta["subject"].to_numpy() == subject
    tr = np.flatnonzero(is_s & (meta["session"].to_numpy() == train_session))
    te = np.flatnonzero(is_s & (meta["session"].to_numpy() == test_session))
    if len(tr) == 0 or len(te) == 0:
        raise SplitError(f"Subject {subject} lacks trials in session {train_session} or {test_session}")
    assert_no_overlap(meta, tr, te, by="trial_id")
    # Validation = last run of the TRAINING session (temporal block, never the test session)
    runs = meta.iloc[tr]["run"].to_numpy()
    last = runs.max()
    val = tr[runs == last]
    tr_fit = tr[runs != last]
    if len(tr_fit) == 0:
        raise SplitError("Training session has a single run; cannot hold out a validation run")
    assert_no_overlap(meta, tr_fit, val, by="trial_id")
    assert_no_overlap(meta, val, te, by="trial_id")
    return _with_val(meta, y, "cross_session", f"S{subject:02d}", tr_fit, val, te)


def _with_val(meta, y, protocol, name, fit_idx, val_idx, test_idx) -> Split:
    """Build a Split where train = fit + val (for classical models using inner CV) and CNN uses fit/val."""
    train = np.concatenate([fit_idx, val_idx])
    return Split(
        protocol, name, train_idx=np.sort(train), test_idx=test_idx, val_idx=val_idx,
        manifest={
            "train_all": _describe(meta, train, y),
            "cnn_fit_part": _describe(meta, fit_idx, y),
            "cnn_validation_part": _describe(meta, val_idx, y),
            "test": _describe(meta, test_idx, y),
        },
    )


def loso_splits(meta: pd.DataFrame, y: np.ndarray, n_val_subjects: int = 2) -> list[Split]:
    """Leave-One-Subject-Out. The CNN validation partition consists of whole OTHER participants.

    For test subject s, validation subjects are the next ``n_val_subjects`` subjects (cyclically) in
    sorted order - a deterministic rule that does not look at any data or results.
    """
    _check_meta(meta)
    subjects = sorted(int(s) for s in meta["subject"].unique())
    if len(subjects) < 3:
        raise SplitError("LOSO with validation needs at least 3 participants")
    if not 1 <= n_val_subjects <= len(subjects) - 2:
        raise SplitError("n_val_subjects must leave at least one training participant")
    subj = meta["subject"].to_numpy()
    out = []
    for i, s in enumerate(subjects):
        others = [x for x in subjects if x != s]
        val_sub = [subjects[(i + k) % len(subjects)] for k in range(1, n_val_subjects + 1)]
        fit_sub = [x for x in others if x not in val_sub]
        test_idx = np.flatnonzero(subj == s)
        fit_idx = np.flatnonzero(np.isin(subj, fit_sub))
        val_idx = np.flatnonzero(np.isin(subj, val_sub))
        for a, b in ((fit_idx, test_idx), (val_idx, test_idx), (fit_idx, val_idx)):
            assert_no_overlap(meta, a, b, by="subject")
        out.append(_with_val(meta, y, "loso", f"heldout_S{s:02d}", fit_idx, val_idx, test_idx))
    return out


def split_to_json(split: Split) -> dict:
    return {"protocol": split.protocol, "name": split.name, **split.manifest}
