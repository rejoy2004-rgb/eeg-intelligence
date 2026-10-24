"""Tools for investigating the pre-cue negative control.

The control decodes the class from the 1 s BEFORE the cue (cue-1.5 ... cue-0.5 s), when the participant cannot yet know the class.
A result clearly above 0.5 would point to a confound (information from after the cue, trial-order structure, carry-over from the
previous trial, ...). These helpers examine the individual candidate explanations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfilt, sosfiltfilt

from src.data.dataset import Recording


def window_bounds(cue_sample: int, sfreq: float, tmin: float, tmax: float) -> tuple[int, int]:
    return int(cue_sample) + int(round(tmin * sfreq)), int(cue_sample) + int(round(tmax * sfreq))


def timing_audit(rec: Recording, trials: pd.DataFrame, tmin: float = -1.5, tmax: float = -0.5, kept_trials=None) -> dict:
    """Check where the pre-cue window lies relative to the trial start, the cue, the previous trial and the data gaps.

    ``trials``: output of ``locate_trials`` (columns trial, trial_start, cue_sample, run). Returns summary numbers in seconds.
    """
    sf = rec.sfreq
    t = trials if kept_trials is None else trials[trials["trial"].isin(kept_trials)]
    starts = trials.set_index("trial")["trial_start"]
    after_start, before_cue, after_prev_imagery_end, overlap_gap = [], [], [], 0
    for _, r in t.iterrows():
        a, b = window_bounds(r.cue_sample, sf, tmin, tmax)
        after_start.append((a - r.trial_start) / sf)
        before_cue.append((r.cue_sample - b) / sf)
        if r.trial > 0 and trials.loc[trials["trial"] == r.trial - 1, "run"].iloc[0] == r.run:
            after_prev_imagery_end.append((a - (starts[r.trial - 1] + int(round(6.0 * sf)))) / sf)      # imagery ends 6 s after trial start
        overlap_gap += any(not (b <= g0 or a >= g1) for g0, g1 in rec.missing_segments)
    q = lambda v: [float(np.min(v)), float(np.median(v)), float(np.max(v))] if len(v) else [None] * 3
    return {"n_trials": int(len(t)), "window_start_after_trial_start_s[min,median,max]": q(after_start),
            "window_end_before_cue_s[min,median,max]": q(before_cue),
            "window_start_after_previous_imagery_end_s[min,median,max]": q(after_prev_imagery_end),
            "windows_overlapping_missing_gap": int(overlap_gap),
            "windows_crossing_trial_start_or_cue": int(sum(1 for v in after_start if v < 0) + sum(1 for v in before_cue if v < 0))}


def label_sequence_stats(labels: dict) -> dict:
    """Structure of the 4-class cue sequences. ``labels``: {(subject, session): array of 288 class labels 1-4}.

    Reports (i) lag-1 transition matrix of the left/right classes and a chi-square test, (ii) agreement of the sequences between
    sessions of one participant and between participants (chance = 0.25 for four equiprobable classes).
    """
    keys = sorted(labels)
    trans = np.zeros((2, 2))
    for k in keys:
        s = labels[k]
        for a, b in zip(s[:-1], s[1:]):
            if a in (1, 2) and b in (1, 2):
                trans[a - 1, b - 1] += 1
    chi = stats.chi2_contingency(trans + 1e-9, correction=False)
    n = len(keys)
    agree = np.full((n, n), np.nan)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i != j:
                agree[i, j] = float((labels[a] == labels[b]).mean())
    same_part = [agree[i, j] for i, a in enumerate(keys) for j, b in enumerate(keys) if a[0] == b[0] and i < j]
    diff_part = [agree[i, j] for i, a in enumerate(keys) for j, b in enumerate(keys) if a[0] != b[0] and i < j]
    return {"lag1_transition_counts_[prev L,R][next L,R]": trans.tolist(),
            "P(next=same | prev in {L,R}, next in {L,R})": float((trans[0, 0] + trans[1, 1]) / trans.sum()),
            "chi2_independence_p": float(chi[1]),
            "position_agreement_between_sessions_of_same_participant[min,mean,max]": [float(np.min(same_part)), float(np.mean(same_part)), float(np.max(same_part))],
            "position_agreement_between_different_participants[min,mean,max]": [float(np.min(diff_part)), float(np.mean(diff_part)), float(np.max(diff_part))],
            "chance_agreement": 0.25, "n_recordings": n}


def impulse_leakage_profile(band, sfreq: float = 250.0, order: int = 4, margin_s: float = 0.5, n: int = 12000) -> dict:
    """How far BEFORE an event does a band-pass filter 'smear' it? An impulse at sample n0 (the cue) is filtered; we report the share of
    response energy located earlier than ``n0 - margin`` (where the pre-cue window ends) for the zero-phase and the causal filter."""
    sos = butter(order, list(band), btype="bandpass", fs=sfreq, output="sos")
    n0 = n // 2
    x = np.zeros(n)
    x[n0] = 1.0
    cut = n0 - int(round(margin_s * sfreq))
    out = {}
    for name, y in (("zero_phase", sosfiltfilt(sos, x)), ("causal", sosfilt(sos, x))):
        e = y**2
        out[name] = {"energy_fraction_before_pre_window_end": float(e[:cut].sum() / e.sum()),
                     "max_abs_before_pre_window_end_relative_to_peak": float(np.abs(y[:cut]).max() / np.abs(y).max()),
                     "earliest_sample_with_1pct_of_peak_s_before_event": float((n0 - np.flatnonzero(np.abs(y) > 0.01 * np.abs(y).max())[0]) / sfreq)}
    return out


def previous_trial_labels(meta: pd.DataFrame, labels: dict, run_of: dict) -> np.ndarray:
    """Class (1-4) of the preceding trial of the same run for every row of ``meta`` (NaN for the first trial of a run).

    ``labels``: {(subject, session): 288 labels}; ``run_of``: {(subject, session): array of run id per trial index}.
    """
    out = np.full(len(meta), np.nan)
    for i, (s, sess, t) in enumerate(zip(meta["subject"], meta["session"], meta["trial"])):
        key = (int(s), str(sess))
        if t > 0 and run_of[key][t - 1] == run_of[key][t]:
            out[i] = labels[key][t - 1]
    return out
