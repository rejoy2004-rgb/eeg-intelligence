"""Event-aligned trial extraction for BCI Competition IV 2a.

Protocol recap (official description, Fig. 2 / Table 2):
    t = 0 s   trial start (event 768), fixation cross + beep
    t = 2 s   cue onset (769 left, 770 right, 771 foot, 772 tongue; 783 = class hidden in E files)
    t = 3.25  cue disappears (cue lasts 1.25 s)
    t = 6 s   fixation cross disappears -> end of motor imagery period

Our epoch is anchored to CUE ONSET and spans ``[epoch_tmin_s, epoch_tmax_s]`` after the cue
(default 0.5 - 2.5 s). Reasons: (1) skip the first 0.5 s, dominated by the visual evoked response
to the cue; (2) stay inside the imagery period; (3) 2 s gives 1 Hz Welch resolution with 250-sample
segments. The window is a documented a-priori choice, not tuned on test data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.dataset import Recording
from src.data.validation import DataValidationError, ValidationReport, validate_labels
from src.signal.preprocessing import preprocess_recording
from src.signal.quality import flag_high_amplitude

CUE_CODES = (769, 770, 771, 772, 783)


@dataclass
class TrialSet:
    """Epoched trials with aligned labels and metadata.

    X : (n_trials, n_channels, n_times) float64, microvolts
    y : (n_trials,) int; 0 = left hand, 1 = right hand, -1 = label unknown (prediction only)
    meta : one row per trial (same order as X and y)
    rejections : one row per trial that was removed, with the reason
    """

    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    rejections: pd.DataFrame
    sfreq: float
    ch_names: list[str]
    summary: dict = field(default_factory=dict)

    def __post_init__(self):
        if not (len(self.X) == len(self.y) == len(self.meta)):
            raise DataValidationError(
                f"Misaligned trial set: X={len(self.X)}, y={len(self.y)}, meta={len(self.meta)}"
            )

    @property
    def n_trials(self) -> int:
        return len(self.X)

    def class_counts(self) -> dict[int, int]:
        vals, cnt = np.unique(self.y, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnt)}

    def select(self, mask) -> "TrialSet":
        mask = np.asarray(mask)                       # boolean mask or integer index array
        idx = np.flatnonzero(mask) if mask.dtype == bool else mask
        return TrialSet(self.X[idx], self.y[idx], self.meta.iloc[idx].reset_index(drop=True),
                        self.rejections, self.sfreq, self.ch_names, dict(self.summary))

    @staticmethod
    def concat(sets: list["TrialSet"]) -> "TrialSet":
        if not sets:
            raise DataValidationError("Cannot concatenate an empty list of trial sets")
        if len({s.sfreq for s in sets}) != 1 or len({tuple(s.ch_names) for s in sets}) != 1:
            raise DataValidationError("Trial sets differ in sampling rate or channels")
        if len({s.X.shape[1:] for s in sets}) != 1:
            raise DataValidationError("Trial sets differ in epoch shape")
        return TrialSet(
            np.concatenate([s.X for s in sets]), np.concatenate([s.y for s in sets]),
            pd.concat([s.meta for s in sets], ignore_index=True),
            pd.concat([s.rejections for s in sets], ignore_index=True),
            sets[0].sfreq, sets[0].ch_names,
            {"n_sets": len(sets)},
        )


def locate_trials(rec: Recording, cfg, external_labels: np.ndarray | None = None) -> pd.DataFrame:
    """Build the full table of trials (all four classes) from the event markers.

    Columns: trial, run, trial_start, cue_sample, cue_code, label4 (1-4 or NaN), expert_artifact.
    Raises ``DataValidationError`` if the markers are inconsistent with the official protocol.
    """
    ev_cfg = cfg.dataset["events"]
    ev = rec.events
    starts = ev.loc[ev["code"] == ev_cfg["trial_start"], "sample"].to_numpy()
    if len(starts) == 0:
        raise DataValidationError("No trial-start events (768) found; this does not look like a 2a recording")
    cues = ev[ev["code"].isin(CUE_CODES)]
    if len(cues) != len(starts):
        raise DataValidationError(f"{len(starts)} trial starts but {len(cues)} cue events")
    cue_samples = cues["sample"].to_numpy()
    cue_codes = cues["code"].to_numpy()

    offset = int(round(cfg.dataset["cue_offset_s"] * rec.sfreq))
    if np.any(np.abs((cue_samples - starts) - offset) > 1):
        raise DataValidationError(f"Cue onsets are not {cfg.dataset['cue_offset_s']} s after trial starts")

    artifact_samples = set(ev.loc[ev["code"] == ev_cfg["rejected_trial"], "sample"].tolist())
    run_starts = ev.loc[ev["code"] == ev_cfg["run_start"], "sample"].to_numpy()
    run_idx = np.searchsorted(run_starts, starts, side="right") - 1
    _, run_idx = np.unique(run_idx, return_inverse=True)  # renumber to 0..n_runs-1 among trial runs

    label4 = np.where(cue_codes == ev_cfg["cue_unknown"], np.nan, cue_codes - ev_cfg["cue_left"] + 1).astype(float)
    if external_labels is not None:
        validate_labels(external_labels, [1, 2, 3, 4], "external labels")
        if len(external_labels) != len(starts):
            raise DataValidationError(f"{len(external_labels)} external labels for {len(starts)} trials")
        known = ~np.isnan(label4)
        if not np.array_equal(label4[known], external_labels[known].astype(float)):
            raise DataValidationError("External labels disagree with the labels stored in the recording's events")
        label4 = external_labels.astype(float)

    return pd.DataFrame(
        {
            "trial": np.arange(len(starts)),
            "run": run_idx,
            "trial_start": starts,
            "cue_sample": cue_samples,
            "cue_code": cue_codes,
            "label4": label4,
            "expert_artifact": [int(s) in artifact_samples for s in starts],
        }
    )


def extract_epochs(signal: np.ndarray, sfreq: float, anchors: np.ndarray, tmin: float, tmax: float):
    """Cut fixed-length epochs around anchor samples.

    Parameters
    ----------
    signal : (channels, samples) preprocessed signal (NaN where data are missing)
    anchors : (n,) sample index of the event (e.g. the cue)
    tmin, tmax : window in seconds relative to the anchor, tmin < tmax

    Returns
    -------
    epochs : (n, channels, n_times) - rows for invalid epochs are NaN
    reasons : list of length n; None for valid epochs, otherwise a string reason.
    """
    if signal.ndim != 2:
        raise DataValidationError(f"signal must be (channels, samples); got {signal.shape}")
    if tmax <= tmin:
        raise DataValidationError("tmax must be greater than tmin")
    start_off = int(round(tmin * sfreq))
    n_times = int(round((tmax - tmin) * sfreq))
    if n_times < 1:
        raise DataValidationError("Epoch window is shorter than one sample")
    anchors = np.asarray(anchors, dtype=int)
    epochs = np.full((len(anchors), signal.shape[0], n_times), np.nan)
    reasons: list[str | None] = []
    for i, a in enumerate(anchors):
        s = a + start_off
        e = s + n_times
        if s < 0 or e > signal.shape[1]:
            reasons.append("epoch extends beyond recording")
            continue
        chunk = signal[:, s:e]
        if not np.isfinite(chunk).all():
            reasons.append("epoch overlaps missing data")
            continue
        epochs[i] = chunk
        reasons.append(None)
    return epochs, reasons


def build_trial_set(
    rec: Recording,
    cfg,
    external_labels: np.ndarray | None = None,
    band_hz: tuple[float, float] | None = None,
    binary: bool = True,
    require_labels: bool = True,
) -> TrialSet:
    """Recording -> preprocessed, labelled, quality-screened epochs.

    binary=True keeps only left (0) / right (1) hand trials. Trials of other classes are *counted*
    in ``summary['n_other_class']`` but are not "rejections" (they are outside the task definition).
    require_labels=False (prediction mode) keeps trials whose class is unknown, with y = -1.
    """
    pre = cfg.preprocessing
    classes = cfg.dataset["task_classes"]
    trials = locate_trials(rec, cfg, external_labels)
    report = ValidationReport()

    # --- task selection ---
    n_all = len(trials)
    if binary and require_labels:
        in_task = trials["label4"].isin([classes["left"], classes["right"]])
    elif binary:
        in_task = trials["label4"].isin([classes["left"], classes["right"]]) | trials["label4"].isna()
    else:
        in_task = pd.Series(True, index=trials.index)
    n_other = int((~in_task).sum())
    if require_labels and trials["label4"].isna().any():
        raise DataValidationError(
            "Some trials have unknown class (cue 783) and no external labels were supplied. "
            "Provide the official label file or use require_labels=False for prediction."
        )
    trials = trials[in_task].reset_index(drop=True)

    # --- preprocess + cut epochs ---
    signal = preprocess_recording(rec, cfg, band_hz)
    epochs, reasons = extract_epochs(signal, rec.sfreq, trials["cue_sample"].to_numpy(),
                                     pre["epoch_tmin_s"], pre["epoch_tmax_s"])

    keep = np.ones(len(trials), dtype=bool)
    for i, reason in enumerate(reasons):
        if reason:
            keep[i] = False
            report.reject(subject=rec.subject, session=rec.session, trial=int(trials.loc[i, "trial"]), reason=reason)
    if pre["reject_expert_artifacts"]:
        for i in np.flatnonzero(trials["expert_artifact"].to_numpy() & keep):
            keep[i] = False
            report.reject(subject=rec.subject, session=rec.session, trial=int(trials.loc[i, "trial"]),
                          reason="expert artifact marker (1023)")
    high = np.zeros(len(trials), dtype=bool)
    if keep.any():
        high[keep] = flag_high_amplitude(epochs[keep], pre["reject_peak_to_peak_uv"])
    for i in np.flatnonzero(high):
        keep[i] = False
        report.reject(subject=rec.subject, session=rec.session, trial=int(trials.loc[i, "trial"]),
                      reason=f"peak-to-peak > {pre['reject_peak_to_peak_uv']:.0f} uV")

    # --- labels ---
    label4 = trials["label4"].to_numpy()
    y = np.full(len(trials), -1, dtype=int)
    y[label4 == classes["left"]] = 0
    y[label4 == classes["right"]] = 1
    trials = trials.assign(subject=rec.subject if rec.subject is not None else -1,
                           session=rec.session if rec.session is not None else "?")
    cols = ["subject", "session", "trial", "run", "cue_sample", "label4", "expert_artifact"]

    kept = TrialSet(
        X=epochs[keep], y=y[keep], meta=trials.loc[keep, cols].reset_index(drop=True),
        rejections=pd.DataFrame(report.rejections, columns=["subject", "session", "trial", "reason"]),
        sfreq=rec.sfreq, ch_names=list(rec.ch_names),
        summary={
            "n_trials_in_file": n_all,
            "n_other_class": n_other,
            "n_task_trials_before_rejection": len(trials),
            "n_rejected": int((~keep).sum()),
            "n_kept": int(keep.sum()),
            "rejection_counts": report.rejection_counts(),
            "epoch_window_s": [pre["epoch_tmin_s"], pre["epoch_tmax_s"]],
            "band_hz": list(band_hz if band_hz is not None else pre["bandpass_hz"]),
        },
    )
    return kept


def build_dual_band_trials(rec: Recording, cfg, external_labels=None, require_labels: bool = True):
    """Build the two epoch sets used by the models, guaranteed to contain the SAME trials.

    * ``bp``  : cfg.preprocessing.bandpass_hz      (band power features, CNN)
    * ``csp`` : cfg.preprocessing.csp_bandpass_hz  (CSP + LDA)
    A trial rejected by the amplitude rule in either band is removed from both (and recorded), so all
    models are compared on identical trials.
    """
    pre = cfg.preprocessing
    bp = build_trial_set(rec, cfg, external_labels, tuple(pre["bandpass_hz"]), require_labels=require_labels)
    csp = build_trial_set(rec, cfg, external_labels, tuple(pre["csp_bandpass_hz"]), require_labels=require_labels)
    key = lambda ts: set(ts.meta["trial"].tolist())
    common = key(bp) & key(csp)
    out = []
    for ts, other in ((bp, csp), (csp, bp)):
        dropped = sorted(key(ts) - common)
        keep = ts.meta["trial"].isin(common).to_numpy()
        new = ts.select(keep)
        if dropped:
            extra = pd.DataFrame(
                {"subject": new.meta["subject"].iloc[0] if len(new.meta) else ts.meta["subject"].iloc[0],
                 "session": ts.meta["session"].iloc[0], "trial": dropped,
                 "reason": "rejected in the other filter band (kept trial sets identical)"}
            )
            new.rejections = pd.concat([ts.rejections, extra], ignore_index=True)
        else:
            new.rejections = ts.rejections
        new.summary = dict(ts.summary, n_kept=int(keep.sum()))
        out.append(new)
    return out[0], out[1]
