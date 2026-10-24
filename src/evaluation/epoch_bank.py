"""An "epoch bank": several (band, window) versions of the SAME trials.

Alternative filter bands / epoch windows must be compared on exactly the trials used in the primary analysis,
otherwise differences could come from different trial sets. The bank therefore:
  * takes the trial list from the primary pipeline (``build_dual_band_trials``: expert-artifact and 200 uV rules
    applied once, with the primary band/window) and never re-screens for other windows;
  * cuts every requested (band, tmin, tmax) from the same continuous, gap-aware filtered signal;
  * raises if any requested window would leave the recording or overlap a missing-data gap.
Arrays are stored as float32 to bound memory (about 100 MB per 2 s window for all 2346 trials).
"""
from __future__ import annotations

import numpy as np

from src.data.dataset import load_external_labels, load_recording
from src.data.validation import DataValidationError
from src.signal.epoching import TrialSet, build_dual_band_trials, extract_epochs
from src.signal.preprocessing import preprocess_recording

PRIMARY = "primary"


def build_epoch_bank(cfg, specs: dict, subjects=None, sessions=None, log=print):
    """specs: {name: (band_hz (lo, hi), tmin_s, tmax_s)}. Returns (bank {name: (n, C, T) float32}, primary TrialSet)."""
    subjects = list(subjects) if subjects is not None else list(cfg.dataset["subjects"])
    sessions = list(sessions) if sessions is not None else list(cfg.dataset["sessions"])
    parts = {name: [] for name in specs}
    primaries = []
    for s in subjects:
        for sess in sessions:
            rec = load_recording(cfg, s, sess)
            bp, _ = build_dual_band_trials(rec, cfg, load_external_labels(cfg, s, sess))
            primaries.append(bp)
            cues = bp.meta["cue_sample"].to_numpy()
            cache = {}
            for name, spec in specs.items():
                band, tmin, tmax, *opt = spec                       # optional 4th element: {"causal": True}
                causal = bool(opt and opt[0].get("causal"))
                key = (tuple(band), causal)
                if key not in cache:
                    cache[key] = preprocess_recording(rec, cfg, key[0], causal=causal)
                ep, reasons = extract_epochs(cache[key], rec.sfreq, cues, tmin, tmax)
                bad = [r for r in reasons if r]
                if bad:
                    raise DataValidationError(f"S{s:02d}{sess} spec '{name}': {len(bad)} epochs invalid ({set(bad)})")
                parts[name].append(ep.astype(np.float32))
            log(f"  bank: S{s:02d}{sess} {len(cues)} trials")
    bank = {name: np.concatenate(p) for name, p in parts.items()}
    return bank, TrialSet.concat(primaries)


def with_epochs(ts: TrialSet, X: np.ndarray) -> TrialSet:
    """A TrialSet that shares labels/metadata with ``ts`` but holds different epoch data (e.g. another window)."""
    return TrialSet(np.asarray(X), ts.y, ts.meta, ts.rejections, ts.sfreq, ts.ch_names, dict(ts.summary))
