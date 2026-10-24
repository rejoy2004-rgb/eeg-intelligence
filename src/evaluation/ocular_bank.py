"""Epoch bank for the ocular-robust evaluation: the SAME primary trials, with and without EOG regression.

For every recording (participant x session) and for both filter bands (``bp`` = 4-40 Hz, ``csp`` = 8-30 Hz) it produces:

    base      EEG epochs exactly as in final_v1 (no ocular handling)
    own       EEG minus B_own @ EOG, B_own fitted on that recording's OWN eye-movement calibration block
              (unlabeled, recorded before the trials, never overlapping them) -> target-calibrated variant
    strict_a  protocol A, "train-only": every trial of participant s is corrected with the coefficients of s's TRAINING session (T)
    strict_b  protocol B, "train-only": every trial of participant s is corrected with coefficients averaged over the calibration
              blocks of the OTHER participants (the training participants of the fold in which s is held out)

``strict_*`` therefore use no data of the held-out evaluation session/participant at all; ``own`` does use the target's
calibration block (but not its trials, not its labels).
Everything is aligned with the primary trial order, so paired comparisons use identical test trials by construction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import load_external_labels, load_recording
from src.signal.epoching import TrialSet, build_dual_band_trials, extract_epochs
from src.signal.ocular import (apply_eog_regression, calibration_slice, eog_features, filter_eog, fit_eog_regression, load_eog)
from src.signal.preprocessing import preprocess_recording

FRONTAL = ("Fz", "FC3", "FC1", "FCz", "FC2", "FC4")
BANDS = ("bp", "csp")


@dataclass
class OcularBank:
    bp: TrialSet                       # base epochs, 4-40 Hz (carries meta / labels / channel names)
    csp: TrialSet                      # base epochs, 8-30 Hz
    arrays: dict                       # {"own_bp", "own_csp", "strict_a_bp", ... } float32 (n, 22, 500)
    eog: np.ndarray                    # (n, 6) EOG-only features
    coefficients: dict                 # {(subject, session, band): B (22, 3)}
    diagnostics: pd.DataFrame

    def trial_set(self, variant: str, band: str) -> TrialSet:
        base = self.bp if band == "bp" else self.csp
        if variant == "base":
            return base
        return TrialSet(self.arrays[f"{variant}_{band}"], base.y, base.meta, base.rejections, base.sfreq, base.ch_names, dict(base.summary))


def _bands(cfg):
    return {"bp": tuple(cfg.preprocessing["bandpass_hz"]), "csp": tuple(cfg.preprocessing["csp_bandpass_hz"])}


def build_ocular_bank(cfg, subjects=None, sessions=None, log=print) -> OcularBank:
    subjects = list(subjects) if subjects is not None else list(cfg.dataset["subjects"])
    sessions = list(sessions) if sessions is not None else list(cfg.dataset["sessions"])
    pre = cfg.preprocessing
    bands = _bands(cfg)
    frontal_idx = [cfg.dataset["eeg_channels"].index(c) for c in FRONTAL]
    other_idx = [i for i in range(len(cfg.dataset["eeg_channels"])) if i not in frontal_idx]
    tmin, tmax, order = pre["epoch_tmin_s"], pre["epoch_tmax_s"], pre["filter_order"]

    prim = {"bp": [], "csp": []}
    arrays = {f"{v}_{b}": [] for v in ("own", "strict_a", "strict_b") for b in BANDS}
    eog_feats, coefs, diag = [], {}, []

    # ---------------- pass 1: base epochs, own-corrected epochs, coefficients ----------------
    for s in subjects:
        for sess in sessions:
            rec = load_recording(cfg, s, sess)
            bp, csp = build_dual_band_trials(rec, cfg, load_external_labels(cfg, s, sess))
            prim["bp"].append(bp)
            prim["csp"].append(csp)
            cues = bp.meta["cue_sample"].to_numpy()
            eog = load_eog(rec)
            eog_feats.append(eog_features(eog, cues, rec.sfreq, tmin, tmax))
            block = calibration_slice(rec)
            for name, band in bands.items():
                eeg_f, eog_f = preprocess_recording(rec, cfg, band), filter_eog(eog, rec.sfreq, band, order)
                B, d = fit_eog_regression(eeg_f, eog_f, block)
                coefs[(s, sess, name)] = B
                ep, why = extract_epochs(apply_eog_regression(eeg_f, eog_f, B), rec.sfreq, cues, tmin, tmax)
                if any(why):
                    raise RuntimeError("corrected epochs invalid although base epochs were valid")
                arrays[f"own_{name}"].append(ep.astype(np.float32))
                base_ep = (bp if name == "bp" else csp).X
                var_ratio = (ep.var(axis=-1) / base_ep.var(axis=-1)).mean(axis=0)          # per channel, mean over trials
                diag.append({"subject": s, "session": sess, "band": name, "calibration_samples": d["n_samples"],
                             "calibration_seconds": d["n_samples"] / rec.sfreq,
                             "r2_in_sample_frontal": float(d["r2_in_sample"][frontal_idx].mean()), "r2_in_sample_other": float(d["r2_in_sample"][other_idx].mean()),
                             "r2_split_half_frontal": float(d["r2_split_half"][frontal_idx].mean()), "r2_split_half_other": float(d["r2_split_half"][other_idx].mean()),
                             "task_variance_ratio_frontal": float(var_ratio[frontal_idx].mean()), "task_variance_ratio_other": float(var_ratio[other_idx].mean())})
            log(f"  pass 1: S{s:02d}{sess}")
            del rec, eog

    # ---------------- pass 2: train-only coefficient variants ----------------
    for s in subjects:
        for sess in sessions:
            rec = load_recording(cfg, s, sess)
            cues = next(p for p in prim["bp"] if p.meta["subject"].iloc[0] == s and p.meta["session"].iloc[0] == sess).meta["cue_sample"].to_numpy()
            eog = load_eog(rec)
            for name, band in bands.items():
                eeg_f, eog_f = preprocess_recording(rec, cfg, band), filter_eog(eog, rec.sfreq, band, order)
                B_a = coefs[(s, "T", name)] if "T" in sessions else coefs[(s, sess, name)]
                others = [B for (ss, se, nm), B in coefs.items() if nm == name and ss != s]
                B_b = np.mean(others, axis=0)
                for tag, B in (("strict_a", B_a), ("strict_b", B_b)):
                    ep, why = extract_epochs(apply_eog_regression(eeg_f, eog_f, B), rec.sfreq, cues, tmin, tmax)
                    if any(why):
                        raise RuntimeError("corrected epochs invalid although base epochs were valid")
                    arrays[f"{tag}_{name}"].append(ep.astype(np.float32))
            log(f"  pass 2: S{s:02d}{sess}")
            del rec, eog

    bp, csp = TrialSet.concat(prim["bp"]), TrialSet.concat(prim["csp"])
    out_arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    eog_arr = np.concatenate(eog_feats)
    if not np.isfinite(eog_arr).all():
        raise RuntimeError("EOG features contain NaN for kept trials")
    return OcularBank(bp, csp, out_arrays, eog_arr, coefs, pd.DataFrame(diag))


# ---------------------------------------------------------------- cache (data/cache is git-ignored)
def cache_key(*files: Path) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(Path(f).read_bytes())
    return h.hexdigest()


def save_bank(bank: OcularBank, path: Path, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"arr_{k}": v for k, v in bank.arrays.items()}
    flat.update(base_bp=bank.bp.X, base_csp=bank.csp.X, eog=bank.eog, y=bank.bp.y)   # float64: baseline must be bit-identical to final_v1
    flat.update({f"coef_{s}_{se}_{b}": B for (s, se, b), B in bank.coefficients.items()})
    np.savez(path, key=np.array(key), **flat)
    bank.bp.meta.to_csv(path.with_suffix(".meta.csv"), index=False)
    bank.diagnostics.to_csv(path.with_suffix(".diag.csv"), index=False)
    (path.with_suffix(".json")).write_text(json.dumps({"ch_names": bank.bp.ch_names, "sfreq": bank.bp.sfreq}), encoding="utf-8")


def load_bank(path: Path, key: str) -> OcularBank | None:
    if not path.exists() or not path.with_suffix(".meta.csv").exists():
        return None
    z = np.load(path, allow_pickle=False)
    if str(z["key"]) != key:
        return None
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    meta = pd.read_csv(path.with_suffix(".meta.csv"))
    y = z["y"]
    empty = pd.DataFrame(columns=["subject", "session", "trial", "reason"])
    mk = lambda X: TrialSet(X, y, meta, empty, info["sfreq"], info["ch_names"])
    arrays = {k[4:]: z[k] for k in z.files if k.startswith("arr_")}
    coefs = {}
    for k in z.files:
        if k.startswith("coef_"):
            _, s, se, b = k.split("_")
            coefs[(int(s), se, b)] = z[k]
    return OcularBank(mk(z["base_bp"]), mk(z["base_csp"]), arrays, z["eog"], coefs,
                      pd.read_csv(path.with_suffix(".diag.csv")))
