"""Additional experiments of the research audit (all written to a NEW directory; final_v1 is never touched).

Usage: python scripts/run_audit_experiments.py <stage> [<stage> ...]     (outputs/audit_v2/)
Stages
    nested    nested model selection for CSP + LDA (24 configurations; selection by inner CV only)
    ablation  descriptive sensitivity of default CSP+LDA to CSP regularisation / LDA shrinkage / scaling
    ea        CSP + LDA after Euclidean alignment  -> uses UNLABELED target data (not zero-shot)
    erd       baseline-normalised (ERD) band power -> logistic regression and CNN
    perm      negative control: all four models with permuted TRAINING labels (should be at chance)
    cnnvar    two pre-specified CNN variants (smaller; stronger weight decay)  [exploratory, none is "selected"]
    pre       negative control: decode from the pre-cue window (should be at chance)
    layout    CNN on a measured-only 8x8 electrode-layout grid (no interpolation)
    pixel     logistic regression on the interpolated image pixels (is information lost by the representation?)
Every stage is exploratory unless stated otherwise in docs/RESEARCH_AUDIT.md; none of them changes final_v1.
"""
import copy
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.evaluation import csp_selection as cs
from src.evaluation.epoch_bank import PRIMARY, build_epoch_bank, with_epochs
from src.evaluation.experiments import run_protocol, save_experiment
from src.evaluation.metrics import compute_metrics
from src.features.alignment import align_by_group
from src.features.spatial import ImageScaler, LayoutGridBuilder, builder_from_config, images_from_epochs

cfg = load_config()
OUT = Path(cfg.resolve_path(cfg.evaluation["outputs_dir"])) / "audit_v2"
CACHE = Path(cfg.resolve_path("data/cache"))
PROTOCOLS = ("cross_session", "loso")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get_bank():
    """Bank of (band, window) epochs for the primary trials (rebuilt on every run, about 3 min)."""
    specs = cs.bank_specs()
    specs["erd"] = ((4.0, 40.0), -1.5, 2.5)
    specs["pre_bp"] = ((4.0, 40.0), -1.5, -0.5)      # pre-cue control windows (1 s, BEFORE the cue is shown)
    specs["pre_csp"] = ((8.0, 30.0), -1.5, -0.5)
    specs[PRIMARY] = (tuple(cfg.preprocessing["bandpass_hz"]), cfg.preprocessing["epoch_tmin_s"], cfg.preprocessing["epoch_tmax_s"])
    f = CACHE / "audit_bank.npz"
    from src.evaluation.epoch_bank import build_epoch_bank as build
    from src.signal.epoching import TrialSet

    log("building primary trial set + epoch bank")
    bank, bp = build(cfg, specs, log=lambda *_: None)
    if (bank[PRIMARY].shape != bp.X.shape) or not np.allclose(bank[PRIMARY], bp.X, rtol=1e-4, atol=1e-3):
        raise RuntimeError("epoch bank does not reproduce the primary epochs")
    log(f"bank ok: {len(bp.X)} trials; consistency with primary epochs verified")
    return bank, bp


def save_df(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    log("saved", name)


def stage_nested(bank, bp):
    rows, picks = [], []
    for p in PROTOCOLS:
        r, k = cs.nested_selection(cfg, bank, bp.meta, bp.y, p, log=log)
        rows.append(r)
        picks.append(k)
    save_df(pd.concat(rows), "csp_nested_all_configs.csv")
    save_df(pd.concat(picks), "csp_nested_selection.csv")


def stage_ablation(bank, bp):
    frames = [cs.ablation(cfg, bank["b8-30_w0.5-2.5"], bp.meta, bp.y, p, log=log) for p in PROTOCOLS]
    save_df(pd.concat(frames), "csp_ablation.csv")


def _save_run(name, res, splits, preds, bp, status, notes):
    return save_experiment(cfg, res, splits, preds, bp, name, status, notes, out_root=OUT, overwrite=True)


def stage_ea(bank, bp):
    X = bank["b8-30_w0.5-2.5"].astype(np.float64)
    groups = (bp.meta["subject"].astype(str) + "_" + bp.meta["session"].astype(str)).to_numpy()
    aligned = with_epochs(bp, align_by_group(X, groups))
    for p in PROTOCOLS:
        res, spl, pred = run_protocol(cfg, bp, aligned, p, models=("csp_lda",), log=log)
        _save_run(f"ea_csp_lda_{p}", res, spl, pred, bp, "exploratory",
                  "Euclidean alignment per participant-session using UNLABELED trials of the target participant/session (not zero-shot).")


def stage_erd(bank, bp):
    erd = with_epochs(bp, bank["erd"])
    for p in PROTOCOLS:
        res, spl, pred = run_protocol(cfg, erd, bp, p, models=("logreg", "cnn"), log=log, features="erd")
        _save_run(f"erd_{p}", res, spl, pred, bp, "exploratory",
                  "Band power relative to the pre-cue baseline (cue-1.5..-0.5 s) of the same trial; single pre-specified configuration.")


def stage_perm(bank, bp):
    csp = with_epochs(bp, bank["b8-30_w0.5-2.5"])
    for p in PROTOCOLS:
        res, spl, pred = run_protocol(cfg, bp, csp, p, log=log, permute_training_labels=True)
        _save_run(f"permuted_labels_{p}", res, spl, pred, bp, "exploratory",
                  "NEGATIVE CONTROL: labels of all non-test trials shuffled; test labels intact. Expected balanced accuracy ~0.5.")


def stage_cnnvar(bank, bp):
    csp = with_epochs(bp, bank["b8-30_w0.5-2.5"])
    variants = {"cnn_small": {"channels": [8, 16, 32]}, "cnn_wd1e-2": {"weight_decay": 1e-2}}
    for name, changes in variants.items():
        c = copy.deepcopy(cfg)
        c["models"]["cnn"].update(changes)
        for p in PROTOCOLS:
            res, spl, pred = run_protocol(c, bp, csp, p, models=("cnn",), log=log)
            _save_run(f"{name}_{p}", res, spl, pred, bp, "exploratory", f"Pre-specified CNN variant {changes}; reported whatever the outcome.")


def stage_layout(bank, bp):
    csp = with_epochs(bp, bank["b8-30_w0.5-2.5"])
    builder = LayoutGridBuilder()
    for p in PROTOCOLS:
        res, spl, pred = run_protocol(cfg, bp, csp, p, models=("cnn",), log=log, builder=builder)
        _save_run(f"cnn_layoutgrid_{p}", res, spl, pred, bp, "exploratory",
                  "CNN on the measured-only 8x8 electrode-layout grid (official figure topology, no interpolation); same hyper-parameters.")


def stage_pre(bank, bp):
    """Temporal-leakage control: decode the class from the 1 s BEFORE the cue (nothing about the class is known yet)."""
    pre_bp, pre_csp = with_epochs(bp, bank["pre_bp"]), with_epochs(bp, bank["pre_csp"])
    for p in PROTOCOLS:
        res, spl, pred = run_protocol(cfg, pre_bp, pre_csp, p, models=("logreg", "csp_lda"), log=log)
        _save_run(f"precue_control_{p}", res, spl, pred, bp, "exploratory",
                  "NEGATIVE CONTROL: epochs from cue-1.5..cue-0.5 s (before the cue is displayed); expected balanced accuracy ~0.5.")


def stage_cnncurves(bank, bp):
    csp = with_epochs(bp, bank["b8-30_w0.5-2.5"])
    hist = []
    for p in PROTOCOLS:
        res, spl, pred = run_protocol(cfg, bp, csp, p, models=("cnn",), log=log, collect_histories=hist)
        res.to_csv(OUT / f"cnn_diagnostics_{p}.csv", index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cnn_learning_curves.json").write_text(json.dumps(hist), encoding="utf-8")
    log("saved cnn_learning_curves.json")


def stage_pixel(bank, bp):
    builder = builder_from_config(bp.ch_names, cfg)
    rows = []
    from src.evaluation.splits import cross_session_split, loso_splits

    for p in PROTOCOLS:
        splits = ([cross_session_split(bp.meta, bp.y, int(s)) for s in sorted(bp.meta["subject"].unique())] if p == "cross_session"
                  else loso_splits(bp.meta, bp.y, cfg.models["cnn"]["n_val_subjects"]))
        for sp in splits:
            tr, te = sp.train_idx, sp.test_idx
            imgs_tr = images_from_epochs(bp.X[tr], bp.sfreq, cfg, builder)
            scaler = ImageScaler().fit(imgs_tr, builder.mask)                           # training only
            f = lambda im: scaler.transform(im)[:, :, builder.mask].reshape(len(im), -1)
            Xtr, Xte = f(imgs_tr), f(images_from_epochs(bp.X[te], bp.sfreq, cfg, builder))
            groups = bp.meta.iloc[tr]["run" if p == "cross_session" else "subject"].to_numpy()
            gs = GridSearchCV(LogisticRegression(max_iter=3000), {"C": list(cfg.models["logreg"]["c_grid"])}, scoring="balanced_accuracy",
                              cv=GroupKFold(n_splits=min(5, len(np.unique(groups)))))
            gs.fit(Xtr, bp.y[tr], groups=groups)
            s = gs.decision_function(Xte)
            m = compute_metrics(bp.y[te], (s > 0).astype(int), s)
            rows.append({"protocol": p, "test_subject": int(bp.meta.iloc[te[0]]["subject"]), "model": "pixel_logreg",
                         "balanced_accuracy": m["balanced_accuracy"], "accuracy": m["accuracy"], "best_C": gs.best_params_["C"], "n_test": m["n"]})
            log(p, rows[-1]["test_subject"], round(m["balanced_accuracy"], 3))
    save_df(pd.DataFrame(rows), "pixel_logreg.csv")


STAGES = {"nested": stage_nested, "ablation": stage_ablation, "ea": stage_ea, "erd": stage_erd, "perm": stage_perm,
          "cnnvar": stage_cnnvar, "pre": stage_pre, "layout": stage_layout, "cnncurves": stage_cnncurves, "pixel": stage_pixel}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    OUT.mkdir(parents=True, exist_ok=True)
    bank, bp = get_bank()
    for w in wanted:
        log("=== stage", w)
        STAGES[w](bank, bp)
    log("done")
