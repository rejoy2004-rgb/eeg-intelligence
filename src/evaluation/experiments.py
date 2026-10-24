"""Running the evaluation protocols and saving auditable results.

Per outer split the order of operations is always:
    1. choose the training / validation / test trial indices (src.evaluation.splits)
    2. fit EVERYTHING that learns from data on training trials only
         - scalers, CSP filters, classifiers (inside sklearn pipelines)
         - ImageScaler statistics for the CNN (on the CNN-fit partition only)
         - hyper-parameter C of the logistic regression (inner grouped CV inside the training set)
         - CNN epoch selection (validation partition, which is never the test partition)
    3. predict the test partition once, compute metrics

Models compared (all on exactly the same trials):
    majority   : always predicts the most frequent training class (floor)
    logreg     : log band power -> logistic regression
    csp_lda    : CSP -> LDA
    cnn        : log band power -> spatial images -> small CNN
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier

from src.config import Config, software_versions
from src.data.dataset import load_external_labels, load_recording
from src.evaluation.metrics import balanced_accuracy_simple, chance_threshold, compute_metrics, summarize, wilson_interval
from src.evaluation.splits import Split, assert_no_overlap, cross_session_split, loso_splits, split_to_json
from src.features.spatial import ImageScaler, builder_from_config, images_from_epochs
from src.models.baseline import decision_scores, erd_band_power, fit_csp_lda, fit_logreg
from src.signal.epoching import TrialSet, build_dual_band_trials

MODELS = ("majority", "logreg", "csp_lda", "cnn")
# Runs that define the established baseline. They can never be written to again (even with overwrite=True).
PROTECTED_RUNS = frozenset({"final_v1", "repro_v1", "repro_v2"})


def load_dataset(cfg: Config, subjects=None, sessions=None, log=print) -> tuple[TrialSet, TrialSet]:
    """Load, preprocess and epoch all requested recordings. Returns (bp_trials, csp_trials).

    Both sets contain the same trials in the same order. One recording is in memory at a time.
    """
    subjects = list(subjects) if subjects is not None else list(cfg.dataset["subjects"])
    sessions = list(sessions) if sessions is not None else list(cfg.dataset["sessions"])
    bp_sets, csp_sets = [], []
    for s in subjects:
        for sess in sessions:
            rec = load_recording(cfg, s, sess)
            labels = load_external_labels(cfg, s, sess)
            bp, csp = build_dual_band_trials(rec, cfg, labels)
            log(f"  S{s:02d}{sess}: {bp.summary['n_kept']}/{bp.summary['n_task_trials_before_rejection']} "
                f"left/right trials kept {bp.summary['rejection_counts']}")
            bp_sets.append(bp)
            csp_sets.append(csp)
            del rec
    return TrialSet.concat(bp_sets), TrialSet.concat(csp_sets)


def _fit_predict_split(cfg: Config, split: Split, bp: TrialSet, csp: TrialSet, models, builder, protocol_seed: int,
                       y_fit=None, features: str = "power"):
    """Fit all requested models on one outer split. Returns {model: (y_pred, scores, extra_info)}."""
    from src.models.eeg_cnn import predict_scores, train_cnn

    # y is used ONLY for fitting / model selection. ``y_fit`` lets the permutation control substitute shuffled
    # training labels; test labels are always taken from bp.y by the caller.
    y = bp.y if y_fit is None else y_fit
    tr, te = split.train_idx, split.test_idx
    out = {}
    group_col = "run" if split.protocol == "cross_session" else "subject"
    groups = bp.meta.iloc[tr][group_col].to_numpy()

    if "majority" in models:
        m = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(tr), 1)), y[tr])
        pred = m.predict(np.zeros((len(te), 1)))
        out["majority"] = (pred, np.zeros(len(te)), {})
    if "logreg" in models:
        m, info = fit_logreg(cfg, bp.sfreq, bp.X[tr], y[tr], groups, features=features)
        s = decision_scores(m, bp.X[te])
        out["logreg"] = ((s > 0).astype(int), s, info)
    if "csp_lda" in models:
        m = fit_csp_lda(cfg, csp.X[tr], y[tr])
        s = decision_scores(m, csp.X[te])
        out["csp_lda"] = ((s > 0).astype(int), s, {})
    if "cnn" in models:
        fit_idx = np.setdiff1d(tr, split.val_idx)
        to_images = (lambda X: images_from_epochs(X, bp.sfreq, cfg, builder)) if features == "power" else \
            (lambda X: builder.transform(erd_band_power(X, bp.sfreq, dict(cfg.features["bands_hz"]),
                                                        cfg.features["welch_nperseg"], cfg.features["welch_overlap"])))
        img_fit = to_images(bp.X[fit_idx])
        scaler = ImageScaler().fit(img_fit, builder.mask)                      # training statistics only
        x_fit = scaler.transform(img_fit)
        x_val = scaler.transform(to_images(bp.X[split.val_idx]))
        x_te = scaler.transform(to_images(bp.X[te]))
        net, hist = train_cnn(x_fit, y[fit_idx], x_val, y[split.val_idx], cfg, seed=protocol_seed)
        s = predict_scores(net, x_te)
        train_scores = predict_scores(net, x_fit)       # diagnostic only (overfitting evidence); not used for selection
        out["cnn"] = ((s > 0).astype(int), s, {"best_epoch": hist["best_epoch"],
                                              "best_val_balanced_accuracy": hist["best_val_balanced_accuracy"],
                                              "epochs_run": len(hist["val_loss"]),
                                              "train_balanced_accuracy": float(balanced_accuracy_simple(y[fit_idx], (train_scores > 0).astype(int))),
                                              "history": hist})
    return out


def run_protocol(cfg: Config, bp: TrialSet, csp: TrialSet, protocol: str, models=MODELS, log=print,
                 permute_training_labels: bool = False, features: str = "power", collect_histories: list | None = None, builder=None,
                 per_split_data=None):
    """Run one evaluation protocol ('cross_session' or 'loso'). Returns (results_df, splits_json, predictions_df)."""
    if protocol not in ("cross_session", "loso"):
        raise ValueError("protocol must be 'cross_session' or 'loso'")
    if not (bp.meta[["subject", "session", "trial"]].equals(csp.meta[["subject", "session", "trial"]])):
        raise ValueError("The two epoch sets are not aligned")
    builder = builder if builder is not None else builder_from_config(bp.ch_names, cfg)
    subjects = sorted(int(s) for s in bp.meta["subject"].unique())
    if protocol == "cross_session":
        splits = [cross_session_split(bp.meta, bp.y, s) for s in subjects]
    else:
        splits = loso_splits(bp.meta, bp.y, cfg.models["cnn"]["n_val_subjects"])

    rows, preds = [], []
    for k, split in enumerate(splits):
        t0 = time.time()
        assert_no_overlap(bp.meta, split.train_idx, split.test_idx, by="trial_id")
        if protocol == "loso":
            assert_no_overlap(bp.meta, split.train_idx, split.test_idx, by="subject")
        test_subject = int(bp.meta.iloc[split.test_idx[0]]["subject"])
        y_fit = None
        if permute_training_labels:      # negative control: destroy the label signal in training/validation only
            y_fit = bp.y.copy()
            non_test = np.setdiff1d(np.arange(len(y_fit)), split.test_idx)
            y_fit[non_test] = np.random.default_rng(cfg.evaluation["seed"] + test_subject).permutation(y_fit[non_test])
        bp_k, csp_k = (bp, csp) if per_split_data is None else per_split_data(test_subject)   # e.g. fold-specific EOG correction
        if per_split_data is not None and not (np.array_equal(bp_k.y, bp.y) and bp_k.meta.equals(bp.meta)):
            raise ValueError("per_split_data must return trial sets with identical labels and metadata")
        res = _fit_predict_split(cfg, split, bp_k, csp_k, models, builder, cfg.evaluation["seed"] + test_subject,
                                 y_fit=y_fit, features=features)
        for name, (pred, scores, info) in res.items():
            info = dict(info)
            history = info.pop("history", None)          # CNN learning curves are kept out of the results table
            if collect_histories is not None and history is not None:
                collect_histories.append({"protocol": protocol, "test_subject": test_subject, "history": history})
            m = compute_metrics(bp.y[split.test_idx], pred, scores if name != "majority" else None)
            lo, hi = wilson_interval(int(round(m["accuracy"] * m["n"])), m["n"])
            rows.append({"protocol": protocol, "split": split.name, "test_subject": test_subject, "model": name,
                         "n_train": len(split.train_idx), "n_test": m["n"],
                         **{k2: v for k2, v in m.items() if k2 not in ("confusion_matrix", "notes")},
                         "accuracy_ci_low": lo, "accuracy_ci_high": hi,
                         "chance_threshold_p05": chance_threshold(m["n"]),
                         "notes": "; ".join(m["notes"]), **{f"info_{a}": b for a, b in info.items()}})
            preds.append(pd.DataFrame({"protocol": protocol, "model": name, "test_subject": test_subject,
                                       **bp.meta.iloc[split.test_idx][["session", "trial"]].reset_index(drop=True),
                                       "y_true": bp.y[split.test_idx], "y_pred": pred, "score": scores}))
        accs = {n: round(r[0].__eq__(bp.y[split.test_idx]).mean(), 3) for n, r in res.items()}
        log(f"  [{protocol}] {split.name}: accuracy {accs} ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows), [split_to_json(s) for s in splits], pd.concat(preds, ignore_index=True)


def aggregate(results: pd.DataFrame) -> dict:
    """Mean / std across held-out participants per protocol and model, plus paired comparisons."""
    out: dict = {}
    for protocol, g in results.groupby("protocol"):
        out[protocol] = {}
        for model, gm in g.groupby("model"):
            out[protocol][model] = {
                "n_splits": int(len(gm)),
                **{metric: summarize(gm[metric].to_numpy()) for metric in
                   ("accuracy", "balanced_accuracy", "f1", "roc_auc", "precision", "recall", "specificity")},
                "pooled_confusion_matrix": [[int(gm["tn"].sum()), int(gm["fp"].sum())],
                                            [int(gm["fn"].sum()), int(gm["tp"].sum())]],
            }
        pivot = g.pivot(index="test_subject", columns="model", values="balanced_accuracy")
        comps = {}
        models = [m for m in pivot.columns if m != "majority"]
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                d = (pivot[a] - pivot[b]).dropna()
                if len(d) >= 6 and np.any(d != 0):
                    p = float(stats.wilcoxon(d).pvalue)
                else:
                    p = float("nan")
                comps[f"{a}_vs_{b}"] = {"mean_difference_balanced_accuracy": float(d.mean()) if len(d) else float("nan"),
                                        "wilcoxon_p_two_sided": p, "n_pairs": int(len(d))}
        out[protocol]["paired_comparisons"] = comps
    return out


def save_experiment(cfg: Config, results: pd.DataFrame, splits_json: dict, preds: pd.DataFrame, bp: TrialSet,
                    run_name: str, status: str, notes: str = "", out_root: Path | None = None, overwrite: bool = False,
                    extra: dict | None = None) -> Path:
    """Write results + everything needed to audit and reproduce them into outputs/<run_name>/."""
    root = out_root or cfg.resolve_path(cfg.evaluation["outputs_dir"])
    out = root / run_name
    if run_name in PROTECTED_RUNS:
        raise FileExistsError(f"'{run_name}' is a protected baseline run and cannot be written to")
    if (out / "summary.json").exists() and not overwrite:
        raise FileExistsError(f"{out} already holds a finished experiment; choose a new run name (or pass overwrite=True)")
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "results_per_split.csv", index=False)
    preds.to_csv(out / "predictions.csv", index=False)
    bp.rejections.to_csv(out / "rejected_trials.csv", index=False)
    (out / "splits.json").write_text(json.dumps(splits_json, indent=2), encoding="utf-8")
    summary = {
        "run_name": run_name,
        "status": status,        # "final" or "exploratory"
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
        "aggregate": aggregate(results),
        "n_trials_total": bp.n_trials,
        "class_counts": bp.class_counts(),
        "label_mapping": {"0": "left hand", "1": "right hand"},
        "config": cfg.to_dict(),
        "software": software_versions(),
        **(extra or {}),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    return out
