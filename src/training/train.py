"""Command-line entry point: run evaluations and export a model artifact.

Examples
--------
    python -m src.training.train evaluate --protocol cross_session loso --name final_v1 --status final
    python -m src.training.train evaluate --subjects 1 2 3 --models logreg csp_lda --name smoke --status exploratory
    python -m src.training.train export-demo --heldout 1

"exploratory" results are for debugging / development; only a run labelled "final" (a single
pre-planned execution of the full protocol) should be quoted as a result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from src.config import Config, load_config, software_versions
from src.evaluation.experiments import MODELS, load_dataset, run_protocol, save_experiment
from src.features.spatial import ImageScaler, builder_from_config, images_from_epochs
from src.models.baseline import fit_csp_lda, fit_logreg
from src.models.eeg_cnn import save_cnn, train_cnn
from src.signal.epoching import TrialSet


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def export_artifact(cfg: Config, bp: TrialSet, csp: TrialSet, out_dir: Path, heldout_subject: int | None = None) -> Path:
    """Train all three models on every participant except ``heldout_subject`` and save them.

    The artifact folder holds everything needed to reproduce the prediction preprocessing: the full
    config, channel order, band definitions, label mapping, CNN input scaler statistics and
    SHA-256 hashes of the model files.
    """
    subj = bp.meta["subject"].to_numpy()
    all_subjects = sorted(int(s) for s in np.unique(subj))
    train_subjects = [s for s in all_subjects if s != heldout_subject]
    n_val = cfg.models["cnn"]["n_val_subjects"]
    val_subjects = train_subjects[-n_val:]                 # deterministic rule, no data inspected
    fit_subjects = [s for s in train_subjects if s not in val_subjects]
    tr = np.flatnonzero(np.isin(subj, train_subjects))
    fit = np.flatnonzero(np.isin(subj, fit_subjects))
    val = np.flatnonzero(np.isin(subj, val_subjects))
    out_dir.mkdir(parents=True, exist_ok=True)

    lr, lr_info = fit_logreg(cfg, bp.sfreq, bp.X[tr], bp.y[tr], subj[tr])
    joblib.dump(lr, out_dir / "logreg.joblib")
    cs = fit_csp_lda(cfg, csp.X[tr], bp.y[tr])
    joblib.dump(cs, out_dir / "csp_lda.joblib")

    builder = builder_from_config(bp.ch_names, cfg)
    img_fit = images_from_epochs(bp.X[fit], bp.sfreq, cfg, builder)
    scaler = ImageScaler().fit(img_fit, builder.mask)
    x_val = scaler.transform(images_from_epochs(bp.X[val], bp.sfreq, cfg, builder))
    net, hist = train_cnn(scaler.transform(img_fit), bp.y[fit], x_val, bp.y[val], cfg, seed=cfg.evaluation["seed"])
    save_cnn(out_dir, net, {"image_scaler": scaler.state(), "best_epoch": hist["best_epoch"],
                            "best_val_balanced_accuracy": hist["best_val_balanced_accuracy"]})

    files = ["logreg.joblib", "csp_lda.joblib", "cnn.pt", "cnn_meta.json"]
    artifact = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Research/demo artifact. Not a diagnostic tool. Scores are NOT calibrated probabilities.",
        "heldout_subject": heldout_subject,
        "train_subjects": train_subjects, "cnn_validation_subjects": val_subjects,
        "n_train_trials": int(len(tr)), "class_counts_train": {int(k): int(v) for k, v in zip(*np.unique(bp.y[tr], return_counts=True))},
        "label_mapping": {"0": "left hand", "1": "right hand"},
        "channels": bp.ch_names, "sfreq": bp.sfreq,
        "logreg_info": lr_info,
        "config": cfg.to_dict(), "software": software_versions(),
        "file_sha256": {f: _sha256(out_dir / f) for f in files},
        "security_note": "joblib files are pickles: only load artifacts you created yourself.",
    }
    (out_dir / "artifact.json").write_text(json.dumps(artifact, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    return out_dir


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="run evaluation protocol(s) and save results")
    ev.add_argument("--protocol", nargs="+", default=["cross_session", "loso"], choices=["cross_session", "loso"])
    ev.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ev.add_argument("--subjects", nargs="+", type=int, default=None)
    ev.add_argument("--name", required=True)
    ev.add_argument("--status", choices=["final", "exploratory", "reproduction"], default="exploratory")
    ev.add_argument("--notes", default="")
    ex = sub.add_parser("export-demo", help="train on all participants except one and save a model artifact")
    ex.add_argument("--heldout", type=int, default=1)
    ex.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.command == "evaluate":
        print("Loading and preprocessing recordings ...")
        bp, csp = load_dataset(cfg, args.subjects)
        import pandas as pd

        frames, splits, preds = [], [], []
        for protocol in args.protocol:
            r, s, p = run_protocol(cfg, bp, csp, protocol, tuple(args.models))
            frames.append(r)
            splits += s
            preds.append(p)
        out = save_experiment(cfg, pd.concat(frames, ignore_index=True), splits, pd.concat(preds, ignore_index=True),
                              bp, args.name, args.status, args.notes)
        print(f"Saved results to {out}")
    else:
        print("Loading and preprocessing recordings ...")
        bp, csp = load_dataset(cfg)
        out = Path(args.out) if args.out else cfg.resolve_path(cfg.evaluation["models_dir"]) / f"demo_heldout_S{args.heldout:02d}"
        export_artifact(cfg, bp, csp, out, args.heldout)
        print(f"Saved artifact to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
