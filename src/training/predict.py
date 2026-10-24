"""Prediction from a saved artifact, using exactly the training-time preprocessing.

Command line:
    python -m src.training.predict --artifact models/demo_heldout_S01 --file data/raw/bci_iv_2a/A01E.gdf \
        --labels data/raw/labels_2a/A01E.mat --model cnn

Important limits (also shown in the dashboard):
    * The classifier is binary (left vs right hand). A recording contains cues for 4 classes; without
      labels the model cannot know which trials are feet/tongue and will still output left/right.
      Pass the official label file (or use a T-file) so non-task trials are excluded.
    * Scores are decision values, NOT calibrated probabilities.
    * This is a research demo; it says nothing about a person's health or mental state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import Config
from src.data.dataset import load_external_labels, load_gdf_file
from src.data.validation import DataValidationError
from src.features.spatial import ImageScaler, builder_from_config, images_from_epochs
from src.models.baseline import decision_scores
from src.models.eeg_cnn import ModelInputError, load_cnn, predict_scores
from src.signal.epoching import TrialSet, build_dual_band_trials

import scipy.io as sio

CLASS_NAMES = {0: "left hand", 1: "right hand"}


@dataclass
class Artifact:
    path: Path
    meta: dict
    cfg: Config
    logreg: object
    csp_lda: object
    cnn: object
    scaler: ImageScaler
    builder: object


def load_artifact(directory: str | Path) -> Artifact:
    """Load a model folder written by ``export_artifact``; verifies file hashes first.

    Hash verification detects accidental corruption / tampering of files after export. It is not a
    defence against a malicious artifact author: joblib files are pickles, so only open artifacts
    you trust.
    """
    d = Path(directory)
    meta_path = d / "artifact.json"
    if not meta_path.is_file():
        raise DataValidationError(f"No artifact.json in {d}; run `python -m src.training.train export-demo` first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for fname, expected in meta["file_sha256"].items():
        fp = d / fname
        if not fp.is_file():
            raise DataValidationError(f"Artifact file missing: {fname}")
        if hashlib.sha256(fp.read_bytes()).hexdigest() != expected:
            raise DataValidationError(f"Hash mismatch for {fname}: file changed since export; refusing to load")
    cfg = Config(meta["config"])
    cnn, cnn_meta = load_cnn(d)
    return Artifact(
        path=d, meta=meta, cfg=cfg, logreg=joblib.load(d / "logreg.joblib"), csp_lda=joblib.load(d / "csp_lda.joblib"),
        cnn=cnn, scaler=ImageScaler.from_state(cnn_meta["image_scaler"]),
        builder=builder_from_config(meta["channels"], cfg),
    )


def predict_trials(art: Artifact, bp: TrialSet, csp: TrialSet, model: str = "cnn") -> pd.DataFrame:
    """Predict epochs that were produced by ``build_dual_band_trials`` with the artifact's config."""
    if model not in ("logreg", "csp_lda", "cnn"):
        raise ValueError("model must be 'logreg', 'csp_lda' or 'cnn'")
    if bp.n_trials == 0:
        raise DataValidationError("No usable trials to predict (all rejected or none present)")
    if list(bp.ch_names) != list(art.meta["channels"]) or bp.sfreq != art.meta["sfreq"]:
        raise DataValidationError("Recording channels / sampling rate do not match the model artifact")
    if model == "logreg":
        scores = decision_scores(art.logreg, bp.X)
    elif model == "csp_lda":
        scores = decision_scores(art.csp_lda, csp.X)
    else:
        images = art.scaler.transform(images_from_epochs(bp.X, bp.sfreq, art.cfg, art.builder))
        scores = predict_scores(art.cnn, images)
    pred = (scores > 0).astype(int)
    out = bp.meta[["subject", "session", "trial", "run"]].copy()
    out["model"] = model
    out["score_right_minus_left"] = scores
    out["predicted_class"] = [CLASS_NAMES[p] for p in pred]
    out["true_class"] = [CLASS_NAMES.get(int(t), "unknown") for t in bp.y]
    return out.reset_index(drop=True)


def predict_recording(art: Artifact, gdf_path: str | Path, labels_path: str | Path | None = None,
                      model: str = "cnn") -> tuple[pd.DataFrame, dict]:
    """Validate a GDF file, run the saved preprocessing, and predict every usable trial.

    Returns (predictions, info) where info reports how many trials were rejected and why.
    """
    rec = load_gdf_file(gdf_path, art.cfg)
    ext = None
    if labels_path is not None:
        lp = Path(labels_path)
        if not lp.is_file():
            raise DataValidationError(f"Label file not found: {lp}")
        mat = sio.loadmat(str(lp))
        if "classlabel" not in mat:
            raise DataValidationError("Label file has no 'classlabel' variable")
        ext = np.asarray(mat["classlabel"]).ravel().astype(int)
    bp, csp = build_dual_band_trials(rec, art.cfg, ext, require_labels=ext is not None or _has_labels(rec))
    df = predict_trials(art, bp, csp, model)
    info = {"summary": bp.summary, "warnings": rec.report.warnings,
            "labelled": bool((bp.y >= 0).any()), "rejections": bp.rejections.to_dict("records")}
    return df, info


def _has_labels(rec) -> bool:
    return bool(rec.events["code"].isin([769, 770, 771, 772]).any())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--model", choices=["logreg", "csp_lda", "cnn"], default="cnn")
    ap.add_argument("--out", default=None, help="optional CSV path for the predictions")
    args = ap.parse_args(argv)
    try:
        art = load_artifact(args.artifact)
        df, info = predict_recording(art, args.file, args.labels, args.model)
    except (DataValidationError, ModelInputError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(df.head(10).to_string(index=False))
    print(f"... {len(df)} trials predicted. Rejected: {info['summary']['rejection_counts']}")
    if info["labelled"]:
        known = df[df["true_class"] != "unknown"]
        print(f"Accuracy on labelled trials: {(known['predicted_class'] == known['true_class']).mean():.3f} (n={len(known)})")
    if args.out:
        df.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
