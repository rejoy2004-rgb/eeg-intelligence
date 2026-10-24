"""Diagnostic: could eye movements / ocular artifacts carry the left-vs-right label?  (S08 investigation)

The cue is a left/right ARROW shown for 1.25 s at t=2..3.25 s, so gaze shifts or blinks time-locked to the cue could
leak the class into frontal EEG channels. The dataset provides 3 EOG channels precisely so that this can be checked;
the providers say they "must not be used for classification". This script therefore uses the EOG ONLY as a
diagnostic - no model of the project ever receives it.

For each participant (primary trials, window cue+0.5..cue+2.5 s, baseline cue-0.5..cue s):
  * class effect size (Cohen's d, right vs left) of the baseline-corrected horizontal EOG (EOG-left minus EOG-right)
  * EOG-only decoding: standardised logistic regression on [mean shift, log variance] of the 3 EOG channels
      - protocol-A-like (train T -> test E, same participant)
      - cross-participant (train on the other 8, test the held-out one)
  * CSP + LDA restricted to channels that are NOT frontal (drops Fz, FC3, FC1, FCz, FC2, FC4) and the largest-|weight|
    channels of the first CSP pattern per participant
Usage: python scripts/investigate_ocular_confound.py [audit_dir_name]  -> outputs/<dir>/ocular_confound.csv + json
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.data.dataset import recording_path
from src.evaluation.experiments import load_dataset
from src.evaluation.metrics import compute_metrics
from src.models.baseline import build_csp_lda_pipeline, decision_scores

mne.set_log_level("ERROR")


def d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


def main(name: str = "audit_v1") -> None:
    cfg = load_config()
    pre = cfg.preprocessing
    out = Path(cfg.resolve_path(cfg.evaluation["outputs_dir"])) / name
    out.mkdir(parents=True, exist_ok=True)
    bp, csp = load_dataset(cfg, log=lambda *_: None)
    meta, y, sf = bp.meta, bp.y, bp.sfreq
    a, b = int(round(pre["epoch_tmin_s"] * sf)), int(round(pre["epoch_tmax_s"] * sf))
    nb = int(round(0.5 * sf))
    feats = np.zeros((len(meta), 7))        # shift(3), logvar(3), horizontal shift
    for (s, sess), g in meta.groupby(["subject", "session"]):
        raw = mne.io.read_raw_gdf(str(recording_path(cfg, int(s), sess)), preload=True, verbose="ERROR")
        eog = raw.get_data(picks=[n for n in raw.ch_names if n.upper().startswith("EOG")]) * 1e6   # (3, N): left, central, right
        for i, cue in zip(g.index, g["cue_sample"]):
            w = eog[:, cue + a: cue + b]
            base = eog[:, cue - nb: cue].mean(axis=1, keepdims=True)
            feats[i, :3] = (w - base).mean(axis=1)
            feats[i, 3:6] = np.log(w.var(axis=1) + 1e-9)
            feats[i, 6] = feats[i, 0] - feats[i, 2]
    subj = meta["subject"].to_numpy()
    sess = meta["session"].to_numpy()
    frontal = [bp.ch_names.index(c) for c in ("Fz", "FC3", "FC1", "FCz", "FC2", "FC4")]
    keep = [i for i in range(22) if i not in frontal]
    rows = []
    for s in sorted(np.unique(subj)):
        m = subj == s
        r = {"subject": int(s), "horizontal_eog_shift_d_right_vs_left": d(feats[m & (y == 1), 6], feats[m & (y == 0), 6])}
        tr, te = m & (sess == "T"), m & (sess == "E")
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(feats[tr][:, :6], y[tr])
        r["eog_only_A_like_balanced_accuracy"] = compute_metrics(y[te], clf.predict(feats[te][:, :6]))["balanced_accuracy"]
        clf.fit(feats[~m][:, :6], y[~m])
        r["eog_only_LOSO_balanced_accuracy"] = compute_metrics(y[m], clf.predict(feats[m][:, :6]))["balanced_accuracy"]
        # CSP + LDA without the frontal channels (where ocular activity is largest), same splits as the primary analysis
        mdl = build_csp_lda_pipeline(cfg).fit(csp.X[tr][:, keep], y[tr])
        r["csp_no_frontal_A_balanced_accuracy"] = compute_metrics(y[te], (decision_scores(mdl, csp.X[te][:, keep]) > 0).astype(int))["balanced_accuracy"]
        mdl = build_csp_lda_pipeline(cfg).fit(csp.X[~m][:, keep], y[~m])
        r["csp_no_frontal_LOSO_balanced_accuracy"] = compute_metrics(y[m], (decision_scores(mdl, csp.X[m][:, keep]) > 0).astype(int))["balanced_accuracy"]
        own = build_csp_lda_pipeline(cfg).fit(csp.X[m], y[m])
        pat = own.named_steps["csp"].patterns_[0]
        top = np.argsort(-np.abs(pat))[:4]
        r["own_csp_pattern0_top_channels"] = ",".join(bp.ch_names[i] for i in top)
        pat2 = own.named_steps["csp"].patterns_[-1]
        r["own_csp_pattern_last_top_channels"] = ",".join(bp.ch_names[i] for i in np.argsort(-np.abs(pat2))[:4])
        rows.append(r)
    tab = pd.DataFrame(rows)
    prim = pd.read_csv(Path(cfg.resolve_path(cfg.evaluation["outputs_dir"])) / "final_v1" / "results_per_split.csv")
    for proto, tag in (("cross_session", "A"), ("loso", "B")):
        tab[f"primary_csp_{tag}"] = [float(prim[(prim.protocol == proto) & (prim.model == "csp_lda") & (prim.test_subject == s)]["balanced_accuracy"].iloc[0]) for s in tab["subject"]]
    tab.to_csv(out / "ocular_confound.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:2])
