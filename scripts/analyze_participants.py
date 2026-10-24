"""Per-participant diagnostics and the S08 investigation (exploratory; n = 9 participants).

Usage: python scripts/analyze_participants.py [run_name] [audit_dir_name]
Reads outputs/<run_name>/ (default final_v1), recomputes descriptive statistics from the real data and writes
outputs/<audit_dir_name>/participants.csv, s08_investigation.json and influence_matrix.csv (default audit_v1).

Nothing here changes any model or any reported result. All correlations are Spearman on 9 points and are descriptive.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import load_config
from src.data.dataset import load_recording
from src.evaluation.experiments import load_dataset
from src.evaluation.metrics import compute_metrics
from src.features.spectral import log_band_power
from src.models.baseline import build_csp_lda_pipeline, decision_scores
from src.signal.quality import epoch_peak_to_peak, recording_quality


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


def pattern_similarity(p_own, p_pool):
    """Mean over own extreme CSP patterns (first and last) of the best |cosine| with the pooled extreme patterns."""
    own = np.stack([p_own[0], p_own[-1]])
    pool = np.stack([p_pool[0], p_pool[-1]])
    own = own / np.linalg.norm(own, axis=1, keepdims=True)
    pool = pool / np.linalg.norm(pool, axis=1, keepdims=True)
    return float(np.abs(own @ pool.T).max(axis=1).mean())


def fit_csp(cfg, X, y):
    return build_csp_lda_pipeline(cfg).fit(X, y)


def main(run: str = "final_v1", audit: str = "audit_v1") -> None:
    cfg = load_config()
    root = Path(cfg.resolve_path(cfg.evaluation["outputs_dir"]))
    res = pd.read_csv(root / run / "results_per_split.csv")
    preds = pd.read_csv(root / run / "predictions.csv")
    rej = pd.read_csv(root / run / "rejected_trials.csv")
    out = root / audit
    out.mkdir(parents=True, exist_ok=True)
    bp, csp = load_dataset(cfg, log=lambda *_: None)
    meta, y = bp.meta, bp.y
    subj = meta["subject"].to_numpy()
    ch = bp.ch_names
    c3, c4 = ch.index("C3"), ch.index("C4")

    # ---------------- participant table ----------------
    ptp = epoch_peak_to_peak(bp.X)
    lbp = log_band_power(bp.X, bp.sfreq, {"mu": cfg.features["bands_hz"]["mu"]}, 250, 0.5)[:, 0, :]
    li = lbp[:, c4] - lbp[:, c3]                      # lateralisation index: high when C3 (left) is desynchronised
    rows = []
    for s in sorted(np.unique(subj)):
        m = subj == s
        qs = []
        for sess in ("T", "E"):
            qs.append(recording_quality(load_recording(cfg, int(s), sess)))
        q = pd.concat(qs)
        r = {"subject": int(s), "n_retained": int(m.sum()), "n_left": int((y[m] == 0).sum()), "n_right": int((y[m] == 1).sum()),
             "n_rejected": int((rej["subject"] == s).sum())}
        r["rejection_rate"] = r["n_rejected"] / (r["n_rejected"] + r["n_retained"])
        r["median_epoch_ptp_uV"] = float(np.median(ptp[m]))
        r["median_channel_rms_uV_raw"] = float(q["rms_uV"].median())
        r["flat_channels"] = int(q["flat"].sum())
        r["mu_lateralisation_d"] = cohens_d(li[m & (y == 1)], li[m & (y == 0)])      # right vs left
        r["mu_lateralisation_d_T"] = cohens_d(li[m & (y == 1) & (meta["session"] == "T").to_numpy()], li[m & (y == 0) & (meta["session"] == "T").to_numpy()])
        r["mu_lateralisation_d_E"] = cohens_d(li[m & (y == 1) & (meta["session"] == "E").to_numpy()], li[m & (y == 0) & (meta["session"] == "E").to_numpy()])
        for protocol, tag in (("cross_session", "A"), ("loso", "B")):
            for model in ("csp_lda", "logreg", "cnn"):
                v = res[(res["protocol"] == protocol) & (res["test_subject"] == s) & (res["model"] == model)]["balanced_accuracy"]
                r[f"{tag}_{model}"] = float(v.iloc[0])
        rows.append(r)
    tab = pd.DataFrame(rows)

    # ---------------- influence of each training participant on each held-out participant (CSP+LDA, LOSO) -------------
    subjects = sorted(np.unique(subj))
    infl = pd.DataFrame(index=subjects, columns=["all"] + subjects, dtype=float)
    patt_sim = {}
    for s in subjects:
        te = subj == s
        others = [o for o in subjects if o != s]
        for drop in [None] + others:
            tr = np.isin(subj, [o for o in others if o != drop])
            model = fit_csp(cfg, csp.X[tr], y[tr])
            pred = (decision_scores(model, csp.X[te]) > 0).astype(int)
            infl.loc[s, "all" if drop is None else drop] = compute_metrics(y[te], pred)["balanced_accuracy"]
            if drop is None:
                own = fit_csp(cfg, csp.X[te], y[te])
                patt_sim[s] = pattern_similarity(own.named_steps["csp"].patterns_, model.named_steps["csp"].patterns_)
    infl.to_csv(out / "influence_matrix.csv")
    tab["loso_csp_pattern_similarity_to_pooled"] = tab["subject"].map(patt_sim)
    tab["loso_csp_min_after_dropping_one_training_subject"] = [float(infl.loc[s, [c for c in subjects if c != s]].min()) for s in tab["subject"]]
    tab["loso_csp_max_after_dropping_one_training_subject"] = [float(infl.loc[s, [c for c in subjects if c != s]].max()) for s in tab["subject"]]
    tab.to_csv(out / "participants.csv", index=False)

    # ---------------- S08 ----------------
    s08 = 8
    g = preds[(preds["protocol"] == "loso") & (preds["model"] == "csp_lda") & (preds["test_subject"] == s08)]
    info = {"confusion_matrix_loso_csp_lda": compute_metrics(g["y_true"], g["y_pred"], g["score"])["confusion_matrix"],
            "per_session_balanced_accuracy_loso_csp_lda": {sess: compute_metrics(gg["y_true"], gg["y_pred"])["balanced_accuracy"]
                                                          for sess, gg in g.groupby("session")}}
    rng = np.random.default_rng(0)
    boots = [compute_metrics(g["y_true"].to_numpy()[i], g["y_pred"].to_numpy()[i])["balanced_accuracy"]
             for i in (rng.integers(0, len(g), len(g)) for _ in range(2000))]
    info["bootstrap_over_trials_95ci_balanced_accuracy_(trials_treated_as_independent)"] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
    info["jackknife_over_training_participants_balanced_accuracy"] = {str(c): float(infl.loc[s08, c]) for c in infl.columns}
    # train on T sessions only, test on E only (a different but legitimate split)
    te = (subj == s08) & (meta["session"] == "E").to_numpy()
    tr = (subj != s08) & (meta["session"] == "T").to_numpy()
    model = fit_csp(cfg, csp.X[tr], y[tr])
    info["train_other_T_only__test_S08_E_only"] = compute_metrics(y[te], (decision_scores(model, csp.X[te]) > 0).astype(int))["balanced_accuracy"]
    info["within_participant_protocol_A_csp_lda"] = float(tab.loc[tab.subject == s08, "A_csp_lda"].iloc[0])
    info["spearman_across_participants"] = {}
    for col in ("rejection_rate", "mu_lateralisation_d", "median_channel_rms_uV_raw", "median_epoch_ptp_uV", "n_retained",
                "loso_csp_pattern_similarity_to_pooled", "A_csp_lda"):
        rho, p = spearmanr(tab[col], tab["B_csp_lda"])
        info["spearman_across_participants"][f"{col}_vs_B_csp_lda"] = {"rho": float(rho), "p_uncorrected": float(p), "n": 9}
        rho, p = spearmanr(tab[col], tab["A_csp_lda"])
        info["spearman_across_participants"][f"{col}_vs_A_csp_lda"] = {"rho": float(rho), "p_uncorrected": float(p), "n": 9}
    (out / "s08_investigation.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(tab.round(3).T)
    print(json.dumps({k: v for k, v in info.items() if k != "jackknife_over_training_participants_balanced_accuracy"}, indent=1))
    print("S08 jackknife:", {k: round(v, 3) for k, v in info["jackknife_over_training_participants_balanced_accuracy"].items()})
    print("influence matrix (rows held-out; cols = dropped training participant):")
    print(infl.round(2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
