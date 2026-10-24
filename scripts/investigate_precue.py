"""Investigation of the pre-cue negative control (outputs/ocular_robust_v1/precue/).

The audit found CSP+LDA balanced accuracy 0.537 (A) / 0.521 (B) when decoding left vs right from the 1 s BEFORE the cue
(cue-1.5 ... cue-0.5 s). This script examines the candidate explanations, one by one, and reports what each does and does not explain:

 1. timing_audit            window position vs trial start, cue, previous trial, data gaps
 2. label_sequences         structure of the cue sequences (lag-1 dependence, identical sequences across sessions/participants)
 3. filter_leakage_profile  backward spread of the zero-phase filter + re-run of the control with a CAUSAL filter
 4. permutation test        mean balanced accuracy vs a null built by permuting labels WITHIN runs (participants as the unit)
 5. one-sample tests        participant-level balanced accuracy vs 0.5 with Holm correction
 6. previous_trial_control  can the class of the PREVIOUS trial be decoded from the pre-cue window? (carry-over hypothesis)
Nothing here changes any primary result.  Usage: python scripts/investigate_precue.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import load_external_labels, load_recording
from src.evaluation import precue as pc
from src.evaluation import stats as st
from src.evaluation.csp_selection import make_splits
from src.evaluation.epoch_bank import build_epoch_bank, with_epochs
from src.evaluation.experiments import run_protocol, save_experiment
from src.evaluation.metrics import compute_metrics
from src.models.baseline import build_csp_lda_pipeline, decision_scores
from src.signal.epoching import locate_trials

cfg = load_config()
ROOT_OUT = Path(cfg.resolve_path(cfg.evaluation["outputs_dir"]))
OUT = ROOT_OUT / "ocular_robust_v1"
PRE = OUT / "precue"
PROTOCOLS = ("cross_session", "loso")
MODELS = ("logreg", "csp_lda")
TMIN, TMAX = -1.5, -0.5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    PRE.mkdir(parents=True, exist_ok=True)
    rejected = pd.read_csv(ROOT_OUT / "final_v1" / "rejected_trials.csv")

    # ---------------------------------------------------------------- 1 + 2: timing and label sequences
    labels, run_of, timing = {}, {}, []
    for s in cfg.dataset["subjects"]:
        for sess in cfg.dataset["sessions"]:
            rec = load_recording(cfg, s, sess)
            lab = load_external_labels(cfg, s, sess)
            tr = locate_trials(rec, cfg, lab)
            labels[(s, sess)], run_of[(s, sess)] = lab, tr["run"].to_numpy()
            rej = set(rejected[(rejected.subject == s) & (rejected.session == sess)]["trial"])
            kept = [t for t, l in zip(tr["trial"], lab) if l in (1, 2) and t not in rej]
            timing.append({"recording": f"S{s:02d}{sess}", **pc.timing_audit(rec, tr, TMIN, TMAX, kept)})
            # inter-trial spacing for the report
            timing[-1]["trial_start_spacing_s[min,median,max]"] = [float(v) for v in np.percentile(np.diff(tr["trial_start"]) / rec.sfreq, [0, 50, 100])]
    agg = {k: [min(t[k][0] for t in timing), float(np.median([t[k][1] for t in timing])), max(t[k][2] for t in timing)]
           for k in ("window_start_after_trial_start_s[min,median,max]", "window_end_before_cue_s[min,median,max]",
                     "window_start_after_previous_imagery_end_s[min,median,max]")}
    agg["windows_overlapping_missing_gap_total"] = int(sum(t["windows_overlapping_missing_gap"] for t in timing))
    agg["windows_crossing_trial_start_or_cue_total"] = int(sum(t["windows_crossing_trial_start_or_cue"] for t in timing))
    agg["trial_start_spacing_s[min,median,max]"] = [min(t["trial_start_spacing_s[min,median,max]"][0] for t in timing),
                                                    float(np.median([t["trial_start_spacing_s[min,median,max]"][1] for t in timing])),
                                                    max(t["trial_start_spacing_s[min,median,max]"][2] for t in timing)]
    seq = pc.label_sequence_stats(labels)
    leak = {name: pc.impulse_leakage_profile(band, 250.0, cfg.preprocessing["filter_order"]) for name, band in
            (("4-40Hz", cfg.preprocessing["bandpass_hz"]), ("8-30Hz", cfg.preprocessing["csp_bandpass_hz"]))}
    (PRE / "timing_audit.json").write_text(json.dumps({"aggregate": agg, "per_recording": timing}, indent=1), encoding="utf-8")
    (PRE / "label_sequences.json").write_text(json.dumps(seq, indent=1), encoding="utf-8")
    (PRE / "filter_leakage_profile.json").write_text(json.dumps(leak, indent=1), encoding="utf-8")
    log("timing:", json.dumps(agg))
    log("sequences:", json.dumps({k: v for k, v in seq.items() if "counts" not in k}))
    log("filter leakage:", json.dumps(leak))

    # ---------------------------------------------------------------- 3: pre-cue control, zero-phase vs causal
    specs = {"zp_bp": ((4.0, 40.0), TMIN, TMAX), "zp_csp": ((8.0, 30.0), TMIN, TMAX),
             "c_bp": ((4.0, 40.0), TMIN, TMAX, {"causal": True}), "c_csp": ((8.0, 30.0), TMIN, TMAX, {"causal": True})}
    log("building pre-cue epoch bank (zero-phase and causal)")
    bank, bp = build_epoch_bank(cfg, specs, log=lambda *_: None)
    meta, y = bp.meta, bp.y
    meta = meta.assign(run_key=meta["subject"].astype(str) + meta["session"] + meta["run"].astype(str))
    first_in_run = meta.groupby("run_key")["trial"].transform("min") == meta["trial"]
    rows, perm_rows, preds_all = [], [], {}
    for filt, (kb, kc) in {"zero_phase": ("zp_bp", "zp_csp"), "causal": ("c_bp", "c_csp")}.items():
        for p in PROTOCOLS:
            name = f"precue_{filt}_{p}"
            if (OUT / name / "summary.json").exists():
                res, pred = pd.read_csv(OUT / name / "results_per_split.csv"), pd.read_csv(OUT / name / "predictions.csv")
            else:
                res, spl, pred = run_protocol(cfg, with_epochs(bp, bank[kb]), with_epochs(bp, bank[kc]), p, models=MODELS, log=lambda *_: None)
                save_experiment(cfg, res, spl, pred, bp, name, "exploratory", f"pre-cue negative control ({filt} filter), window cue{TMIN}..cue{TMAX} s", out_root=OUT,
                                extra={"window_s": [TMIN, TMAX], "filter": filt})
            preds_all[(filt, p)] = pred
            for m in MODELS:
                g = res[res.model == m]
                rows.append({"filter": filt, "protocol": p, "model": m, **{f"{k}": v for k, v in st.one_sample_vs_chance(g["balanced_accuracy"]).items()},
                             "n_above_threshold": int((g["accuracy"] > g["chance_threshold_p05"]).sum())})
                # within-run permutation test (all trials, then excluding the first trial of every run = causal-filter start-up)
                pr = pred[pred.model == m].merge(meta[["subject", "session", "trial", "run_key"]].rename(columns={"subject": "test_subject"}),
                                                 on=["test_subject", "session", "trial"], how="left")
                for label, mask in (("all_trials", np.ones(len(pr), bool)),
                                    ("excluding_first_trial_of_each_run", ~pr.merge(meta.assign(first=first_in_run)[["subject", "session", "trial", "first"]].rename(columns={"subject": "test_subject"}),
                                                                                  on=["test_subject", "session", "trial"], how="left")["first"].to_numpy(bool))):
                    q = pr[mask]
                    perm_rows.append({"filter": filt, "protocol": p, "model": m, "subset": label, **st.within_group_permutation_test(
                        q["y_true"].to_numpy(), q["y_pred"].to_numpy(), q["run_key"].to_numpy(), q["test_subject"].to_numpy(), n_perm=5000)})
            log("done", name)
    sm = pd.DataFrame(rows)
    sm["p_holm_within_filter_model"] = np.nan
    for _, g in sm.groupby(["filter", "model"]):
        sm.loc[g.index, "p_holm_within_filter_model"] = st.holm(g["p_wilcoxon"].to_numpy())
    sm.to_csv(PRE / "precue_one_sample_tests.csv", index=False)
    pm = pd.DataFrame(perm_rows)
    pm["p_holm_over_all_permutation_tests"] = st.holm(pm["p_one_sided"].to_numpy())
    pm.to_csv(PRE / "precue_permutation_tests.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(sm.round(3).to_string(index=False))
    print(pm.round(4).to_string(index=False))

    part = []
    for (filt, p), pred in preds_all.items():
        for (m, s), g in pred.groupby(["model", "test_subject"]):
            part.append({"filter": filt, "protocol": p, "model": m, "participant": s, "balanced_accuracy": compute_metrics(g["y_true"], g["y_pred"], g["score"])["balanced_accuracy"], "n_test": len(g)})
    pd.DataFrame(part).to_csv(PRE / "precue_participant_level.csv", index=False)

    # ---------------------------------------------------------------- 6: previous-trial (carry-over) control
    yprev = pc.previous_trial_labels(meta, labels, run_of)
    mask = np.isin(yprev, (1, 2))
    ycur_given_prev = pd.crosstab(yprev[mask], y[mask]).to_dict()
    rows = []
    for p in PROTOCOLS:
        for sp in make_splits(meta, y, p, cfg):
            tr, te = sp.train_idx[mask[sp.train_idx]], sp.test_idx[mask[sp.test_idx]]
            Xc = bank["zp_csp"]
            mdl = build_csp_lda_pipeline(cfg).fit(Xc[tr].astype(np.float64), (yprev[tr] - 1).astype(int))
            sc = decision_scores(mdl, Xc[te].astype(np.float64))
            m = compute_metrics((yprev[te] - 1).astype(int), (sc > 0).astype(int), sc)
            rows.append({"protocol": p, "participant": int(meta.iloc[sp.test_idx[0]]["subject"]), "target": "previous_trial_class",
                         "balanced_accuracy": m["balanced_accuracy"], "n_test": m["n"]})
    prev = pd.DataFrame(rows)
    prev.to_csv(PRE / "previous_trial_control.csv", index=False)
    prev_sum = [{"protocol": p, **st.one_sample_vs_chance(g["balanced_accuracy"])} for p, g in prev.groupby("protocol")]
    (PRE / "previous_trial_control_summary.json").write_text(json.dumps({"current_class_given_previous_class_counts": {str(k): v for k, v in ycur_given_prev.items()},
                                                                         "n_trials_with_LR_previous": int(mask.sum()), "decoding_previous_class": prev_sum}, indent=1, default=float), encoding="utf-8")
    log("previous-trial control:", json.dumps(prev_sum, default=float))

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pl = pd.read_csv(PRE / "precue_participant_level.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for j, p in enumerate(PROTOCOLS):
        for k, (filt, c) in enumerate((("zero_phase", "#1f77b4"), ("causal", "#d62728"))):
            g = pl[(pl["filter"] == filt) & (pl.protocol == p) & (pl.model == "csp_lda")].sort_values("participant")
            axes[j].plot(g["participant"] + (k - 0.5) * 0.15, g["balanced_accuracy"], "o", color=c, label=f"CSP+LDA, {filt.replace('_', '-')} filter")
        axes[j].axhline(0.5, color="k", ls="--", lw=0.7)
        axes[j].set_title(f"pre-cue window (cue-1.5..-0.5 s), protocol {'A' if p == 'cross_session' else 'B'}", fontsize=9)
        axes[j].set_xlabel("participant")
        axes[j].set_xticks(range(1, 10))
    axes[0].set_ylabel("balanced accuracy")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PRE / "fig5_precue_control.png", dpi=120)
    log("done")


if __name__ == "__main__":
    main()
