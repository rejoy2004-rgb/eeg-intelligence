"""Ocular-robust evaluation (design frozen in configs/ocular_robust_v1.yaml). Outputs: outputs/ocular_robust_v1/

Usage: python scripts/run_ocular_robust.py <stage> [<stage> ...]
    bank        build (or load from data/cache) the epoch bank incl. EOG-corrected variants; writes calibration diagnostics
    eog         EOG-only diagnostic (protocols A and B, fixed classifier, no selection)
    strategies  baseline / frontal exclusion / EOG regression (train-only, target-calibrated) x {CSP+LDA, logreg} x {A, B}
    report      statistics (Holm), CSV summaries and figures from the saved runs
Safety: the established baseline (final_v1, repro_v1, repro_v2) is hash-checked before and after every stage and can never be written to.
"""
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT, load_config
from src.evaluation import stats as st
from src.evaluation.csp_selection import make_splits
from src.evaluation.experiments import PROTECTED_RUNS, run_protocol, save_experiment
from src.evaluation.metrics import chance_threshold, compute_metrics
from src.evaluation.ocular_bank import FRONTAL, OcularBank, build_ocular_bank, cache_key, load_bank, save_bank
from src.signal.epoching import TrialSet

cfg = load_config()
CONFIG_FILE = PROJECT_ROOT / "configs" / "ocular_robust_v1.yaml"
PLAN = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
OUT = Path(cfg.resolve_path(cfg.evaluation["outputs_dir"])) / PLAN["experiment"]
CACHE = PROJECT_ROOT / "data" / "cache" / "ocular_bank.npz"
PROTOCOLS = ("cross_session", "loso")
MODELS = tuple(PLAN["models"])
STRATEGIES = ("baseline", "frontal_exclusion", "eog_regression_train_only", "eog_regression_target_calibrated")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint() -> str:
    """No git repository is available, so the 'code revision' is a hash of all project source files."""
    h = hashlib.sha256()
    for f in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        h.update(f.relative_to(PROJECT_ROOT).as_posix().encode() + f.read_bytes())
    return h.hexdigest()


def baseline_manifest() -> dict:
    root = Path(cfg.resolve_path(cfg.evaluation["outputs_dir"]))
    return {r: {f.name: sha(f) for f in sorted((root / r).glob("*")) if f.is_file()} for r in sorted(PROTECTED_RUNS)}


def check_baseline(tag: str):
    """Create the manifest on first use; afterwards require it to be unchanged."""
    OUT.mkdir(parents=True, exist_ok=True)
    f, now = OUT / "baseline_manifest.json", baseline_manifest()
    if f.exists():
        if json.loads(f.read_text(encoding="utf-8")) != now:
            raise RuntimeError("BASELINE OUTPUTS CHANGED - refusing to continue")
        log(f"baseline manifest verified ({tag})")
    else:
        f.write_text(json.dumps(now, indent=1), encoding="utf-8")
        log("baseline manifest written")


def dataset_manifest() -> dict:
    """SHA-256 of all 18 GDF and 18 label files (written once; identifies the exact dataset used)."""
    f = OUT / "dataset_manifest.json"
    if not f.exists():
        files = sorted(cfg.resolve_path(cfg.dataset["raw_dir"]).glob("*.gdf")) + sorted(cfg.resolve_path(cfg.dataset["labels_dir"]).glob("*.mat"))
        f.write_text(json.dumps({x.name: sha(x) for x in files}, indent=1), encoding="utf-8")
    return json.loads(f.read_text(encoding="utf-8"))


def provenance() -> dict:
    return {"experiment": PLAN["experiment"], "config_sha256": {"ocular_robust_v1.yaml": sha(CONFIG_FILE), "default.yaml": sha(PROJECT_ROOT / "configs" / "default.yaml")},
            "source_sha256": source_fingerprint(), "code_revision_note": "no git repository; source_sha256 hashes all src/**/*.py",
            "dataset": cfg.dataset["name"], "dataset_manifest_sha256": hashlib.sha256((OUT / "dataset_manifest.json").read_bytes()).hexdigest() if dataset_manifest() else None,
            "seed": PLAN["seed"], "baseline_manifest_sha256": hashlib.sha256((OUT / "baseline_manifest.json").read_bytes()).hexdigest()}


def get_bank() -> OcularBank:
    key = cache_key(CONFIG_FILE, PROJECT_ROOT / "configs" / "default.yaml", PROJECT_ROOT / "src" / "signal" / "ocular.py",
                    PROJECT_ROOT / "src" / "evaluation" / "ocular_bank.py", PROJECT_ROOT / "src" / "signal" / "preprocessing.py")
    bank = load_bank(CACHE, key)
    if bank is None:
        log("building ocular bank (about 4 minutes)")
        bank = build_ocular_bank(cfg, log=lambda *_: None)
        save_bank(bank, CACHE, key)
    else:
        log("loaded ocular bank from cache")
    prim = pd.read_csv(Path(cfg.resolve_path(cfg.evaluation["outputs_dir"])) / "final_v1" / "predictions.csv")
    n = len(bank.bp.meta)
    assert n == 2346 and set(bank.bp.meta["subject"]) == set(range(1, 10)), "bank does not hold the primary 2346 trials"
    return bank


def stage_bank():
    bank = get_bank()
    OUT.mkdir(parents=True, exist_ok=True)
    bank.diagnostics.to_csv(OUT / "calibration_diagnostics.csv", index=False)
    d = bank.diagnostics
    log("calibration: split-half R2 frontal / other (mean over recordings):",
        d.groupby("band")[["r2_split_half_frontal", "r2_split_half_other", "task_variance_ratio_frontal", "task_variance_ratio_other"]].mean().round(3).to_dict())


def stage_eog():
    bank = get_bank()
    meta, y = bank.bp.meta, bank.bp.y
    rows = []
    for p in PROTOCOLS:
        for sp in make_splits(meta, y, p, cfg):
            tr, te = sp.train_idx, sp.test_idx
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(bank.eog[tr], y[tr])
            s = clf.decision_function(bank.eog[te])
            m = compute_metrics(y[te], (s > 0).astype(int), s)
            sub = int(meta.iloc[te[0]]["subject"])
            rows.append({"protocol": p, "participant": sub, "balanced_accuracy": m["balanced_accuracy"], "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
                         "n_train": len(tr), "n_test": m["n"], "chance_threshold_p05": chance_threshold(m["n"])})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "eog_only_participant_level.csv", index=False)
    log("EOG-only diagnostic saved")
    print(df.pivot(index="participant", columns="protocol", values="balanced_accuracy").round(3))


def _subset(ts: TrialSet, keep_names):
    idx = [ts.ch_names.index(c) for c in keep_names]
    return TrialSet(ts.X[:, idx], ts.y, ts.meta, ts.rejections, ts.sfreq, [ts.ch_names[i] for i in idx], dict(ts.summary))


def stage_strategies():
    bank = get_bank()
    check_baseline("before strategies")
    base_bp, base_csp = bank.trial_set("base", "bp"), bank.trial_set("base", "csp")
    keep = [c for c in base_bp.ch_names if c not in FRONTAL]
    prov = provenance()
    prim = pd.read_csv(Path(cfg.resolve_path(cfg.evaluation["outputs_dir"])) / "final_v1" / "results_per_split.csv")

    def datasets(strategy, protocol):
        """-> (bp, csp, per_split_data or None)"""
        if strategy == "baseline":
            return base_bp, base_csp, None
        if strategy == "frontal_exclusion":
            return _subset(base_bp, keep), _subset(base_csp, keep), None
        if strategy == "eog_regression_target_calibrated":
            return bank.trial_set("own", "bp"), bank.trial_set("own", "csp"), None
        # train-only regression
        if protocol == "cross_session":
            return bank.trial_set("strict_a", "bp"), bank.trial_set("strict_a", "csp"), None
        own_bp, own_csp = bank.trial_set("own", "bp"), bank.trial_set("own", "csp")
        subj = own_bp.meta["subject"].to_numpy()

        def per_split(test_subject):
            m = subj == test_subject
            out = []
            for own, band in ((own_bp, "bp"), (own_csp, "csp")):
                X = own.X.copy()
                X[m] = bank.arrays[f"strict_b_{band}"][m]           # held-out participant: coefficients from the OTHER participants only
                out.append(TrialSet(X, own.y, own.meta, own.rejections, own.sfreq, own.ch_names, dict(own.summary)))
            return tuple(out)
        return own_bp, own_csp, per_split

    for strategy in STRATEGIES:
        for p in PROTOCOLS:
            name = f"{strategy}_{p}"
            if (OUT / name / "summary.json").exists():
                log("exists, skipping", name)
                continue
            bp, csp, fn = datasets(strategy, p)
            res, spl, pred = run_protocol(cfg, bp, csp, p, models=MODELS, log=lambda *_: None, per_split_data=fn)
            if strategy == "baseline":                       # the harness must reproduce final_v1 exactly
                ref = prim[(prim.protocol == p) & prim.model.isin(MODELS)].set_index(["test_subject", "model"])["balanced_accuracy"]
                new = res.set_index(["test_subject", "model"])["balanced_accuracy"]
                diff = float((ref - new.reindex(ref.index)).abs().max())
                log(f"baseline vs final_v1 ({p}) max |diff| = {diff:.2e}")
                if diff > 1e-12:
                    raise RuntimeError("baseline strategy does not reproduce final_v1")
            extra = {**prov, "strategy": strategy, "strategy_definition": PLAN["strategies"].get(strategy),
                     "protocol": p, "models": list(MODELS),
                     "zero_shot_wrt_target_participant": strategy != "eog_regression_target_calibrated"}
            save_experiment(cfg, res, spl, pred, bp, name, "exploratory",
                            f"ocular_robust_v1 strategy={strategy}; design frozen in configs/ocular_robust_v1.yaml", out_root=OUT, extra=extra)
            log("saved", name, res.groupby("model")["balanced_accuracy"].mean().round(3).to_dict())
    check_baseline("after strategies")


# ------------------------------------------------------------------------------------------- report
def stage_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    check_baseline("report")
    OUT.mkdir(parents=True, exist_ok=True)
    long, preds = [], {}
    for strategy in STRATEGIES:
        for p in PROTOCOLS:
            d = OUT / f"{strategy}_{p}"
            r = pd.read_csv(d / "results_per_split.csv")
            r["strategy"] = strategy
            long.append(r)
            preds[(strategy, p)] = pd.read_csv(d / "predictions.csv")
    df = pd.concat(long, ignore_index=True)
    cols = ["protocol", "strategy", "model", "test_subject", "n_test", "balanced_accuracy", "accuracy", "f1", "roc_auc", "precision", "recall", "specificity", "tn", "fp", "fn", "tp"]
    df[cols].rename(columns={"test_subject": "participant"}).to_csv(OUT / "participant_level_results.csv", index=False)

    # identical test trials for every paired comparison (hard check)
    for p in PROTOCOLS:
        for m in MODELS:
            tables = [preds[(s, p)][preds[(s, p)]["model"] == m] for s in STRATEGIES]
            n = st.assert_same_test_trials(*tables)
            log(f"{p}/{m}: all {len(STRATEGIES)} strategies scored on identical {n} test trials")

    summ = df.groupby(["protocol", "model", "strategy"])["balanced_accuracy"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    summ.to_csv(OUT / "strategy_summary.csv", index=False)

    # paired comparisons vs baseline, Holm within pre-declared families
    fam = {("csp_lda", "frontal_exclusion"): "primary", ("csp_lda", "eog_regression_train_only"): "primary",
           ("csp_lda", "eog_regression_target_calibrated"): "secondary"}
    rows = []
    for p in PROTOCOLS:
        for m in MODELS:
            base = df[(df.protocol == p) & (df.model == m) & (df.strategy == "baseline")].set_index("test_subject")["balanced_accuracy"]
            for s in STRATEGIES[1:]:
                other = df[(df.protocol == p) & (df.model == m) & (df.strategy == s)].set_index("test_subject")["balanced_accuracy"]
                rows.append({"protocol": p, "model": m, "strategy": s, "family": fam.get((m, s), "tertiary"), **st.paired_comparison(other, base)})
    pc = pd.DataFrame(rows)
    pc["p_holm"] = np.nan
    for f, g in pc.groupby("family"):
        pc.loc[g.index, "p_holm"] = st.holm(g["p_wilcoxon"].to_numpy())
    pc.to_csv(OUT / "paired_comparisons_vs_baseline.csv", index=False)

    eog = pd.read_csv(OUT / "eog_only_participant_level.csv")
    rows = []
    for p, g in eog.groupby("protocol"):
        rows.append({"protocol": p, **st.one_sample_vs_chance(g["balanced_accuracy"]), "n_above_chance_threshold": int((g["accuracy"] > g["chance_threshold_p05"]).sum())})
    es = pd.DataFrame(rows)
    es["p_holm"] = st.holm(es["p_wilcoxon"].to_numpy())
    es.to_csv(OUT / "eog_only_summary.csv", index=False)

    # ---- figures
    colors = {"baseline": "#444", "frontal_exclusion": "#d95f02", "eog_regression_train_only": "#1b9e77", "eog_regression_target_calibrated": "#7570b3"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for i, m in enumerate(MODELS):
        for j, p in enumerate(PROTOCOLS):
            ax = axes[i, j]
            for k, s in enumerate(STRATEGIES):
                g = df[(df.protocol == p) & (df.model == m) & (df.strategy == s)].sort_values("test_subject")
                ax.plot(g["test_subject"] + (k - 1.5) * 0.12, g["balanced_accuracy"], "o", color=colors[s], label=s, ms=5)
            ax.axhline(0.5, color="k", lw=0.6, ls="--")
            ax.set_title(f"{m} - {'A: same participant, unseen session' if p == 'cross_session' else 'B: unseen participant (LOSO)'}", fontsize=9)
            ax.set_xticks(range(1, 10))
            ax.set_xlabel("participant")
            ax.set_ylabel("balanced accuracy")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_participant_level_balanced_accuracy.png", dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for j, p in enumerate(PROTOCOLS):
        ax = axes[j]
        pos = 0
        ticks, labels = [], []
        for m in MODELS:
            base = df[(df.protocol == p) & (df.model == m) & (df.strategy == "baseline")].set_index("test_subject")["balanced_accuracy"]
            for s in STRATEGIES[1:]:
                d = (df[(df.protocol == p) & (df.model == m) & (df.strategy == s)].set_index("test_subject")["balanced_accuracy"] - base)
                ax.plot(np.full(len(d), pos) + np.linspace(-0.15, 0.15, len(d)), d.values, "o", color=colors[s], ms=4)
                row = pc[(pc.protocol == p) & (pc.model == m) & (pc.strategy == s)].iloc[0]
                ax.errorbar(pos, row.mean_diff, yerr=[[row.mean_diff - row.ci_low], [row.ci_high - row.mean_diff]], color="k", capsize=3, marker="D", ms=5)
                ticks.append(pos)
                labels.append(f"{m}\n{s.replace('eog_regression_', 'EOG ').replace('_', ' ')}")
                pos += 1
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_title(f"paired difference vs baseline - protocol {'A' if p == 'cross_session' else 'B'}\n(dots = participants, diamond = mean with bootstrap 95% CI)", fontsize=8)
        ax.set_ylabel("Delta balanced accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_paired_differences_vs_baseline.png", dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for j, p in enumerate(PROTOCOLS):
        e = eog[eog.protocol == p].set_index("participant")["balanced_accuracy"]
        b = df[(df.protocol == p) & (df.model == "csp_lda") & (df.strategy == "baseline")].set_index("test_subject")["balanced_accuracy"]
        axes[j].scatter(e, b.reindex(e.index))
        for s in e.index:
            axes[j].annotate(f"S{s:02d}", (e[s], b[s]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axes[j].axvline(0.5, color="k", lw=0.5, ls="--")
        axes[j].set_xlabel("EOG-only balanced accuracy (diagnostic)")
        axes[j].set_ylabel("EEG CSP+LDA balanced accuracy")
        axes[j].set_title(f"protocol {'A' if p == 'cross_session' else 'B'}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_eog_only_vs_eeg_csp.png", dpi=120)

    d = pd.read_csv(OUT / "calibration_diagnostics.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d["r2_split_half_other"], d["r2_split_half_frontal"], c=(d["band"] == "csp"), cmap="coolwarm", s=18)
    ax.set_xlabel("split-half R2 of EOG regression, non-frontal channels")
    ax.set_ylabel("split-half R2, frontal channels")
    ax.set_title("EOG regression quality on the calibration block\n(36 recording x band points; red = 8-30 Hz)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_calibration_quality.png", dpi=120)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(summ.round(3).to_string(index=False))
    print(pc.round(3).to_string(index=False))
    print(es.round(3).to_string(index=False))
    (OUT / "provenance.json").write_text(json.dumps(provenance(), indent=1), encoding="utf-8")
    check_baseline("end of report")


STAGES = {"bank": stage_bank, "eog": stage_eog, "strategies": stage_strategies, "report": stage_report}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    for w in wanted:
        if w not in STAGES:
            raise SystemExit(f"unknown stage {w}; choose from {list(STAGES)}")
    check_baseline("start")
    for w in wanted:
        log("=== stage", w)
        STAGES[w]()
    log("done")
