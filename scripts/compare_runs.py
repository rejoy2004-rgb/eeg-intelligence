"""Compare two experiment folders and re-derive the summary statistics from the per-split CSV.

Usage: python scripts/compare_runs.py final_v1 repro_v1
Reports: (1) per-split metric differences between runs, (2) whether the means/SDs in summary.json equal an
independent recomputation from results_per_split.csv, (3) trial counts per model (same trials for all models?).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "outputs"
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]


def load(name):
    d = ROOT / name
    return pd.read_csv(d / "results_per_split.csv"), json.loads((d / "summary.json").read_text()), pd.read_csv(d / "predictions.csv")


def main(a: str, b: str) -> None:
    ra, sa, pa = load(a)
    rb, sb, pb = load(b)
    key = ["protocol", "test_subject", "model"]
    m = ra.merge(rb, on=key, suffixes=("_a", "_b"))
    print(f"rows: {a}={len(ra)} {b}={len(rb)} merged={len(m)}")
    for model, g in m.groupby("model"):
        diffs = {k: float(np.nanmax(np.abs(g[f"{k}_a"] - g[f"{k}_b"]))) for k in METRICS}
        same_n = bool((g["n_test_a"] == g["n_test_b"]).all())
        print(f"  {model:9s} max|diff| per metric: " + ", ".join(f"{k}={v:.4f}" for k, v in diffs.items()) + f" | same n_test: {same_n}")
    pm = pa.merge(pb, on=["protocol", "model", "test_subject", "session", "trial"], suffixes=("_a", "_b"))
    for model, g in pm.groupby("model"):
        print(f"  {model:9s} predictions: {len(g)} trials, identical y_pred in {(g['y_pred_a'] == g['y_pred_b']).mean():.4f}, "
              f"max|score diff| {np.nanmax(np.abs(g['score_a'] - g['score_b'])):.2e}")
    print("\nIndependent recomputation of summary.json statistics (run:", a, ")")
    worst = 0.0
    for protocol, g in ra.groupby("protocol"):
        for model, gm in g.groupby("model"):
            for k in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
                v = gm[k].dropna()
                ref = sa["aggregate"][protocol][model][k]
                mean_ok = np.isnan(ref["mean"]) and v.empty or abs(v.mean() - ref["mean"]) < 1e-12
                std = v.std(ddof=1) if len(v) > 1 else float("nan")
                std_ok = (np.isnan(std) and np.isnan(ref["std"])) or abs(std - ref["std"]) < 1e-12
                worst = max(worst, 0.0 if (mean_ok and std_ok) else 1.0)
                if not (mean_ok and std_ok):
                    print("  MISMATCH", protocol, model, k, v.mean(), ref)
    print("  all summary means/SDs match recomputation" if worst == 0 else "  SOME MISMATCHES")
    print("\nTrial sets per model (same intended trials?)")
    for protocol, g in pa.groupby("protocol"):
        sets = {m: set(zip(gm["test_subject"], gm["session"], gm["trial"])) for m, gm in g.groupby("model")}
        print(f"  {protocol}: identical test-trial set for all models: {len({frozenset(s) for s in sets.values()}) == 1}; "
              f"n = {len(next(iter(sets.values())))}")
    print("\nParticipants:", sorted(ra['test_subject'].unique()), "| retained trials:", sa["n_trials_total"], "| rejected:",
          len(pd.read_csv(ROOT / a / 'rejected_trials.csv')))


if __name__ == "__main__":
    main(*sys.argv[1:3])
