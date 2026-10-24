"""Does the sharing of cue sequences between recordings explain (a) the pre-cue score or (b) inflate the main results?

Background (measured): the 18 recordings contain only 4 distinct 288-trial cue sequences; 13 recordings share one identical order.
If EEG features vary with serial position (slow drift, fatigue, run-onset effects) a model trained on recordings with the same order
as the test recording could exploit position -> class. This script compares results for participants whose training and test
recordings do / do not share a sequence. n = 9 participants: everything here is DESCRIPTIVE and cannot establish absence of an effect.

Usage: python scripts/analyze_sequence_sharing.py   (needs outputs/ocular_robust_v1/precue and final_v1)
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config
from src.data.dataset import load_external_labels

cfg = load_config()
ROOT = Path(cfg.resolve_path(cfg.evaluation["outputs_dir"]))
PRE = ROOT / "ocular_robust_v1" / "precue"


def main():
    labels = {(s, ss): load_external_labels(cfg, s, ss) for s in range(1, 10) for ss in "TE"}
    keys = sorted(labels)
    uniq = {}
    for k in keys:
        uniq.setdefault(labels[k].tobytes(), []).append(f"S{k[0]:02d}{k[1]}")
    seq_id = {k: i for i, g in enumerate(uniq.values()) for k in keys if f"S{k[0]:02d}{k[1]}" in g}
    agree = lambda a, b: float((labels[a] == labels[b]).mean())
    # protocol A: does the participant's training session (T) share its order with the test session (E)?
    a_same = {s: agree((s, "T"), (s, "E")) for s in range(1, 10)}
    # protocol B: mean agreement of the held-out participant's sessions with all training participants' sessions
    b_agree = {s: float(np.mean([agree((s, x), (o, y)) for x in "TE" for o in range(1, 10) if o != s for y in "TE"])) for s in range(1, 10)}
    out = {"n_distinct_sequences": len(uniq), "recordings_per_sequence": list(uniq.values()),
           "A_agreement_T_vs_E": a_same, "B_mean_agreement_with_training_recordings": b_agree}

    pl = pd.read_csv(PRE / "precue_participant_level.csv")
    prim = pd.read_csv(ROOT / "final_v1" / "results_per_split.csv")
    rows = []
    for filt in ("zero_phase", "causal"):
        g = pl[(pl["filter"] == filt) & (pl.protocol == "cross_session") & (pl.model == "csp_lda")].set_index("participant")["balanced_accuracy"]
        same = [s for s in g.index if a_same[s] == 1.0]
        diff = [s for s in g.index if a_same[s] < 1.0]
        rows.append({"analysis": f"pre-cue CSP+LDA, protocol A, {filt}", "identical_order_participants": same, "mean_identical": float(g[same].mean()),
                     "different_order_participants": diff, "mean_different": float(g[diff].mean())})
        gb = pl[(pl["filter"] == filt) & (pl.protocol == "loso") & (pl.model == "csp_lda")].set_index("participant")["balanced_accuracy"]
        rho, p = stats.spearmanr([b_agree[s] for s in gb.index], gb.values)
        rows.append({"analysis": f"pre-cue CSP+LDA, protocol B, {filt}: Spearman(agreement with training recordings, BA)", "rho": float(rho), "p_uncorrected": float(p), "n": 9})
    for m in ("csp_lda", "logreg", "cnn"):
        g = prim[(prim.protocol == "cross_session") & (prim.model == m)].set_index("test_subject")["balanced_accuracy"]
        same = [s for s in g.index if a_same[s] == 1.0]
        diff = [s for s in g.index if a_same[s] < 1.0]
        rows.append({"analysis": f"MAIN task-window result {m}, protocol A (final_v1)", "identical_order_participants": same, "mean_identical": float(g[same].mean()),
                     "different_order_participants": diff, "mean_different": float(g[diff].mean())})
        gb = prim[(prim.protocol == "loso") & (prim.model == m)].set_index("test_subject")["balanced_accuracy"]
        rho, p = stats.spearmanr([b_agree[s] for s in gb.index], gb.values)
        rows.append({"analysis": f"MAIN task-window result {m}, protocol B: Spearman(agreement with training recordings, BA)", "rho": float(rho), "p_uncorrected": float(p), "n": 9})
    out["comparisons"] = rows
    (PRE / "sequence_sharing_analysis.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
