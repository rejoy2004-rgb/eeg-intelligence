"""Test of the carry-over explanation of the pre-cue score (exploratory).

Hypothesis H_carry: the pre-cue window of trial k still contains traces of the PREVIOUS trial's imagery (class k-1), and because
consecutive cues are negatively dependent (a left cue is more often followed by a right cue), a decoder of the previous class
scores above chance on the CURRENT class without any knowledge of the current trial.

Test: permute the true labels only among trials that have the SAME previous-trial class (and the same run). If H_carry explains the
whole effect, the observed score is no longer exceptional under this null; if the effect remains, carry-over through the cue
sequence is not the (only) explanation. The analytic expectation is also reported.
Usage: python scripts/analyze_precue_carryover.py   -> outputs/ocular_robust_v1/precue/carryover_test.json
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.dataset import load_external_labels
from src.evaluation import stats as st

cfg = load_config()
OUT = Path(cfg.resolve_path(cfg.evaluation["outputs_dir"])) / "ocular_robust_v1"
PRE = OUT / "precue"


def main():
    labels = {(s, ss): load_external_labels(cfg, s, ss) for s in range(1, 10) for ss in "TE"}
    seq = json.loads((PRE / "label_sequences.json").read_text())
    p_diff = 1.0 - seq["P(next=same | prev in {L,R}, next in {L,R})"]
    prev_dec = json.loads((PRE / "previous_trial_control_summary.json").read_text())["decoding_previous_class"]
    out = {"P_next_class_differs": p_diff, "tests": [], "analytic_expectation": {}}
    for r in prev_dec:                                       # expected accuracy if the whole effect were carry-over
        q = r["mean"]
        out["analytic_expectation"][r["protocol"]] = {"previous_class_decoding_ba": q, "expected_current_class_accuracy_from_carryover_only": (1 - p_diff) + (2 * p_diff - 1) * q}
    for filt in ("zero_phase", "causal"):
        for p in ("cross_session", "loso"):
            pr = pd.read_csv(OUT / f"precue_{filt}_{p}" / "predictions.csv")
            pr = pr[pr.model == "csp_lda"].copy()
            run = pr["trial"] // 48
            prev = [labels[(int(s), str(se))][t - 1] if (t % 48) != 0 else 0 for s, se, t in zip(pr.test_subject, pr.session, pr.trial)]
            same_run_group = pr["test_subject"].astype(str) + pr["session"] + run.astype(str)
            r1 = st.within_group_permutation_test(pr.y_true.to_numpy(), pr.y_pred.to_numpy(), same_run_group.to_numpy(), pr.test_subject.to_numpy(), n_perm=5000)
            strat = (same_run_group + "_" + pd.Series(prev, index=pr.index).astype(str)).to_numpy()
            r2 = st.within_group_permutation_test(pr.y_true.to_numpy(), pr.y_pred.to_numpy(), strat, pr.test_subject.to_numpy(), n_perm=5000)
            out["tests"].append({"filter": filt, "protocol": p, "observed_mean_ba": r1["observed_mean_ba"],
                                 "permute_within_run": {"null_mean": r1["null_mean"], "p": r1["p_one_sided"]},
                                 "permute_within_run_and_previous_class": {"null_mean": r2["null_mean"], "null_sd": r2["null_sd"], "p": r2["p_one_sided"]}})
    (PRE / "carryover_test.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
