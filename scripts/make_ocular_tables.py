"""Markdown tables for docs/OCULAR_ROBUST_EVALUATION.md from the saved outputs/ocular_robust_v1/ files.

Usage: python scripts/make_ocular_tables.py [--embed]     (--embed rewrites the marked appendix of the document)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ocular_robust_v1"
PRE = OUT / "precue"
NAMES = {"baseline": "baseline (22 ch)", "frontal_exclusion": "frontal exclusion (16 ch)", "eog_regression_train_only": "EOG regression, train-only",
         "eog_regression_target_calibrated": "EOG regression, target-calibrated*"}
PNAME = {"cross_session": "A (same participant, unseen session)", "loso": "B (unseen participant, LOSO)"}
L = []
w = L.append


def ms(v):
    v = pd.Series(v).dropna()
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


# ---------------------------------------------------------------- EOG-only
eog = pd.read_csv(OUT / "eog_only_participant_level.csv")
es = pd.read_csv(OUT / "eog_only_summary.csv")
w("### EOG-only diagnostic (EOG used ONLY as a diagnostic; fixed standardised logistic regression, C = 1)")
w("")
w("| Participant | A: balanced accuracy | A: n test | B: balanced accuracy | B: n test | accuracy needed for p<0.05 (A / B, trials treated as independent) |")
w("|---|---|---|---|---|---|")
for s in range(1, 10):
    a, b = eog[(eog.protocol == "cross_session") & (eog.participant == s)].iloc[0], eog[(eog.protocol == "loso") & (eog.participant == s)].iloc[0]
    w(f"| S{s:02d} | {a.balanced_accuracy:.3f} | {int(a.n_test)} | {b.balanced_accuracy:.3f} | {int(b.n_test)} | {a.chance_threshold_p05:.3f} / {b.chance_threshold_p05:.3f} |")
w("")
for _, r in es.iterrows():
    w(f"* Protocol {PNAME[r.protocol]}: mean ± SD = {r['mean']:.3f} ± {r.sd:.3f} over {int(r.n_participants)} participants; exact Wilcoxon vs 0.5 p = {r.p_wilcoxon:.4f} "
      f"(Holm over the 2 protocols: {r.p_holm:.4f}); {int(r.n_above_chance_threshold)}/9 participants above their own chance threshold.")
w("")

# ---------------------------------------------------------------- strategies
pl = pd.read_csv(OUT / "participant_level_results.csv")
sm = pd.read_csv(OUT / "strategy_summary.csv")
pc = pd.read_csv(OUT / "paired_comparisons_vs_baseline.csv")
w("### Strategy comparison - balanced accuracy, mean ± SD over the 9 participants (min – max)")
w("")
w("| Protocol | Model | " + " | ".join(NAMES.values()) + " |")
w("|---|---|" + "---|" * len(NAMES))
for p in PNAME:
    for m in ("csp_lda", "logreg"):
        cells = []
        for s in NAMES:
            r = sm[(sm.protocol == p) & (sm.model == m) & (sm.strategy == s)].iloc[0]
            cells.append(f"{r['mean']:.3f} ± {r['std']:.3f} ({r['min']:.2f}–{r['max']:.2f})")
        w(f"| {PNAME[p]} | {m} | " + " | ".join(cells) + " |")
w("")
w("\\* uses the target recording's own unlabeled eye-movement calibration block - not zero-shot with respect to the target participant.")
w("")
for p in PNAME:
    w(f"**Per-participant balanced accuracy, CSP + LDA, protocol {PNAME[p]}**")
    w("")
    w("| Participant | " + " | ".join(NAMES.values()) + " |")
    w("|---|" + "---|" * len(NAMES))
    for s in range(1, 10):
        cells = [f"{pl[(pl.protocol == p) & (pl.model == 'csp_lda') & (pl.strategy == k) & (pl.participant == s)]['balanced_accuracy'].iloc[0]:.3f}" for k in NAMES]
        w(f"| S{s:02d} | " + " | ".join(cells) + " |")
    w("")
w("### Paired comparisons with the baseline (participant level, n = 9; Holm within the pre-declared family)")
w("")
w("| Protocol | Model | Strategy | Family | mean diff. | bootstrap 95 % CI | better / worse (participants) | exact Wilcoxon p | Holm-adjusted p |")
w("|---|---|---|---|---|---|---|---|---|")
for _, r in pc.iterrows():
    w(f"| {PNAME[r.protocol].split(' ')[0]} | {r.model} | {NAMES[r.strategy]} | {r.family} | {r.mean_diff:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {int(r.n_better)} / {int(r.n_worse)} | {r.p_wilcoxon:.3f} | {r.p_holm:.3f} |")
w("")
cd = pd.read_csv(OUT / "calibration_diagnostics.csv")
w("### Quality of the EOG regression on the calibration blocks (mean over the 18 recordings)")
w("")
w("| band | calibration length (s) | R² in-sample frontal / other | R² split-half frontal / other | task-window variance after / before correction, frontal / other |")
w("|---|---|---|---|---|")
for b, g in cd.groupby("band"):
    w(f"| {'4–40 Hz' if b == 'bp' else '8–30 Hz'} | {g.calibration_seconds.mean():.0f} | {g.r2_in_sample_frontal.mean():.2f} / {g.r2_in_sample_other.mean():.2f} | "
      f"{g.r2_split_half_frontal.mean():.2f} / {g.r2_split_half_other.mean():.2f} | {g.task_variance_ratio_frontal.mean():.2f} / {g.task_variance_ratio_other.mean():.2f} |")
w("")
w("Per participant (4–40 Hz): split-half R² frontal / task-variance ratio frontal — " + "; ".join(
    f"S{int(s):02d} {g.r2_split_half_frontal.mean():.2f}/{g.task_variance_ratio_frontal.mean():.2f}" for s, g in cd[cd.band == "bp"].groupby("subject")))
w("")

# ---------------------------------------------------------------- pre-cue
if (PRE / "precue_one_sample_tests.csv").exists():
    t = pd.read_csv(PRE / "precue_one_sample_tests.csv")
    pm = pd.read_csv(PRE / "precue_permutation_tests.csv")
    w("### Pre-cue negative control (window cue−1.5 … cue−0.5 s)")
    w("")
    w("| Filter | Protocol | Model | mean ± SD BA | participants above chance threshold | exact Wilcoxon vs 0.5 p | Holm (within filter and model) |")
    w("|---|---|---|---|---|---|---|")
    for _, r in t.iterrows():
        w(f"| {r['filter'].replace('_', '-')} | {PNAME[r.protocol].split(' ')[0]} | {r.model} | {r['mean']:.3f} ± {r.sd:.3f} | {int(r.n_above_threshold)}/9 | {r.p_wilcoxon:.3f} | {r.p_holm_within_filter_model:.3f} |")
    w("")
    w("Within-run label-permutation test (participants as the unit; one-sided; Holm over all 16 tests):")
    w("")
    w("| Filter | Protocol | Model | trials used | observed mean BA | null mean ± SD | p | Holm p |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in pm.iterrows():
        w(f"| {r['filter'].replace('_', '-')} | {PNAME[r.protocol].split(' ')[0]} | {r.model} | {r.subset.replace('_', ' ')} | {r.observed_mean_ba:.3f} | {r.null_mean:.3f} ± {r.null_sd:.3f} | {r.p_one_sided:.4f} | {r.p_holm_over_all_permutation_tests:.4f} |")
    w("")
if (PRE / "previous_trial_control_summary.json").exists():
    j = json.loads((PRE / "previous_trial_control_summary.json").read_text())
    w("### Carry-over control: decoding the PREVIOUS trial's class from the pre-cue window")
    w("")
    for r in j["decoding_previous_class"]:
        w(f"* Protocol {PNAME[r['protocol']]}: {r['mean']:.3f} ± {r['sd']:.3f} (exact Wilcoxon vs 0.5 p = {r['p_wilcoxon']:.3f}, n = {r['n_participants']}).")
    w(f"* Trials with a left/right previous trial: {j['n_trials_with_LR_previous']}; current class given previous class: {j['current_class_given_previous_class_counts']}.")
    w("")
if (PRE / "carryover_test.json").exists():
    j = json.loads((PRE / "carryover_test.json").read_text())
    w("### Does carry-over from the previous trial explain the pre-cue score? (CSP + LDA; permutation null conditioned on the previous trial's class)")
    w("")
    w(f"P(next class differs from previous class) = {j['P_next_class_differs']:.3f}. Analytic expectation of the pre-cue balanced accuracy if the whole effect were carry-over: "
      + "; ".join(f"{('A' if k == 'cross_session' else 'B')} {v['expected_current_class_accuracy_from_carryover_only']:.3f} (previous-class decoding {v['previous_class_decoding_ba']:.3f})" for k, v in j["analytic_expectation"].items()) + ".")
    w("")
    w("| Filter | Protocol | observed mean BA | null: permute within run (mean, p) | null: permute within run AND previous class (mean ± SD, p) |")
    w("|---|---|---|---|---|")
    for t in j["tests"]:
        a, b = t["permute_within_run"], t["permute_within_run_and_previous_class"]
        w(f"| {t['filter'].replace('_', '-')} | {'A' if t['protocol'] == 'cross_session' else 'B'} | {t['observed_mean_ba']:.3f} | {a['null_mean']:.3f}, p = {a['p']:.4f} | {b['null_mean']:.3f} ± {b['null_sd']:.3f}, p = {b['p']:.4f} |")
    w("")
if (PRE / "sequence_sharing_analysis.json").exists():
    j = json.loads((PRE / "sequence_sharing_analysis.json").read_text())
    w("### Cue-sequence sharing between recordings")
    w("")
    w(f"* {j['n_distinct_sequences']} distinct 288-trial cue sequences among 18 recordings: " + "; ".join("[" + ", ".join(v) + "]" for v in j["recordings_per_sequence"]))
    w("")
    w("| Analysis | identical-order participants | mean | different-order participants | mean | rho (p, uncorrected) |")
    w("|---|---|---|---|---|---|")
    for r in j["comparisons"]:
        if "mean_identical" in r:
            w(f"| {r['analysis']} | {r['identical_order_participants']} | {r['mean_identical']:.3f} | {r['different_order_participants']} | {r['mean_different']:.3f} | |")
        else:
            w(f"| {r['analysis']} | | | | | {r['rho']:+.2f} ({r['p_uncorrected']:.2f}) |")
    w("")

text = "\n".join(L)
(OUT / "tables.md").write_text(text, encoding="utf-8")
if "--embed" in sys.argv:
    doc = ROOT / "docs" / "OCULAR_ROBUST_EVALUATION.md"
    d = doc.read_text(encoding="utf-8")
    a, b = "<!-- BEGIN TABLES -->", "<!-- END TABLES -->"
    doc.write_text(d[: d.index(a) + len(a)] + "\n\n" + text + "\n" + d[d.index(b):], encoding="utf-8")
    print("embedded")
else:
    print(text)
