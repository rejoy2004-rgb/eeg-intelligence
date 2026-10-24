# Ocular-robust evaluation and pre-cue investigation (`ocular_robust_v1`)

*Completes the two open items of `docs/RESEARCH_AUDIT.md` §11: (1) is the left-vs-right EEG decoding contaminated by ocular activity, and
(2) what explains the small above-chance score of the pre-cue negative control.*

Evidence labels: **[confirmed]** measured with verifiable outputs and checked by a control · **[exploratory]** measured but not pre-declared, or low power ·
**[unresolved]** open. Nothing below was tuned on outer-test results. All outputs are in `outputs/ocular_robust_v1/`; the established baseline
(`final_v1`, `repro_v1`, `repro_v2`) is hash-checked before and after every stage and cannot be written to.

## 1. What was already complete (verified at the start of this work)

| Item | Evidence |
|---|---|
| `final_v1` baseline, `repro_v1`, `repro_v2` | `scripts/compare_runs.py`: all 3 529 predictions identical, means/SDs recomputable, same test trials for all models |
| Runtime leakage audit (18 checks × 18 folds), negative controls, S08 analysis, nested CSP selection, alignment, ERD, CNN variants | `outputs/audit_v1`, `audit_v2`, `audit_v3` |
| A **descriptive** EOG-only and frontal-excluded diagnostic | `outputs/audit_v1/ocular_confound.csv` — not a pre-specified strategy comparison, no paired outputs |
| Zero-phase pre-cue control | `outputs/audit_v2/precue_control_*` |

Planned but **not** existing before this work: an EOG-correction method; a paired strategy comparison with identical outer splits and a frozen design;
a causal-filter pre-cue control; any explanation of the pre-cue score.

## 2. Frozen design (`configs/ocular_robust_v1.yaml`)

Written before any outer-test result of the strategies below existed; its SHA-256 is stored in every run folder (`summary.json → config_sha256`).
**Disclosure:** descriptive test-set numbers for frontal exclusion (A 0.784, B 0.646) and for EOG-only decoding had already been seen in the audit, so these two are *not* blind
confirmations; the frontal channel set (Fz, FC3, FC1, FCz, FC2, FC4) had been written into an earlier script before those numbers existed and no other set was tried.
The EOG regression and the pre-cue investigation had not been run before.

| Strategy | Definition | Target data used? |
|---|---|---|
| baseline | all 22 EEG channels (must equal `final_v1` exactly; it does: max |diff| = 0) | – |
| frontal exclusion | drop Fz, FC3, FC1, FCz, FC2, FC4 (rows 1–2 of the montage, closest to the eyes) | no |
| EOG regression, **train-only** | OLS from the 3 EOG channels to every EEG channel, fitted on the dataset's own eye-movement calibration block (event 1072; 80–166 s, recorded before the trials, unlabeled). A: coefficients of the participant's training session T; B: mean of the coefficients of the 8 training participants | **no** |
| EOG regression, **target-calibrated** | same, but coefficients from the target recording's own calibration block | calibration block of the target (unlabeled, separate from every trial) → **not zero-shot** |

Same outer splits, same 2 346 trials, same preprocessing and model settings as `final_v1` (CSP + LDA and band-power logistic regression; the CNN was deliberately not
part of this evaluation). The regression is fitted after band-passing EEG and EOG identically (per gap-free segment), with no intercept and no regulariser (nothing to tune).
Calibration blocks end before the first trial in all 18 files (checked in code, `calibration_slice` raises otherwise).

## 3. Findings

### 3.1 EOG-only diagnostic [confirmed]
Fixed standardised logistic regression on 6 EOG features (mean shift vs pre-cue baseline and log variance of the 3 EOG channels, task window), same splits as the EEG analyses:

* **Protocol A**: 0.602 ± 0.096 (exact Wilcoxon vs 0.5: Holm-adjusted p = 0.020); 4/9 participants above their own chance threshold.
* **Protocol B**: 0.637 ± 0.073 (Holm-adjusted p = 0.008); **9/9** participants above their chance threshold.
* Participants S05, S07, S09 (and S08) have the strongest EOG-only scores (0.64–0.76).

*What this establishes:* the eye channels carry left/right class information, consistently enough to transfer across participants (B) — eye or gaze behaviour is related to the cue direction.
*What it does not establish:* that the EEG classifier uses it. EOG-only accuracy does not measure the EEG model's reliance on ocular activity.

### 3.2 Ocular-robust strategies [confirmed for the tested interventions; interpretation limited]
(Participant-level tables, CIs, Holm-adjusted tests are in the appendix; figures `fig1`, `fig2`.)

| Balanced accuracy (mean ± SD, 9 participants) | baseline | frontal exclusion | EOG regression, train-only | EOG regression, target-calibrated* |
|---|---|---|---|---|
| CSP + LDA, A | 0.783 ± 0.143 | 0.784 ± 0.144 | 0.776 ± 0.148 | 0.778 ± 0.148 |
| CSP + LDA, B | 0.632 ± 0.150 | 0.646 ± 0.141 | 0.629 ± 0.150 | 0.631 ± 0.152 |
| Logistic regression, A | 0.645 ± 0.117 | 0.658 ± 0.117 | 0.655 ± 0.125 | 0.658 ± 0.129 |
| Logistic regression, B | 0.575 ± 0.054 | 0.565 ± 0.068 | 0.586 ± 0.073 | 0.577 ± 0.084 |

* **No intervention changed CSP + LDA beyond noise**: mean differences −0.007 … +0.014, every bootstrap CI includes 0, all Holm-adjusted p ≥ 0.66 (primary family of 4 tests: 0.66–0.80).
  The participant-level bootstrap intervals span about ±0.01 (EOG regression) to ±0.04 (frontal exclusion), so an ocular dependence much larger than ≈ 0.04 balanced accuracy would probably have been visible; smaller ones cannot be excluded with n = 9.
* **Heterogeneity**: individual participants change a lot even though the mean does not (frontal exclusion, CSP + LDA: S04 A 0.667 → 0.802, S01 B 0.529 → 0.626, but 6 of 9 participants are slightly worse in A).
  The positive mean differences are carried by a few participants (S04 in A: +0.135; S01 +0.097 and S06 +0.051 in B); the median change is −0.007 (A) and +0.004 (B), i.e. about zero.
* The EOG regression is effective as a regression on average (frontal split-half R² ≈ 0.39–0.40, other channels ≈ 0.24; task-window frontal variance falls to ≈ 0.52–0.60 of its value; see appendix) yet leaves decoding unchanged. Its quality is very uneven across participants (frontal split-half R² from 0.01 for S09 to 0.65 for S08), so for some participants the correction is close to ineffective.
* S08 (the 0.955 outlier) is unchanged by every strategy (0.955 → 0.959): another piece of evidence that its signal is not a frontal/ocular artifact.

*Interpretation (careful):* removing the linear EOG-predictable component, or the most ocular-exposed channels, does not reduce CSP + LDA accuracy. That is *compatible with* the decoding not
relying on ocular activity, but it is **not proof**, because (i) the monopolar EOG electrodes also record brain activity and share the mastoid reference, so the regression removes a mixture and can over-correct
(for S08 it removed ≈ 63–65 % of the task-window variance in frontal and ≈ 44–47 % in the other channels, 4–40 Hz — we therefore do not treat the correction as a clean ocular estimate); (ii) volume-conducted ocular activity also reaches posterior channels that remain;
(iii) a linear correction does not remove non-linear or non-ocular gaze-related effects; (iv) with 9 participants small effects are undetectable.

### 3.3 Pre-cue negative control [partly explained]
The audit reported CSP + LDA balanced accuracy 0.537 (A) and 0.521 (B) from the window cue−1.5 … cue−0.5 s (re-run here: identical numbers).

| Candidate explanation | Result | Verdict |
|---|---|---|
| Post-cue information enters the window through the **zero-phase filter** | impulse test: energy of the filter response located earlier than the window end ≈ 3×10⁻⁶ (4–40 Hz) / 2×10⁻⁷ (8–30 Hz); earliest 1 % of the response 0.21–0.23 s before an event vs the 0.5 s margin. Re-run with a strictly **causal** filter: 0.542 (A), 0.521 (B) | **excluded** |
| Window boundaries / timing | window = trial start + 0.5 s … cue − 0.5 s in all trials; starts 2.1–3.0 s after the previous trial's imagery ended; no overlap with gaps, trial start or cue (0 of 2 346) | **excluded** |
| Label alignment, trial order bookkeeping, split overlap | verified for all 18 files (`audit_v1/label_integrity.json`), runtime audit passed; logistic regression on the same epochs is at chance (0.488 / 0.484 zero-phase, 0.497 / 0.484 causal) | **no defect found** |
| Start-up transient of the causal filter | repeated excluding the first trial of every run: unchanged (0.541 / 0.522) | **excluded** |
| Pure chance | within-run label-permutation test (participants as unit): p = 0.0004 – 0.007 (A, B; causal and zero-phase); Holm over 16 tests: A 0.006–0.018, B 0.026–0.059 | **not chance**: a small but real association (≈ +0.02 … +0.04 BA), CSP + LDA only |
| **Sharing of cue sequences between recordings** | only 4 distinct 288-trial sequences exist among the 18 recordings; 13 recordings share one identical order. Pre-cue score (zero-phase) for participants whose T and E orders are identical: 0.538; for different orders: 0.537 (causal filter: 0.555 vs 0.526) | **no consistent support** as the explanation (n = 5 vs 4, low power) |
| **Carry-over of the previous trial + serial dependence of the cue order** | consecutive left/right cues are negatively dependent (P(next differs) = 0.571, χ² p = 8×10⁻⁷); the previous trial's class can be decoded from the pre-cue window (A 0.601 ± 0.146, p = 0.16; B 0.572 ± 0.080, p = 0.027, uncorrected); analytic expectation of the resulting current-class score 0.514 (A) / 0.510 (B); permuting labels only among trials with the same previous class raises the null mean to 0.510–0.515 | **explains most of it**: B excess no longer significant (p = 0.16 zero-phase, 0.10 causal); in **A a residual ≈ 0.03 remains** (p = 0.014 / 0.002) → **[unresolved]** |

Consequences, stated narrowly: (a) the pre-cue score is **not** evidence that post-cue information leaks into pre-cue windows; (b) it **is** evidence that the dataset's cue order contains exploitable
structure (history dependence) that a CSP-based decoder partly picks up; (c) the same mechanism could add a few hundredths of balanced accuracy to the task-window results — their size cannot be separated
from true motor-related decoding with the present data; the effect sizes of the main results (0.63–0.78) are far larger, and the task-window result of identical-order vs different-order participants is equal
(A, CSP + LDA: 0.782 vs 0.783), but this is a low-power descriptive comparison.

### 3.4 Dataset finding: only four distinct cue sequences [confirmed]
S01E, S02E/T, S03E/T, S05E/T, S06E, S07E/T, S08E/T, S09E share exactly the same order; S01T = S04E; S04T and S06T = S09T are separate. Any position- or history-linked nuisance structure is therefore
shared between sessions and participants, which can only favour the decoder in protocols A and B (it is not an issue within a single randomly ordered recording). This should be reported with any result on this
dataset; it motivates evaluating on a dataset with independently randomised orders (§6).

## 4. Tests added (full suite: see `docs/TESTING.md`)

`tests/test_ocular_precue.py` (24 tests): causal filter cannot see later samples; impulse leakage profile; pre-cue window geometry; **control sensitivity** (a post-cue effect is invisible to the pre-cue window
but detected when the window overlaps it); label-sequence statistics; EOG regression recovers planted coefficients, uses only the calibration block, keeps gaps, validates shapes; calibration block must exist,
be long enough and not overlap trials; frontal exclusion tied to the frozen config; **paired comparisons require identical test trials**; Holm / bootstrap / permutation-test behaviour; **held-out labels never influence
predictions** (flipping a held-out participant's labels leaves its predictions and scores bit-identical, both protocols); target adaptation data kept separate; **baseline runs cannot be overwritten** (even with `overwrite=True`).

## 5. Limitations

* 9 participants, one dataset, public E-session labels. All paired tests are exploratory, indicative only.
* The EOG regression is linear, fitted on a short calibration block, and can remove neural signal; the target-calibrated variant is not zero-shot.
* Frontal exclusion discards neural signal as well as ocular signal; posterior channels still receive volume-conducted ocular activity.
* The unresolved A residual in the pre-cue control (≈ 0.03) has no confirmed explanation; additional sequence/history structure or session effects are possible.
* The pre-cue investigation steps beyond those in the frozen config (previous-trial decoding, carry-over-conditioned permutation, sequence-sharing comparison) were added after seeing the sequence statistics and are exploratory.
* The CNN was not evaluated here.
* No claim is made about clinical validity or about what participants thought or did.

## 6. Next steps (research, not required for this deliverable)
Evaluate on a dataset with independently randomised, history-balanced cue orders; add an order-conditioned permutation null (previous-class strata) to the main task-window results; test ocular robustness with ICA/regression fitted per fold on more participants.

## 7. Reproduce

```bash
python -m pytest                                              # whole suite (needs the real data for 5 tests)
python scripts/run_ocular_robust.py bank eog strategies report   # ~25 min; bank is cached in data/cache (git-ignored)
python scripts/investigate_precue.py                          # ~15 min
python scripts/analyze_sequence_sharing.py
python scripts/analyze_precue_carryover.py
python scripts/make_ocular_tables.py --embed                  # regenerates the appendix of this file
```
Seeds: 42 (+ participant id for any CNN; none used here); permutation tests use `numpy.random.default_rng(0)` with 5 000 permutations; bootstrap: 10 000 resamples, seed 0.
Provenance stored per run: SHA-256 of both configs, of all `src/**/*.py` (no git repository exists), of the 18 GDF + 18 label files (`dataset_manifest.json`) and of the baseline files (`baseline_manifest.json`).

## Output files (`outputs/ocular_robust_v1/`)

| File | Content |
|---|---|
| `participant_level_results.csv` | every participant × protocol × strategy × model: balanced accuracy, accuracy, F1, ROC-AUC, confusion counts |
| `strategy_summary.csv`, `paired_comparisons_vs_baseline.csv` | mean/SD/min/max; paired differences, bootstrap CIs, Wilcoxon p, Holm p |
| `eog_only_participant_level.csv`, `eog_only_summary.csv` | EOG-only diagnostic |
| `calibration_diagnostics.csv` | R² and variance-ratio diagnostics of the EOG regression per recording and band |
| `<strategy>_<protocol>/` | complete run folders (results, predictions, splits, rejected trials, summary with provenance) |
| `precue_*` run folders, `precue/` | pre-cue control runs; `timing_audit.json`, `label_sequences.json`, `filter_leakage_profile.json`, `precue_one_sample_tests.csv`, `precue_permutation_tests.csv`, `precue_participant_level.csv`, `previous_trial_control*.{csv,json}`, `sequence_sharing_analysis.json`, `carryover_test.json` |
| `fig1…fig4*.png`, `precue/fig5_precue_control.png` | participant-level scores; paired differences; EOG-only vs EEG; calibration quality; pre-cue control |
| `baseline_manifest.json`, `dataset_manifest.json`, `provenance.json`, `tables.md` | provenance and the generated tables |

---

## Appendix: generated tables

*Produced by `python scripts/make_ocular_tables.py --embed` from the files above. Do not edit by hand.*

<!-- BEGIN TABLES -->

### EOG-only diagnostic (EOG used ONLY as a diagnostic; fixed standardised logistic regression, C = 1)

| Participant | A: balanced accuracy | A: n test | B: balanced accuracy | B: n test | accuracy needed for p<0.05 (A / B, trials treated as independent) |
|---|---|---|---|---|---|
| S01 | 0.483 | 141 | 0.584 | 279 | 0.574 / 0.552 |
| S02 | 0.556 | 142 | 0.608 | 278 | 0.577 / 0.554 |
| S03 | 0.563 | 137 | 0.562 | 274 | 0.577 / 0.555 |
| S04 | 0.479 | 116 | 0.573 | 245 | 0.586 / 0.555 |
| S05 | 0.672 | 135 | 0.675 | 264 | 0.578 / 0.553 |
| S06 | 0.575 | 108 | 0.581 | 221 | 0.593 / 0.561 |
| S07 | 0.764 | 140 | 0.738 | 273 | 0.579 / 0.553 |
| S08 | 0.635 | 134 | 0.652 | 266 | 0.582 / 0.553 |
| S09 | 0.692 | 130 | 0.757 | 246 | 0.577 / 0.557 |

* Protocol A (same participant, unseen session): mean ± SD = 0.602 ± 0.096 over 9 participants; exact Wilcoxon vs 0.5 p = 0.0195 (Holm over the 2 protocols: 0.0195); 4/9 participants above their own chance threshold.
* Protocol B (unseen participant, LOSO): mean ± SD = 0.637 ± 0.073 over 9 participants; exact Wilcoxon vs 0.5 p = 0.0039 (Holm over the 2 protocols: 0.0078); 9/9 participants above their own chance threshold.

### Strategy comparison - balanced accuracy, mean ± SD over the 9 participants (min – max)

| Protocol | Model | baseline (22 ch) | frontal exclusion (16 ch) | EOG regression, train-only | EOG regression, target-calibrated* |
|---|---|---|---|---|---|
| A (same participant, unseen session) | csp_lda | 0.783 ± 0.143 (0.61–0.95) | 0.784 ± 0.144 (0.58–0.95) | 0.776 ± 0.148 (0.57–0.96) | 0.778 ± 0.148 (0.58–0.95) |
| A (same participant, unseen session) | logreg | 0.645 ± 0.117 (0.50–0.82) | 0.658 ± 0.117 (0.54–0.83) | 0.655 ± 0.125 (0.51–0.82) | 0.658 ± 0.129 (0.49–0.83) |
| B (unseen participant, LOSO) | csp_lda | 0.632 ± 0.150 (0.49–0.95) | 0.646 ± 0.141 (0.49–0.96) | 0.629 ± 0.150 (0.50–0.96) | 0.631 ± 0.152 (0.49–0.96) |
| B (unseen participant, LOSO) | logreg | 0.575 ± 0.054 (0.49–0.67) | 0.565 ± 0.068 (0.46–0.69) | 0.586 ± 0.073 (0.51–0.75) | 0.577 ± 0.084 (0.49–0.76) |

\* uses the target recording's own unlabeled eye-movement calibration block - not zero-shot with respect to the target participant.

**Per-participant balanced accuracy, CSP + LDA, protocol A (same participant, unseen session)**

| Participant | baseline (22 ch) | frontal exclusion (16 ch) | EOG regression, train-only | EOG regression, target-calibrated* |
|---|---|---|---|---|
| S01 | 0.857 | 0.850 | 0.836 | 0.843 |
| S02 | 0.606 | 0.577 | 0.613 | 0.592 |
| S03 | 0.948 | 0.948 | 0.940 | 0.948 |
| S04 | 0.667 | 0.802 | 0.659 | 0.676 |
| S05 | 0.615 | 0.608 | 0.574 | 0.581 |
| S06 | 0.684 | 0.655 | 0.674 | 0.683 |
| S07 | 0.795 | 0.757 | 0.816 | 0.816 |
| S08 | 0.948 | 0.948 | 0.955 | 0.948 |
| S09 | 0.923 | 0.915 | 0.915 | 0.915 |

**Per-participant balanced accuracy, CSP + LDA, protocol B (unseen participant, LOSO)**

| Participant | baseline (22 ch) | frontal exclusion (16 ch) | EOG regression, train-only | EOG regression, target-calibrated* |
|---|---|---|---|---|
| S01 | 0.529 | 0.626 | 0.518 | 0.518 |
| S02 | 0.491 | 0.494 | 0.495 | 0.495 |
| S03 | 0.714 | 0.688 | 0.692 | 0.681 |
| S04 | 0.596 | 0.554 | 0.600 | 0.604 |
| S05 | 0.512 | 0.523 | 0.516 | 0.516 |
| S06 | 0.532 | 0.583 | 0.532 | 0.532 |
| S07 | 0.616 | 0.640 | 0.601 | 0.605 |
| S08 | 0.955 | 0.959 | 0.959 | 0.959 |
| S09 | 0.740 | 0.743 | 0.747 | 0.769 |

### Paired comparisons with the baseline (participant level, n = 9; Holm within the pre-declared family)

| Protocol | Model | Strategy | Family | mean diff. | bootstrap 95 % CI | better / worse (participants) | exact Wilcoxon p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|---|
| A | csp_lda | frontal exclusion (16 ch) | primary | +0.002 | [-0.021, +0.038] | 1 / 6 | 0.266 | 0.797 |
| A | csp_lda | EOG regression, train-only | primary | -0.007 | [-0.018, +0.004] | 3 / 6 | 0.164 | 0.656 |
| A | csp_lda | EOG regression, target-calibrated* | secondary | -0.004 | [-0.014, +0.005] | 2 / 5 | 0.469 | 0.938 |
| A | logreg | frontal exclusion (16 ch) | tertiary | +0.013 | [-0.003, +0.026] | 7 / 2 | 0.129 | 0.773 |
| A | logreg | EOG regression, train-only | tertiary | +0.009 | [-0.010, +0.027] | 7 / 2 | 0.301 | 1.000 |
| A | logreg | EOG regression, target-calibrated* | tertiary | +0.013 | [-0.007, +0.031] | 7 / 2 | 0.250 | 1.000 |
| B | csp_lda | frontal exclusion (16 ch) | primary | +0.014 | [-0.009, +0.041] | 7 / 2 | 0.301 | 0.797 |
| B | csp_lda | EOG regression, train-only | primary | -0.003 | [-0.009, +0.003] | 5 / 3 | 0.742 | 0.797 |
| B | csp_lda | EOG regression, target-calibrated* | secondary | -0.001 | [-0.011, +0.009] | 5 / 3 | 0.945 | 0.945 |
| B | logreg | frontal exclusion (16 ch) | tertiary | -0.010 | [-0.035, +0.008] | 4 / 5 | 0.910 | 1.000 |
| B | logreg | EOG regression, train-only | tertiary | +0.011 | [-0.008, +0.032] | 6 / 3 | 0.570 | 1.000 |
| B | logreg | EOG regression, target-calibrated* | tertiary | +0.002 | [-0.023, +0.032] | 4 / 5 | 0.910 | 1.000 |

### Quality of the EOG regression on the calibration blocks (mean over the 18 recordings)

| band | calibration length (s) | R² in-sample frontal / other | R² split-half frontal / other | task-window variance after / before correction, frontal / other |
|---|---|---|---|---|
| 4–40 Hz | 139 | 0.47 / 0.27 | 0.39 / 0.23 | 0.60 / 0.76 |
| 8–30 Hz | 139 | 0.47 / 0.27 | 0.40 / 0.24 | 0.52 / 0.71 |

Per participant (4–40 Hz): split-half R² frontal / task-variance ratio frontal — S01 0.31/0.74; S02 0.53/0.48; S03 0.58/0.44; S04 0.38/0.74; S05 0.44/0.66; S06 0.28/0.64; S07 0.35/0.67; S08 0.65/0.36; S09 0.01/0.68

### Pre-cue negative control (window cue−1.5 … cue−0.5 s)

| Filter | Protocol | Model | mean ± SD BA | participants above chance threshold | exact Wilcoxon vs 0.5 p | Holm (within filter and model) |
|---|---|---|---|---|---|---|
| zero-phase | A | logreg | 0.488 ± 0.041 | 0/9 | 0.652 | 0.652 |
| zero-phase | A | csp_lda | 0.537 ± 0.052 | 2/9 | 0.074 | 0.109 |
| zero-phase | B | logreg | 0.484 ± 0.026 | 0/9 | 0.164 | 0.328 |
| zero-phase | B | csp_lda | 0.521 ± 0.032 | 1/9 | 0.055 | 0.109 |
| causal | A | logreg | 0.497 ± 0.040 | 0/9 | 0.910 | 0.910 |
| causal | A | csp_lda | 0.542 ± 0.040 | 1/9 | 0.020 | 0.039 |
| causal | B | logreg | 0.484 ± 0.030 | 0/9 | 0.098 | 0.195 |
| causal | B | csp_lda | 0.521 ± 0.030 | 1/9 | 0.055 | 0.055 |

Within-run label-permutation test (participants as the unit; one-sided; Holm over all 16 tests):

| Filter | Protocol | Model | trials used | observed mean BA | null mean ± SD | p | Holm p |
|---|---|---|---|---|---|---|---|
| zero-phase | A | logreg | all trials | 0.488 | 0.499 ± 0.014 | 0.7852 | 1.0000 |
| zero-phase | A | logreg | excluding first trial of each run | 0.489 | 0.499 ± 0.014 | 0.7672 | 1.0000 |
| zero-phase | A | csp_lda | all trials | 0.537 | 0.499 ± 0.013 | 0.0014 | 0.0182 |
| zero-phase | A | csp_lda | excluding first trial of each run | 0.540 | 0.499 ± 0.013 | 0.0006 | 0.0084 |
| zero-phase | B | logreg | all trials | 0.484 | 0.500 ± 0.009 | 0.9574 | 1.0000 |
| zero-phase | B | logreg | excluding first trial of each run | 0.481 | 0.500 ± 0.010 | 0.9752 | 1.0000 |
| zero-phase | B | csp_lda | all trials | 0.521 | 0.501 ± 0.007 | 0.0042 | 0.0462 |
| zero-phase | B | csp_lda | excluding first trial of each run | 0.522 | 0.502 ± 0.007 | 0.0022 | 0.0264 |
| causal | A | logreg | all trials | 0.497 | 0.499 ± 0.014 | 0.5553 | 1.0000 |
| causal | A | logreg | excluding first trial of each run | 0.497 | 0.499 ± 0.014 | 0.5599 | 1.0000 |
| causal | A | csp_lda | all trials | 0.542 | 0.499 ± 0.013 | 0.0004 | 0.0064 |
| causal | A | csp_lda | excluding first trial of each run | 0.541 | 0.499 ± 0.013 | 0.0004 | 0.0064 |
| causal | B | logreg | all trials | 0.484 | 0.500 ± 0.009 | 0.9650 | 1.0000 |
| causal | B | logreg | excluding first trial of each run | 0.482 | 0.500 ± 0.010 | 0.9688 | 1.0000 |
| causal | B | csp_lda | all trials | 0.521 | 0.501 ± 0.008 | 0.0058 | 0.0580 |
| causal | B | csp_lda | excluding first trial of each run | 0.522 | 0.502 ± 0.008 | 0.0066 | 0.0594 |

### Carry-over control: decoding the PREVIOUS trial's class from the pre-cue window

* Protocol A (same participant, unseen session): 0.601 ± 0.146 (exact Wilcoxon vs 0.5 p = 0.164, n = 9).
* Protocol B (unseen participant, LOSO): 0.572 ± 0.080 (exact Wilcoxon vs 0.5 p = 0.027, n = 9).
* Trials with a left/right previous trial: 1065; current class given previous class: {'0': {'1.0': 218, '2.0': 313}, '1': {'1.0': 286, '2.0': 248}}.

### Does carry-over from the previous trial explain the pre-cue score? (CSP + LDA; permutation null conditioned on the previous trial's class)

P(next class differs from previous class) = 0.571. Analytic expectation of the pre-cue balanced accuracy if the whole effect were carry-over: A 0.514 (previous-class decoding 0.601); B 0.510 (previous-class decoding 0.572).

| Filter | Protocol | observed mean BA | null: permute within run (mean, p) | null: permute within run AND previous class (mean ± SD, p) |
|---|---|---|---|---|
| zero-phase | A | 0.537 | 0.499, p = 0.0014 | 0.511 ± 0.012, p = 0.0142 |
| zero-phase | B | 0.521 | 0.501, p = 0.0042 | 0.515 ± 0.006, p = 0.1588 |
| causal | A | 0.542 | 0.499, p = 0.0004 | 0.510 ± 0.011, p = 0.0020 |
| causal | B | 0.521 | 0.501, p = 0.0058 | 0.512 ± 0.007, p = 0.0958 |

### Cue-sequence sharing between recordings

* 4 distinct 288-trial cue sequences among 18 recordings: [S01E, S02E, S02T, S03E, S03T, S05E, S05T, S06E, S07E, S07T, S08E, S08T, S09E]; [S01T, S04E]; [S04T]; [S06T, S09T]

| Analysis | identical-order participants | mean | different-order participants | mean | rho (p, uncorrected) |
|---|---|---|---|---|---|
| pre-cue CSP+LDA, protocol A, zero_phase | [2, 3, 5, 7, 8] | 0.538 | [1, 4, 6, 9] | 0.537 | |
| pre-cue CSP+LDA, protocol B, zero_phase: Spearman(agreement with training recordings, BA) | | | | | +0.52 (0.15) |
| pre-cue CSP+LDA, protocol A, causal | [2, 3, 5, 7, 8] | 0.555 | [1, 4, 6, 9] | 0.526 | |
| pre-cue CSP+LDA, protocol B, causal: Spearman(agreement with training recordings, BA) | | | | | +0.19 (0.62) |
| MAIN task-window result csp_lda, protocol A (final_v1) | [2, 3, 5, 7, 8] | 0.782 | [1, 4, 6, 9] | 0.783 | |
| MAIN task-window result csp_lda, protocol B: Spearman(agreement with training recordings, BA) | | | | | +0.06 (0.89) |
| MAIN task-window result logreg, protocol A (final_v1) | [2, 3, 5, 7, 8] | 0.656 | [1, 4, 6, 9] | 0.633 | |
| MAIN task-window result logreg, protocol B: Spearman(agreement with training recordings, BA) | | | | | -0.31 (0.41) |
| MAIN task-window result cnn, protocol A (final_v1) | [2, 3, 5, 7, 8] | 0.609 | [1, 4, 6, 9] | 0.603 | |
| MAIN task-window result cnn, protocol B: Spearman(agreement with training recordings, BA) | | | | | -0.44 (0.24) |

<!-- END TABLES -->
