# Research audit of EEG Intelligence

*Independent re-examination of the experimental design, the existing `final_v1` results and the code, carried out after the project reported 139 passing tests.
Passing tests were not taken as evidence of scientific correctness. Every statement below is tagged by how it was established:*
**[reproduced]** re-ran existing results · **[measured]** new experiment, results in `outputs/audit_v1|v2|v3` · **[verified]** checked in code or data ·
**[tested]** covered by an automated test · **[hypothesis]** plausible, not established · **[unresolved]** open.

`outputs/final_v1/` was **not modified**. All new experiments live in `outputs/audit_v1/`, `outputs/audit_v2/`, `outputs/audit_v3/`, `outputs/repro_v1/`, `outputs/repro_v2/`.

## 0. Summary

| Question | Answer |
|---|---|
| Can `final_v1` be reproduced? | **Yes, exactly** [reproduced]. All 72 per-split rows, all 3 529 predictions and the CNN scores are bit-identical on a second run (`repro_v1`); reported means/SDs equal an independent recomputation. |
| Is there data leakage in the fitting path? | **None found** [verified, measured]. A runtime audit that spies on what every `fit` really receives passed 18 checks in each of the 18 outer folds; four negative controls (permuted training labels, pre-cue window, label-flipped test participant, injected-leak detection) behave as they should. |
| Is the S08 result (0.955) an artifact of the split or of one training participant? | **No** [measured]. Stable under dropping any training participant (0.910–0.963), in both sessions (0.947 / 0.963), and equal to its within-participant score (0.948). Cause of S08's strong signal is **undetermined** (§5). |
| Did nested model selection improve CSP + LDA? | **No reliable gain** [measured]: A 0.764 nested vs 0.783 default; B 0.647 vs 0.632 (differences within noise). |
| Did anything improve cross-participant performance? | **Euclidean alignment** raised protocol B from 0.632 to 0.730, but it uses *unlabeled data of the target participant*: it is **not zero-shot** and was an exploratory single configuration (§6). |
| Is the CNN broken? | **No implementation defect found** [measured]; it is simply not better than a logistic regression on the same band-power features (§7). It remains below CSP + LDA, and that result is retained. |
| Channel names? | **Confirmed against the official Figure 3** (numbering, rows, 5 named anchors) — see §4. This resolves an item the earlier documentation listed as unconfirmed. Electrode *positions* remain template positions. |
| New risk found | **Ocular / gaze leakage is possible in this dataset**: EOG-only decoding is above chance for several participants (§4.4). The primary pipeline does not remove ocular activity; this is a validity threat, not a leak between splits. **Follow-up (`docs/OCULAR_ROBUST_EVALUATION.md`)**: frontal-channel exclusion and EOG regression (train-only and target-calibrated) left CSP + LDA unchanged (all Holm p ≥ 0.66); EOG-only decoding is above chance (B: 0.637, 9/9 participants). |

## 1. Method

1. Read all prioritised source files, configs, tests and docs; compared documentation with behaviour.
2. **Runtime leakage audit** (`src/evaluation/leakage_audit.py`, `scripts/audit_leakage.py`): wraps `fit_csp_lda`, `fit_logreg`, `images_from_epochs` and `ImageScaler.fit` inside the *real* `_fit_predict_split`, identifies each trial by a hash of its epoch samples, and checks identities and fitted statistics fold by fold.
3. **Independent recomputation** of labels, trial counts and rejections straight from GDF annotations and `.mat` files (`scripts/verify_labels_all.py`), and of every summary statistic from the per-split CSV (`scripts/compare_runs.py`).
4. **Negative controls**: permuted training labels; pre-cue window decoding; flipped labels of the held-out participant (selection must not change); an injected leak that the audit must catch.
5. **New experiments** (`scripts/run_audit_experiments.py`), each with a fixed, written-down design, all reported including those that did not help.
6. No preprocessing, hyper-parameter or protocol setting was changed before reproduction was recorded.

## 2. Reproduction of the existing results [reproduced]

* `repro_v1` (unchanged code and config): 72/72 per-split rows identical, 3 529/3 529 predictions identical, scores equal to 0.0 absolute difference (CNN included — same machine, same library versions).
* Reported means and SDs recomputed from `results_per_split.csv` equal `summary.json` to 1e-12.
* 9 participants, 2 346 retained and 246 rejected left/right trials (all 246 are expert-marked artifacts; 2 592 = 9 × 2 × 144), protocol A tests 1 183 E-session trials, protocol B tests all 2 346; all four models are scored on the identical test-trial set in each protocol.
* `repro_v2`: the final, post-audit code (refactored `experiments.py`) reproduces `final_v1` bit-for-bit (same checks as `repro_v1`).

## 3. Leakage audit — one entry per suspected mechanism

| # | Suspected mechanism | Where | Status | Evidence | Correction / effect |
|---|---|---|---|---|---|
| A | CSP fitted before splitting or on test data | `models/baseline.py::build_csp_lda_pipeline`, `fit_csp_lda`; called from `experiments._fit_predict_split` with `csp.X[tr]` | **Not confirmed** | Spy: the trials passed to CSP = exactly the training trials in 18/18 folds; CSP filters equal a fresh refit on training trials only; they differ from filters fitted on train + test. | None needed. Effect on results: none. |
| B1 | Band-power scaler fitted on test data | `StandardScaler` inside the logistic-regression `Pipeline` | **Not confirmed** | Scaler mean equals the training mean (1e-9) and differs from the train + test mean, 18/18 folds. | None needed. |
| B2 | Spatial image normalisation fitted on test data (the original EEGLearn weakness) | `features/spatial.py::ImageScaler`, used in `_fit_predict_split` | **Not confirmed** | Mean/SD equal those of the CNN-fit partition only; differ from train + test statistics; fit/validation/test image batches pairwise disjoint (also by participant in B). | None needed. |
| C1 | Hyper-parameter selection on test data: logistic-regression C | `models/baseline.py::fit_logreg` | **Not confirmed** | Grouped inner CV on the training set (groups = runs in A, participants in B; verified that no test participant appears in any inner group). | None. |
| C2 | CSP components, bands, epoch window, CNN settings, 200 µV rule, etc. chosen using test results | `configs/default.yaml` | **Not confirmed, but cannot be verified from code alone** | They are conventional literature values written before the final run; a smoke run on S01–S02 (protocol A) had been seen (documented in `EXPERIMENTS.md`). The nested analysis (§5) shows the default is neither favoured nor disfavoured: nested selection by inner CV gave A −0.018, B +0.016 relative to the default. | Documented; nested estimate added. |
| C3 | CNN epoch selection optimistic | `train_cnn` | **Confirmed (benign)** | Validation BA at the selected epoch (A 0.708, B 0.587) exceeds test BA (0.606, 0.545): a maximum over epochs on a small validation set is optimistically biased. The *test* data are untouched. | Reported in §7; not a leak. |
| D1 | Participant overlap in protocol B | `splits.py::loso_splits` | **Not confirmed** | Run-time `assert_no_overlap(by="subject")` plus audit: train/validation/test participants pairwise disjoint in 9/9 folds. | None. |
| D2 | Session overlap in protocol A | `splits.py::cross_session_split` | **Not confirmed** | Train sessions = {T}, test sessions = {E} in 9/9 folds; validation = last training run. | None. |
| D3 | Duplicate / overlapping epochs | `epoching.py` | **Not confirmed** | 0 duplicate epochs among 2 346 (content hash); one 2 s epoch per trial, trials ≥ 8 s apart; every kept epoch lies outside all missing-data gaps [tested]. | None. |
| E1 | Label/event/ordering errors | `epoching.py::locate_trials`, `data/dataset.py` | **Not confirmed** | Independent re-derivation for all 18 files: 288 starts/cues, cue exactly 500 samples after start, T-file event codes = `.mat` labels, 72 trials per class per file, artifact markers coincide with trial starts, and the expected 246 rejections match trial-for-trial. | None. |
| E2 | Models compared on different trials | `epoching.py::build_dual_band_trials` | **Not confirmed** | Identical test-trial sets for all four models in both protocols (recomputed from `predictions.csv`). | None. |
| F1 | Amplitude/artifact rejection uses test epochs | `build_trial_set` | **By design, not leakage** | Fixed rule, no learned parameter. It does change the *composition* of the test set (artifact-free trials only, as in the official competition scoring). | Documented. |
| F2 | Public E-session labels | dataset | **Inherent limitation** | Labels have been public since 2008; no untouched test set exists. | Documented. |
| F3 | **Alignment uses unlabeled target data** | new `features/alignment.py` | **By design, explicitly flagged** | No labels used, but the target participant's unlabeled EEG is. | Results labelled "not zero-shot" everywhere. |
| F4 | Trial-level inference treats trials as independent | `metrics.py::wilson_interval`, `chance_threshold` | **Confirmed statistical caveat** | Trials within a session are serially correlated; the per-participant CI and chance threshold are therefore optimistic. | Stated in `LIMITATIONS.md`; inference across participants (n = 9) uses paired participant-level statistics. |

**Negative controls [measured]** (`outputs/audit_v2/permuted_labels_*`, `precue_control_*`):

* *Permuted training labels* (test labels intact), all models, both protocols: logistic regression 0.497 / 0.516, CSP + LDA 0.472 / 0.499, CNN 0.475 / 0.496 (A / B), majority 0.500 — i.e. at chance, as required if no label information reaches the model other than through the training labels.
* *Pre-cue window (cue−1.5 … cue−0.5 s, nothing about the class is known yet)*: logistic regression 0.488 / 0.484; CSP + LDA 0.537 ± 0.052 / 0.521 ± 0.032 with 2/9 and 1/9 participants above their per-participant chance threshold. The logistic regression is at chance; CSP + LDA is slightly above 0.5 on average. **Follow-up (`docs/OCULAR_ROBUST_EVALUATION.md` §3.3)**: not caused by the filter (causal filter: 0.542 / 0.521; the zero-phase backward spread is ≈ 3×10⁻⁶ of the energy) and not chance (within-run permutation p ≤ 0.007); it is largely explained by carry-over from the previous trial combined with the negatively dependent cue order (conditioning the null on the previous class lifts it to 0.510–0.515), leaving a residual ≈ 0.03 in protocol A that remains **[unresolved]**. The dataset also contains only 4 distinct cue sequences across the 18 recordings.
* *Flipped labels of the held-out participant* → inner CV scores and the selected configuration are unchanged, outer scores change [tested].
* *Injected leak* (CSP or scaler additionally fitted on test-participant trials) → the audit fails [tested]. The audit is therefore able to detect leaks of this kind.

## 4. Data assumptions

### 4.1 Between-run gaps [verified]
All 18 files: gaps are exactly 100 samples of −1600 µV on all 22 channels, beginning at each run-start event (8 gaps; 6 in A04T). They become NaN, are never filled, the filter treats each gap-free segment separately, and **no kept epoch overlaps a gap** [tested on synthetic structure; checked for all kept real trials through the epoch-extraction validity test].

### 4.2 Filtering [verified, tested]
Zero-phase Butterworth per gap-free segment along the time axis; edge effects are confined to segment ends, which lie outside every epoch (in the files inspected the first trial starts 350 samples after the run start; epochs begin ≥ 2.5 s after trial start). Non-causal, hence offline only.

### 4.3 Channel order and positions
* **Order [verified against the official document]**: Figure 3 of `desc_2a.pdf` shows 22 numbered electrodes in rows {1}, {2–6}, {7–13}, {14–18}, {19–21}, {22} with **#1 = Fz, #8 = C3, #10 = Cz, #12 = C4, #20 = Pz** labelled. The files carry exactly these five names at 0-based positions 0, 7, 9, 11, 19, i.e. the file order equals the figure numbering. The figure does not print the other 17 names; they follow from their position in the 10-20 grid (e.g. #2–#6 are the row above C3…C4 → FC3 FC1 FCz FC2 FC4). Together with the data-driven check (correlation falls with distance: Spearman −0.86 … −0.95 vs ≈ −0.05 for random relabellings) this is strong, but the 17 names remain **inferred from figure geometry**, not printed labels. A swap of two neighbouring unnamed channels would hardly change either check.
* **Positions**: MNE `colin27_1020` template positions, **not** participant measurements. For spatial interpolation and the CNN this means the 2-D map is an approximation of the scalp layout; individual head shapes and cap placements are ignored. The measured-only layout grid (§7) uses the figure topology and no template coordinates.
* **No coordinates were invented**; an unknown channel name raises an error.

### 4.4 Ocular confound [measured] — new finding
The cue is a left/right arrow visible until t = 3.25 s, so gaze shifts time-locked to the cue can carry the class. Using the EOG channels **only as a diagnostic**:
horizontal-EOG shift d (right vs left) is large for S05 (0.90), S07 (1.16), S09 (1.06); EOG-only classifiers (never used in any model) reach balanced accuracy 0.67–0.76 (S05, S07, S09), 0.64–0.65 (S08) in at least one protocol. The primary CSP + LDA, which includes the frontal channels, is therefore **not guaranteed to be free of ocular contributions** for those participants. The diagnostic that drops the six frontal channels gives mean balanced accuracy A 0.784 vs 0.783, B 0.646 vs 0.632 — frontal channels are not what makes the primary result work on average, but individual participants change (e.g. S04 A 0.667 → 0.802, S01 B 0.529 → 0.626). This is a descriptive diagnostic: it was not used to select a model. **[unresolved]**: a pre-specified ocular-robust pipeline (§11).

## 5. The S08 result [measured]

CSP + LDA, protocol B, balanced accuracy 0.955 (confusion matrix: 128 left correct, 4 left → right, 8 right → left, 126 right correct).

| Check | Result | Reading |
|---|---|---|
| Reproducible | identical in `repro_v1` | yes |
| Sessions | T 0.947, E 0.963 | both high |
| Dropping any one of the 8 training participants | 0.910 – 0.963 | not driven by one training participant |
| Train on others' T sessions only, test S08's E session | 0.970 | different split, same conclusion |
| Own T → own E (protocol A) | 0.948 | the signal is present within S08 as well; rank correlation of A and B CSP + LDA across participants ρ = +0.87 (p = 0.002, exploratory, n = 9) |
| Trial-bootstrap 95 % CI | 0.929 – 0.977 | too narrow (trials not independent) but shows the result is not a small-sample fluke |
| Data quality | rejection 7.6 % (middle of the range 3–23 %), no flat channels, highest median epoch peak-to-peak (51.8 µV) and second-highest RMS | high amplitude; not obviously poor quality |
| Textbook C3/C4 mu lateralisation | **small** (d = 0.08; S03 has d = 1.97) | the result is *not* the classical single-channel mu effect |
| Ocular | EOG-only 0.64 / 0.65; CSP without frontal channels 0.948 / 0.959 | ocular leakage is **not** the main source |
| Channel groups (CSP restricted) A / B | central row 0.90 / 0.83; centro-parietal 0.94 / 0.88; sensorimotor (12 ch) 0.96 / 0.86; posterior only 0.90 / 0.74; frontal only 0.74 / 0.77 | information is **spread over many regions**; sensorimotor sites suffice but so do posterior sites |
| Own CSP pattern (descriptive) | strongest channels CP4, P2, CP2, C6 / FCz, FC3, Cz, C3 | right centro-parietal weighting; not a focal C3/C4 pattern |

**Conclusions**: the S08 number is real in the sense that it is stable and reproducible under every evaluation variation tried and is not caused by leakage, one training participant, or the frontal/ocular channels. **Why** S08 transfers is **undetermined**: the signal is broad and not the classical mu lateralisation; contributions from visual-attention alpha, muscle activity or other non-motor processes cannot be excluded with this data. Hypotheses: *strong whole-head alpha/beta modulation with a stable spatial pattern across sessions and participants* — supported by within/between-participant consistency, not tested directly; *ocular* — contradicted for S08; *one-participant training influence* — contradicted; *data-quality artifact* — not supported. S08 was **not removed**; without S08 the protocol B CSP + LDA mean is 0.591 (aligned: 0.709), and the S08 outlier explains most of the difference between CSP + LDA (0.632) and the other models in B.

Exploratory participant-level correlations (n = 9, uncorrected, not evidence of any relationship): rejection rate, retained trial count, noise level (RMS), peak-to-peak amplitude and similarity of the participant's own CSP patterns to the pooled patterns all show |ρ| ≤ 0.57 with p ≥ 0.11 (table in the appendix).

## 6. Improving CSP + LDA [measured]

* **Regularisation sensitivity** (ledoit-wolf vs none vs OAS CSP; LDA shrinkage vs none; feature scaling): differences ≤ 0.005 balanced accuracy in both protocols (appendix). Not an issue.
* **Nested selection** over 24 pre-defined configurations (components {2,4,6,8} × band {8–30, 8–13, 13–30 Hz} × window {0.5–2.5, 1.0–3.0 s}), chosen by inner CV inside each outer training set only: **A 0.764 ± 0.175, B 0.647 ± 0.133**, versus the a-priori default 0.783 ± 0.143 / 0.632 ± 0.150 (paired differences −0.018 and +0.016; bootstrap CIs include 0; the default in the grid reproduces `final_v1` exactly). Inner CV picked 8 components in 6 of 9 B folds. The descriptive table of all 24 configurations' outer scores (appendix) shows 8 components scoring best in B (0.666) — a *biased* look at test results; the unbiased estimate of "tune by inner CV" is 0.647, an improvement not distinguishable from zero. The primary protocol was **not** changed and `final_v1` remains the headline result.
* **Euclidean alignment** (per participant and session, mean covariance whitening; unsupervised): A 0.801 ± 0.163 (+0.018, p = 0.25), **B 0.730 ± 0.114 (+0.098, CI [+0.046, +0.147], exact Wilcoxon p = 0.020 uncorrected)**; B without S08: 0.709 vs 0.591. It **requires unlabeled EEG of the target participant** (here: all their trials of a session) before prediction — a calibrated, not zero-shot, system. It was a single, pre-specified configuration (not tuned), but it is one of several exploratory variants tried in this audit, so the p-value should not be read as a confirmed finding.
* **Baseline-normalised (ERD) power** for logistic regression and CNN did **not** help (0.540 / 0.532 and 0.534 / 0.507). The implementation is correct (verified: task window equals the primary epochs, baseline window is the intended pre-cue segment); the likely cause is that the 1 s baseline gives a single noisy spectral estimate: for S03 the within-class SD of the C3 mu feature rose from 0.68 to 1.03 and the lateralisation effect shrank (d 1.92 → 0.52) **[hypothesis, supported by that one participant's numbers]**.
* Training-only feature normalisation is already in place (§3).

## 7. The CNN [measured]

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Implementation defect (labels, ordering, scaling, orientation) | permuted-label control at chance (0.475 / 0.496); class-difference maps show the expected contralateral mu pattern with correct left/right orientation; spatial mapping tested with a linear field and channel permutation; training and inference use one scaler; model trains (training BA 0.76 / 0.65, loss falls) | **no defect found** |
| Representation loses information | logistic regression on the 3 × 95 image pixels equals logistic regression on the 66 features (A 0.663 vs 0.645; B 0.574 vs 0.575) | **not confirmed** (interpolation is a linear re-expression of the same 66 numbers) |
| Interpolated vs measured-only input | CNN on the 8×8 layout grid (official-figure topology, no interpolation): A 0.587, B 0.585 vs 0.606 / 0.545 for the 16×16 interpolation | **no consistent difference** (n = 9) |
| Capacity / regularisation | smaller CNN: 0.600 / 0.541; weight decay 1e-2: 0.599 / 0.535 | **no improvement** |
| Overfitting (A) | training BA 0.756 vs validation 0.708 vs test 0.606 | **partly supported** (generalisation gap; validation optimistic through epoch selection) |
| Too little data (A) | ≈ 110 training trials per participant | **plausible, not tested** (no learning-curve experiment) |
| Weak cross-participant signal in power features (B) | training BA only 0.653 even with ≈ 1 700 trials; logistic regression on the same features 0.575 | **supported**: the CNN adds nothing beyond band power here; CSP (spatial filtering of covariance) carries information that power features lack |
| Batch size, learning rate, class balance | classes are balanced (1165 / 1181); learning rate / batch size were not varied | **untested** |

The CNN remains below CSP + LDA and statistically indistinguishable from the band-power logistic regression. This result is kept. Holm-corrected paired tests (appendix) support "CSP + LDA > logistic regression and > CNN" in protocol A (adjusted p = 0.012, n = 9) and **not** in protocol B (adjusted p = 0.387).

## 8. The spatial representation [verified, tested, measured]

* Electrode → feature mapping: tested with a hand-checkable linear field (every pixel equals its coordinate) and broken by permuting channels; matrix interpolation equals direct `griddata` to 1e-4; orientation (nose up, left ear left) tested; mapping, scaling and transform are identical at training and inference.
* 16 × 16 is a coarse choice for 22 electrodes: 95 of 256 pixels lie inside the electrode hull; electrodes C5, C6, Fz and POz sit on the hull boundary and influence few pixels. **Interpolated pixels are not independent measurements**: they are fixed linear mixtures of ≤ 3 neighbouring electrode values. The pixel-vs-feature logistic regression shows no loss of linear information, but also no gain.
* The simple alternative (measured-only 8×8 grid) gave comparable CNN scores (above). No claim is made that either representation is better.

## 9. Statistics

* Per participant: balanced accuracy, trial and class counts, confusion matrices: `outputs/final_v1/results_per_split.csv`, `docs/EXPERIMENTS.md`, appendix.
* Across participants (n = 9): paired comparisons use exact Wilcoxon signed-rank tests with Holm correction within a protocol and a participant-level percentile bootstrap for the mean difference (appendix). Trials are never treated as participants.
* The best mean among four models on nine participants is **not** a claim of general superiority: in protocol B, CSP + LDA beats the others by 0.057–0.087 but the corrected tests are not significant and the mean is driven by S08.
* Exploratory variants (alignment, ERD, layout grid, CNN variants, nested selection, channel-group diagnostics) were not corrected for multiplicity, and all results are reported, including those that did not help.

## 10. Code changes made in this audit

| Change | Purpose |
|---|---|
| `src/evaluation/leakage_audit.py`, `scripts/audit_leakage.py` | runtime leakage audit |
| `src/evaluation/experiments.py` (default behaviour unchanged): `y_fit` / `permute_training_labels`, `features`, `builder`, `collect_histories`, CNN train-BA diagnostics | negative controls and CNN diagnostics |
| `src/models/baseline.py`: ERD features, parametrised CSP pipeline, `features=` option | experiments |
| `src/features/alignment.py`, `src/evaluation/epoch_bank.py`, `src/evaluation/csp_selection.py`, `src/features/spatial.py::LayoutGridBuilder` | new, clearly separated experiments |
| `scripts/` : `compare_runs`, `verify_labels_all`, `analyze_participants`, `investigate_ocular_confound`, `investigate_channel_groups`, `run_audit_experiments`, `make_audit_tables` | reproducible analyses |
| `src/training/train.py`: `--status reproduction` | label reproduction runs |
| Tests: 19 new (leakage audit incl. injected leaks, epoch/gap boundaries, label alignment, aggregation, alignment, ERD, nested selection, permutation control, layout grid) + 1 real-data test over all 18 files | regression |
| Docs | corrected the channel-name status (now confirmed from the figure), added the audit results |

No bug in the primary pipeline was found, so no correction to `final_v1` was necessary. Post-refactor reproduction (`repro_v2`) and leakage audit on the final code (`audit_v3`): **see the status line at the end of this section.**

**Status [reproduced, measured]**: `repro_v2` (final code, unchanged config) is bit-identical to `final_v1` (72/72 rows, 3 529/3 529 predictions, score differences 0.0). The runtime leakage audit re-run on the final code (`outputs/audit_v3/leakage_audit.json`) passed in all 18 folds of both protocols with 0 duplicate epochs.

## 11. Remaining limitations and next research tasks

**Unresolved**: (1) cause of S08's strong broad-band signal; (2) residual ≈ 0.03 of the pre-cue score in protocol A after accounting for carry-over (the rest is explained, see `OCULAR_ROBUST_EVALUATION.md`); (3) ocular/gaze contribution: EOG regression and frontal exclusion do not change CSP + LDA, which is compatible with no reliance but does not prove it; (4) 17 of 22 channel names inferred from figure geometry; (5) template electrode positions; (6) nine participants; (7) public E-session labels; (8) non-causal offline pipeline.

**Next three priorities**
1. *(done in `ocular_robust_v1`, with the limits stated there)* **Pre-registered ocular-robust evaluation**: decide a priori between frontal-channel exclusion and EOG regression/ICA fitted on training data only, add the EOG-only decoding control to every report, and test it on more participants.
2. **Calibrated cross-participant methods on a larger dataset**: evaluate alignment-type adaptation (Euclidean/Riemannian) with an honest "unlabeled calibration block" protocol (how many target trials are needed?) on a dataset with many more participants (e.g. PhysioNet/MOABB motor imagery) to learn whether the S08-type transfer generalises and to replace n = 9.
3. **Fair deep-learning comparison**: only after 1–2, compare EEGNet-style raw-signal CNNs and the multi-window EEGLearn recurrent model to the CSP baselines with nested, pre-registered hyper-parameter search and learning curves over training-set size.

---

## Appendix: generated tables

*Generated by `python scripts/make_audit_tables.py --embed` from the saved experiment folders. Do not edit by hand.*

<!-- BEGIN TABLES -->

### Primary results recomputed from `final_v1/results_per_split.csv`

| Protocol | Model | Balanced accuracy (mean ± SD, n=9) | min | max |
|---|---|---|---|---|
| A (same participant, unseen session) | majority | 0.500 ± 0.000 | 0.500 | 0.500 |
| A (same participant, unseen session) | logreg | 0.645 ± 0.117 | 0.501 | 0.816 |
| A (same participant, unseen session) | csp_lda | 0.783 ± 0.143 | 0.606 | 0.948 |
| A (same participant, unseen session) | cnn | 0.606 ± 0.112 | 0.492 | 0.800 |
| B (unseen participant, LOSO) | majority | 0.500 ± 0.000 | 0.500 | 0.500 |
| B (unseen participant, LOSO) | logreg | 0.575 ± 0.054 | 0.490 | 0.669 |
| B (unseen participant, LOSO) | csp_lda | 0.632 ± 0.150 | 0.491 | 0.955 |
| B (unseen participant, LOSO) | cnn | 0.545 ± 0.042 | 0.496 | 0.616 |

### Nested selection of CSP + LDA settings (24 configurations; selection by inner CV only)

**Protocol A (same participant, unseen session)**: nested (selected) 0.764 ± 0.175; a-priori default 0.783 ± 0.143; paired difference -0.018 (bootstrap 95 % CI over participants [-0.045, +0.008], exact Wilcoxon p = 0.301, n = 9).

| Held-out participant | selected configuration | inner CV BA | outer BA (selected) | outer BA (default) |
|---|---|---|---|---|
| S01 | c6_b8-30_w0.5-2.5 | 0.911 | 0.907 | 0.857 |
| S02 | c6_b8-30_w0.5-2.5 | 0.630 | 0.570 | 0.606 |
| S03 | c2_b8-13_w0.5-2.5 | 0.987 | 0.970 | 0.948 |
| S04 | c8_b8-30_w1.0-3.0 | 0.721 | 0.642 | 0.667 |
| S05 | c4_b8-13_w0.5-2.5 | 0.620 | 0.516 | 0.615 |
| S06 | c6_b13-30_w0.5-2.5 | 0.720 | 0.664 | 0.684 |
| S07 | c6_b8-30_w1.0-3.0 | 0.833 | 0.746 | 0.795 |
| S08 | c2_b8-30_w0.5-2.5 | 0.975 | 0.956 | 0.948 |
| S09 | c4_b8-13_w0.5-2.5 | 0.973 | 0.908 | 0.923 |

Selected-configuration counts: {'c6_b8-30_w0.5-2.5': 2, 'c4_b8-13_w0.5-2.5': 2, 'c2_b8-13_w0.5-2.5': 1, 'c8_b8-30_w1.0-3.0': 1, 'c6_b13-30_w0.5-2.5': 1, 'c6_b8-30_w1.0-3.0': 1, 'c2_b8-30_w0.5-2.5': 1}

*Descriptive only (outer-test scores of ALL configurations; picking the best row would be selection on test data).* Top 5 / bottom 3 of 24 for protocol cross_session:

| configuration | mean outer BA | SD |
|---|---|---|
| c6_b8-30_w0.5-2.5 | 0.788 | 0.161 |
| c4_b8-30_w0.5-2.5 | 0.783 | 0.143 |
| c8_b8-30_w0.5-2.5 | 0.779 | 0.179 |
| c4_b13-30_w0.5-2.5 | 0.771 | 0.121 |
| c6_b13-30_w0.5-2.5 | 0.766 | 0.125 |
| c4_b8-13_w1.0-3.0 | 0.701 | 0.172 |
| c8_b8-13_w1.0-3.0 | 0.700 | 0.180 |
| c2_b8-13_w1.0-3.0 | 0.699 | 0.189 |

**Protocol B (unseen participant, LOSO)**: nested (selected) 0.647 ± 0.133; a-priori default 0.632 ± 0.150; paired difference +0.016 (bootstrap 95 % CI over participants [-0.016, +0.057], exact Wilcoxon p = 1.000, n = 9).

| Held-out participant | selected configuration | inner CV BA | outer BA (selected) | outer BA (default) |
|---|---|---|---|---|
| S01 | c8_b8-30_w0.5-2.5 | 0.688 | 0.680 | 0.529 |
| S02 | c8_b8-30_w0.5-2.5 | 0.702 | 0.562 | 0.491 |
| S03 | c4_b8-30_w1.0-3.0 | 0.676 | 0.667 | 0.714 |
| S04 | c8_b8-30_w0.5-2.5 | 0.672 | 0.585 | 0.596 |
| S05 | c8_b8-30_w0.5-2.5 | 0.687 | 0.527 | 0.512 |
| S06 | c8_b8-30_w0.5-2.5 | 0.669 | 0.528 | 0.532 |
| S07 | c8_b8-30_w0.5-2.5 | 0.685 | 0.610 | 0.616 |
| S08 | c4_b8-13_w0.5-2.5 | 0.641 | 0.955 | 0.955 |
| S09 | c6_b8-30_w0.5-2.5 | 0.714 | 0.712 | 0.740 |

Selected-configuration counts: {'c8_b8-30_w0.5-2.5': 6, 'c4_b8-30_w1.0-3.0': 1, 'c4_b8-13_w0.5-2.5': 1, 'c6_b8-30_w0.5-2.5': 1}

*Descriptive only (outer-test scores of ALL configurations; picking the best row would be selection on test data).* Top 5 / bottom 3 of 24 for protocol loso:

| configuration | mean outer BA | SD |
|---|---|---|
| c8_b8-30_w0.5-2.5 | 0.666 | 0.146 |
| c8_b8-13_w0.5-2.5 | 0.656 | 0.169 |
| c6_b8-13_w0.5-2.5 | 0.653 | 0.169 |
| c6_b8-30_w0.5-2.5 | 0.646 | 0.144 |
| c4_b8-13_w0.5-2.5 | 0.639 | 0.158 |
| c2_b8-30_w0.5-2.5 | 0.579 | 0.095 |
| c2_b13-30_w0.5-2.5 | 0.578 | 0.072 |
| c2_b8-13_w1.0-3.0 | 0.567 | 0.098 |

Consistency check: default configuration in the nested run equals final_v1 CSP+LDA for protocol cross_session: max |diff| = 0.00e+00
Consistency check: default configuration in the nested run equals final_v1 CSP+LDA for protocol loso: max |diff| = 0.00e+00

### Sensitivity of default CSP + LDA to regularisation choices (descriptive)

| Variant | Protocol A | Protocol B |
|---|---|---|
| default (ledoit_wolf CSP, LDA auto) | 0.783 ± 0.143 | 0.632 ± 0.150 |
| CSP unregularised | 0.781 ± 0.140 | 0.632 ± 0.150 |
| CSP oas | 0.782 ± 0.141 | 0.632 ± 0.150 |
| LDA no shrinkage | 0.787 ± 0.142 | 0.631 ± 0.150 |
| feature StandardScaler before LDA | 0.783 ± 0.143 | 0.632 ± 0.150 |

### Reference rows (final_v1) next to the new variants

| Variant | Protocol A | Protocol B | participants above chance threshold A / B |
|---|---|---|---|
| final_v1 majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0/9 / 0/9 |
| final_v1 band-power LogReg | 0.645 ± 0.117 | 0.575 ± 0.054 | 5/9 / 5/9 |
| final_v1 CSP + LDA | 0.783 ± 0.143 | 0.632 ± 0.150 | 9/9 / 5/9 |
| final_v1 CNN (power images, 16x16) | 0.606 ± 0.112 | 0.545 ± 0.042 | 4/9 / 3/9 |
| CSP + LDA after Euclidean alignment (uses unlabeled target data!) | 0.801 ± 0.163 | 0.730 ± 0.114 | 9/9 / 9/9 |
| LogReg on ERD features (baseline-normalised) | 0.540 ± 0.052 | 0.532 ± 0.038 | 3/9 / 2/9 |
| CNN on ERD images | 0.534 ± 0.055 | 0.507 ± 0.036 | 1/9 / 1/9 |
| CNN, measured-only 8x8 layout grid (power) | 0.587 ± 0.107 | 0.585 ± 0.081 | 3/9 / 4/9 |
| CNN small (8-16-32 filters) | 0.600 ± 0.150 | 0.541 ± 0.057 | 3/9 / 3/9 |
| CNN weight decay 1e-2 | 0.599 ± 0.123 | 0.535 ± 0.036 | 3/9 / 3/9 |

### Is information lost by the interpolated image representation? (logistic regression on the 3 x 95 in-hull pixel values)

| Model | Protocol A | Protocol B |
|---|---|---|
| LogReg on 66 band-power features (final_v1) | 0.645 ± 0.117 | 0.575 ± 0.054 |
| LogReg on image pixels (3 bands x 95 pixels) | 0.663 ± 0.116 | 0.574 ± 0.054 |

### CSP + LDA with Euclidean alignment (UNLABELED target data used - not zero-shot) vs the primary CSP + LDA

| Protocol | aligned | primary | paired diff. [bootstrap 95 % CI] | exact Wilcoxon p |
|---|---|---|---|---|
| A (same participant, unseen session) | 0.801 ± 0.163 | 0.783 ± 0.143 | +0.018 [-0.007, +0.042] | 0.250 |
| B (unseen participant, LOSO) | 0.730 ± 0.114 | 0.632 ± 0.150 | +0.098 [+0.046, +0.147] | 0.020 |

| Participant | A primary | A aligned | B primary | B aligned |
|---|---|---|---|---|
| S01 | 0.857 | 0.930 | 0.529 | 0.667 |
| S02 | 0.606 | 0.599 | 0.491 | 0.622 |
| S03 | 0.948 | 0.985 | 0.714 | 0.927 |
| S04 | 0.667 | 0.733 | 0.596 | 0.710 |
| S05 | 0.615 | 0.593 | 0.512 | 0.596 |
| S06 | 0.684 | 0.638 | 0.532 | 0.710 |
| S07 | 0.795 | 0.829 | 0.616 | 0.674 |
| S08 | 0.948 | 0.963 | 0.955 | 0.895 |
| S09 | 0.923 | 0.938 | 0.740 | 0.765 |

Protocol B mean without S08: primary 0.591, aligned 0.709.

### Negative control: decoding from the 1 s BEFORE the cue (cue-1.5 ... cue-0.5 s)

| Model | Protocol A | Protocol B | participants above chance threshold A / B |
|---|---|---|---|
| logreg | 0.488 ± 0.041 | 0.484 ± 0.026 | 0/9 / 0/9 |
| csp_lda | 0.537 ± 0.052 | 0.521 ± 0.032 | 2/9 / 1/9 |

Per participant (balanced accuracy, CSP + LDA, A / B): S01 0.555/0.500; S02 0.521/0.514; S03 0.587/0.541; S04 0.601/0.518; S05 0.454/0.504; S06 0.461/0.483; S07 0.575/0.538; S08 0.551/0.590; S09 0.531/0.500

### Negative control: training labels permuted (test labels intact)

| Model | Protocol A | Protocol B |
|---|---|---|
| majority | 0.500 ± 0.000 | 0.500 ± 0.000 |
| logreg | 0.497 ± 0.028 | 0.516 ± 0.042 |
| csp_lda | 0.472 ± 0.056 | 0.499 ± 0.069 |
| cnn | 0.475 ± 0.037 | 0.496 ± 0.025 |

* logreg: across both protocols' 18 permuted folds, mean 0.507, min 0.457, max 0.583 (expected ≈ 0.5 ± sampling noise).
* csp_lda: across both protocols' 18 permuted folds, mean 0.485, min 0.369, max 0.642 (expected ≈ 0.5 ± sampling noise).
* cnn: across both protocols' 18 permuted folds, mean 0.485, min 0.408, max 0.533 (expected ≈ 0.5 ± sampling noise).

### CNN diagnostics (same configuration as final_v1)

| Protocol | training BA at selected epoch | validation BA (selected epoch) | test BA | median selected epoch / epochs run |
|---|---|---|---|---|
| A (same participant, unseen session) | 0.756 ± 0.145 | 0.708 ± 0.119 | 0.606 ± 0.112 | 6 / 14 |
| B (unseen participant, LOSO) | 0.653 ± 0.030 | 0.587 ± 0.027 | 0.545 ± 0.042 | 6 / 14 |

Mean learning curves over folds (first 10 epochs): training loss / validation loss / validation BA

* cross_session: e1: 0.782/0.678/0.568; e2: 0.695/0.671/0.571; e3: 0.615/0.668/0.591; e4: 0.615/0.649/0.579; e5: 0.535/0.641/0.646; e6: 0.544/0.649/0.622; e7: 0.537/0.642/0.611; e8: 0.495/0.667/0.620; e9: 0.464/0.688/0.587
* loso: e1: 0.750/0.710/0.527; e2: 0.695/0.699/0.541; e3: 0.671/0.691/0.557; e4: 0.655/0.698/0.550; e5: 0.646/0.704/0.558; e6: 0.638/0.689/0.555; e7: 0.630/0.688/0.571; e8: 0.621/0.698/0.554; e9: 0.616/0.698/0.559; e10: 0.614/0.689/0.554

### Per-participant comparison (final_v1 results + descriptive data characteristics)

| S | retained (L/R) | rejected % | median epoch p-p (uV) | median raw RMS (uV) | mu lateralisation d (right vs left) | A: CSP / LogReg / CNN | B: CSP / LogReg / CNN |
|---|---|---|---|---|---|---|---|
| 01 | 279 (140/139) | 3.1 | 33.1 | 11.7 | +0.26 | 0.857 / 0.574 / 0.589 | 0.529 / 0.601 / 0.580 |
| 02 | 278 (138/140) | 3.5 | 40.1 | 13.8 | +0.26 | 0.606 / 0.535 / 0.521 | 0.491 / 0.529 / 0.499 |
| 03 | 274 (136/138) | 4.9 | 40.5 | 13.0 | +1.97 | 0.948 / 0.816 / 0.762 | 0.714 / 0.601 / 0.616 |
| 04 | 245 (121/124) | 14.9 | 31.1 | 11.5 | +0.22 | 0.667 / 0.577 / 0.492 | 0.596 / 0.578 / 0.535 |
| 05 | 264 (133/131) | 8.3 | 32.1 | 10.8 | +0.17 | 0.615 / 0.501 / 0.536 | 0.512 / 0.535 / 0.519 |
| 06 | 221 (109/112) | 23.3 | 45.1 | 13.6 | +0.17 | 0.684 / 0.602 / 0.532 | 0.532 / 0.616 / 0.537 |
| 07 | 273 (138/135) | 5.2 | 31.0 | 11.2 | -0.06 | 0.795 / 0.643 / 0.555 | 0.616 / 0.490 / 0.496 |
| 08 | 266 (132/134) | 7.6 | 51.8 | 14.7 | +0.08 | 0.948 / 0.783 / 0.670 | 0.955 / 0.669 / 0.531 |
| 09 | 246 (118/128) | 14.6 | 50.8 | 15.9 | +0.11 | 0.923 / 0.777 / 0.800 | 0.740 / 0.554 / 0.592 |

Mu lateralisation d = Cohen's d between right- and left-hand trials of log mu power at C4 minus C3 (positive = the expected contralateral desynchronisation). Flat channels: none in any participant.

### Exploratory Spearman correlations across the 9 participants (uncorrected; descriptive)

| characteristic | rho vs B CSP+LDA | rho vs A CSP+LDA |
|---|---|---|
| rejection_rate | +0.23 (p = 0.55) | -0.12 (p = 0.77) |
| mu_lateralisation_d | -0.45 (p = 0.22) | -0.22 (p = 0.58) |
| median_channel_rms_uV_raw | +0.43 (p = 0.24) | +0.43 (p = 0.24) |
| median_epoch_ptp_uV | +0.48 (p = 0.19) | +0.57 (p = 0.11) |
| n_retained | -0.23 (p = 0.55) | +0.12 (p = 0.77) |
| loso_csp_pattern_similarity_to_pooled | +0.08 (p = 0.83) | -0.08 (p = 0.83) |
| A CSP+LDA vs B CSP+LDA | +0.87 (p = 0.002) | - |

### S08 stability

* LOSO CSP+LDA confusion matrix (rows true left/right; columns predicted): [[128, 4], [8, 126]]; per session: {'E': 0.963, 'T': 0.947}
* Bootstrap over trials (trials treated as independent, so too narrow): 95 % CI [0.929, 0.977]
* Dropping each single training participant: balanced accuracy range 0.910 - 0.963 (all training participants: 0.955)
* Train on the other participants' T sessions only, test on S08's E session: 0.970
* Protocol A (own T -> own E): 0.948

### Influence of single training participants on each held-out participant (CSP+LDA, LOSO)

| Held-out | all 8 training | min after dropping one | max after dropping one |
|---|---|---|---|
| S01 | 0.529 | 0.515 | 0.763 |
| S02 | 0.491 | 0.472 | 0.585 |
| S03 | 0.714 | 0.667 | 0.909 |
| S04 | 0.596 | 0.564 | 0.641 |
| S05 | 0.512 | 0.504 | 0.569 |
| S06 | 0.532 | 0.518 | 0.694 |
| S07 | 0.616 | 0.567 | 0.679 |
| S08 | 0.955 | 0.910 | 0.963 |
| S09 | 0.740 | 0.725 | 0.785 |

### Ocular-confound diagnostic (EOG used ONLY for diagnosis, never as model input)

| S | horizontal-EOG shift d (right vs left) | EOG-only BA A-like / LOSO | CSP+LDA without frontal channels A / B | primary CSP+LDA A / B | own CSP pattern 1: strongest channels |
|---|---|---|---|---|---|
| 01 | +0.21 | 0.483 / 0.584 | 0.850 / 0.626 | 0.857 / 0.529 | FC1,CP3,FC3,Fz |
| 02 | +0.48 | 0.556 / 0.608 | 0.577 / 0.494 | 0.606 / 0.491 | FC3,Fz,FCz,FC1 |
| 03 | +0.27 | 0.563 / 0.562 | 0.948 / 0.688 | 0.948 / 0.714 | C3,CP3,C1,FC3 |
| 04 | +0.32 | 0.479 / 0.573 | 0.802 / 0.554 | 0.667 / 0.596 | C4,CP4,C6,CP2 |
| 05 | +0.90 | 0.672 / 0.675 | 0.608 / 0.523 | 0.615 / 0.512 | C4,CP4,FC4,CP2 |
| 06 | +0.25 | 0.575 / 0.581 | 0.655 / 0.583 | 0.684 / 0.532 | CP4,C4,CP2,C6 |
| 07 | +1.16 | 0.764 / 0.738 | 0.757 / 0.640 | 0.795 / 0.616 | C4,CP4,C6,C2 |
| 08 | +0.53 | 0.635 / 0.652 | 0.948 / 0.959 | 0.948 / 0.955 | CP4,P2,CP2,C6 |
| 09 | +1.06 | 0.692 / 0.757 | 0.915 / 0.743 | 0.923 / 0.740 | CP4,C4,C6,CP2 |

Means without frontal channels (Fz, FC1-4): A 0.784 vs primary 0.783; B 0.646 vs primary 0.632 (descriptive diagnostic, not a selected model).

### Which scalp region carries the information? CSP + LDA restricted to channel groups (diagnostic only)

| channel group | all participants A / B (mean BA) | S08 A / B | participants other than S08 A / B |
|---|---|---|---|
| all 22 | 0.783 / 0.632 | 0.948 / 0.955 | 0.762 / 0.591 |
| frontal (6) | 0.637 / 0.595 | 0.739 / 0.767 | 0.624 / 0.574 |
| central row (7) | 0.723 / 0.625 | 0.896 / 0.827 | 0.702 / 0.600 |
| centro-parietal row (5) | 0.700 / 0.630 | 0.941 / 0.877 | 0.670 / 0.599 |
| sensorimotor = central + centro-parietal (12) | 0.750 / 0.619 | 0.955 / 0.858 | 0.724 / 0.589 |
| posterior (4) | 0.636 / 0.581 | 0.903 / 0.741 | 0.603 / 0.561 |

### Paired participant-level comparisons on the primary results (n = 9; Holm-corrected within protocol)

| Protocol | Comparison | mean diff. BA | bootstrap 95 % CI | exact Wilcoxon p | Holm-adjusted p |
|---|---|---|---|---|---|
| A (same participant, unseen session) | csp_lda − logreg | +0.137 | [+0.103, +0.180] | 0.004 | 0.012 |
| A (same participant, unseen session) | csp_lda − cnn | +0.176 | [+0.132, +0.221] | 0.004 | 0.012 |
| A (same participant, unseen session) | logreg − cnn | +0.039 | [+0.005, +0.072] | 0.129 | 0.129 |
| B (unseen participant, LOSO) | csp_lda − logreg | +0.057 | [-0.019, +0.137] | 0.359 | 0.387 |
| B (unseen participant, LOSO) | csp_lda − cnn | +0.087 | [+0.012, +0.182] | 0.164 | 0.387 |
| B (unseen participant, LOSO) | logreg − cnn | +0.030 | [-0.000, +0.065] | 0.129 | 0.387 |

Wilcoxon exact p-values with n = 9 cannot go below 0.0039; the bootstrap interval resamples participants (not trials) and is itself coarse at n = 9.

<!-- END TABLES -->
