# Methodology

All values below are the defaults in `configs/default.yaml`. They were fixed *before* the final evaluation (see `docs/EXPERIMENTS.md`
for the history) and are not tuned on test data.

## 1. Scientific task

Binary classification of **cued left-hand versus right-hand motor imagery** from 22-channel EEG (BCI Competition IV 2a).
Label mapping: 0 = left hand (class 1 in the dataset), 1 = right hand (class 2). Classes 3 (feet) and 4 (tongue) are excluded from the task and
counted in the trial summary (`n_other_class`).

Neuroscience background (motivation, not a result of this project): imagining a hand movement tends to reduce (desynchronise) mu (≈8–12 Hz) and
beta (≈13–30 Hz) power over the *contralateral* sensorimotor cortex (around C3 for the right hand, C4 for the left). The size, frequency and
location of this effect differs between people; band edges are conventions, not biological constants.

## 2. Signal handling

1. **Channels**: 22 EEG channels (EOG not used as a feature; used only in a diagnostic script). Channel names are positional: numbering and the five named anchors are confirmed by the official Figure 3, the other 17 names are inferred from figure geometry (`docs/DATASET.md`).
2. **Missing samples**: the 100-sample overflow gaps between runs become NaN; they are never interpolated or filled.
3. **Band-pass filter**: 4th-order Butterworth, applied forward and backward (`sosfiltfilt`) ⇒ zero phase, effective order 8, applied along the time
   axis of each *gap-free segment* separately. Default band 4–40 Hz (band power features, CNN), 8–30 Hz (CSP). Validity checks: 0 < low < high < Nyquist,
   segment long enough for the filter padding, no NaN inside a segment, no ∞.
   *Assumptions/limits*: zero-phase filtering uses future samples (offline only); the recordings already contain an amplifier 0.5–100 Hz band-pass and a
   50 Hz notch, so no extra notch is applied (the 40 Hz upper edge removes line noise anyway).
4. **Reference**: left-mastoid reference as recorded (`rereference: none`). `average` is available but not used.
5. **Epoching** (anchored to **cue onset**, event 769/770 or 783 with external label): window **+0.5 … +2.5 s** after the cue = 500 samples.
   *Why*: skips the first 0.5 s (visual response to the cue), stays inside the 4 s imagery period (cue + 4 s = 6 s), and 2 s gives 1 Hz spectral resolution
   with 1 s Welch segments. This is a conventional but not unique choice; other windows may work better for some participants.
   The pre-cue fixation period, the cue itself and the post-imagery break are *not* in the epoch.
6. **Rejection** (always recorded with trial id and reason; nothing is dropped silently):
   * expert artifact marker 1023 (provided by the data owners for both sessions);
   * epoch outside the recording or overlapping missing data;
   * peak-to-peak amplitude > 200 µV on any channel (screening rule; fixed a priori; not a guarantee that remaining trials are artifact-free).
   In the reported run only expert-marked trials were rejected (246 of 2592 left/right trials).
   The ICA / automatic artifact removal was **not** implemented: with 22 channels, a short recording and no EOG-regression validation, claiming
   “clean” data would be unjustified; the EOG channels are deliberately unused as the dataset providers require.

## 3. Features

**Band power** (`src/features/spectral.py`): Welch PSD per channel (250-sample segments, 50 % overlap, constant detrend) ⇒ 1 Hz resolution; for each band the
**mean PSD** (µV²/Hz) over the bins within the band edges; natural log. Bands: mu 8–12, beta_low 13–20, beta_high 21–30 Hz. Output (n, 3, 22) band-major, mirroring
EEGLearn's (theta, alpha, beta) × electrodes layout. Features use only a trial's own samples (no training-set statistics).

## 4. Spatial EEG images (inspired by EEGLearn's `gen_images`)

`src/features/spatial.py`; details of what is identical to / different from the original are in the module docstring and `ORIGINAL_EEGLEARN_ANALYSIS.md`.

1. **Electrode positions**: MNE `colin27_1020` template, looked up *by channel name* (an unknown name is an error, never a guess).
2. **3-D → 2-D**: least-squares sphere centre → azimuthal equidistant projection (angle = azimuth, radius = angular distance from the vertex). Cz lands near
   the origin, nose up, left ear left.
3. **Grid**: 16 × 16 pixels, isotropic (same scale in x and y), spanning the larger of the x/y extents of the projected electrodes. 95 of 256 pixels lie inside the
   convex hull of the 22 electrodes.
4. **Interpolation**: `scipy.interpolate.griddata(method="cubic")` (Clough–Tocher). Because it is linear in the electrode values it is evaluated once per electrode to
   build a weight matrix; `images = features @ weights.T` (tests confirm agreement with direct `griddata`). Pixels outside the hull are NaN (mask kept), then 0 after scaling.
5. **Scaling**: per feature channel (band) mean/std over in-hull pixels of the **training** images only (`ImageScaler.fit`), applied unchanged to validation/test.
6. **Meaning of pixels**: every in-hull pixel is *interpolated*; only the electrode markers are positions of real sensors. Image channels are frequency bands, not colours.

## 5. Models

| Name | Input | Steps | Learned from training data only |
|---|---|---|---|
| majority | – | predicts most frequent training class | class prior |
| logreg | epochs (4–40 Hz) | log band power (66 features) → StandardScaler → L2 logistic regression | scaler, weights, C (inner grouped CV, grid 0.01…10) |
| csp_lda | epochs (8–30 Hz) | CSP (4 components, Ledoit–Wolf covariance) → log-variance → shrinkage LDA | spatial filters, LDA |
| cnn | images (3, 16, 16) | see below | ImageScaler, CNN weights, epoch (early stopping on validation) |

**CNN** (`src/models/eeg_cnn.py`): Conv3×3(16)-BN-ELU → Conv3×3(32)-BN-ELU-MaxPool → Conv3×3(64)-BN-ELU-MaxPool → Flatten → Dropout(0.5) → Linear(1024→64)-ELU →
Dropout(0.5) → Linear(64→2); 89 538 parameters. Adam (lr 1e-3, weight decay 1e-4), batch 32, ≤ 40 epochs, early stopping (patience 8) on validation balanced accuracy
(ties → lower validation loss), best weights restored. Cross-entropy loss; logits are not probabilities. CPU only. This is **not** the EEGLearn architecture (no multi-window
recurrent branch, smaller, batch-normalised).

**Scores**: classical models output `decision_function` values; the CNN outputs logit_right − logit_left. Positive ⇒ “right”. None is calibrated; ROC-AUC uses them only for ranking.

## 6. Evaluation protocols (`src/evaluation/splits.py`)

| Protocol | Question | Train | Selection | Test |
|---|---|---|---|---|
| **A `cross_session`** | same person, another day? | participant's session T | logreg C: 5-fold grouped by *run* inside T; CNN: last run of T | same participant's session E |
| **B `loso`** | unseen person? | classical models: the 8 other participants (T+E); CNN: 6 of them | logreg C: 5-fold grouped by *subject* inside the 8; CNN: the other 2 whole participants (next two IDs, fixed rule) | held-out participant, sessions T+E |

The dataset's own split (T for training, E for testing) is the basis of protocol A. A plain random split over epochs is deliberately **not** offered as an evaluation:
trials from one recording are correlated and such a split answers neither question.

### Leakage safeguards (where enforced → where tested)

| Risk | Safeguard | Test |
|---|---|---|
| Scalers/CSP fitted on test data | learned steps live inside sklearn pipelines fitted on the training indices; `ImageScaler.fit` receives only the CNN-fit partition | `test_scaler_inside_pipeline_is_fitted_on_training_data_only`, `test_scaler_uses_training_statistics_only` |
| Participant in train and test (protocol B) | `loso_splits` + `assert_no_overlap(..., by="subject")` called for every fold at run time | `test_loso_partitions_are_disjoint_complete_and_subject_pure`, `test_overlap_detector_catches_violations`, `test_protocol_runs_and_records_everything[loso]` |
| Same trial in train and test | `assert_no_overlap(..., by="trial_id")` at run time | `test_overlap_detector_catches_violations` |
| Model selection on test data | C chosen by inner CV inside the training set; CNN epoch chosen on a validation partition that is never the test partition; CNN/CSP/band hyper-parameters fixed in the config | `test_cross_session_split_integrity` |
| Validation not independent of training | LOSO validation = whole other participants; protocol A validation = a different run | `test_loso_partitions_...`, `test_cross_session_split_integrity` |
| Overlapping windows across partitions | exactly one 2 s epoch per trial; trials are ≥ 8 s apart, so epochs never overlap; no sliding windows | by construction (`locate_trials`, 8 s trial spacing) |
| Preprocessing learned from test data | filters/thresholds are fixed constants, each file filtered independently | `test_preprocess_recording_...`, `test_gaps_are_preserved_...` |
| Different trials per model | `build_dual_band_trials` forces identical trial sets | `test_dual_band_sets_contain_identical_trials` |
| Silent trial loss | every rejection recorded with reason; counts checked | `test_expert_artifact_rejection_is_recorded`, `test_epoch_overlapping_gap_is_rejected_...` |

Each split's participant IDs, sessions, trial counts and class counts are written to `outputs/<run>/splits.json`.

### Verification beyond unit tests (research audit)

The table above was re-checked at **run time** with `src/evaluation/leakage_audit.py`: it records which trials every `fit` actually receives and compares fitted statistics (scaler means, CSP
filters, image-scaler statistics) with recomputations from training trials only (18 checks in each of the 18 outer folds, all passed; `outputs/audit_v1/leakage_audit.json`, repeated on the final code in
`outputs/audit_v3/`). Negative controls: training labels permuted (all models at chance) and pre-cue-window decoding (see `docs/RESEARCH_AUDIT.md`). Additional, *exploratory* analyses that are not
part of the primary protocol: nested CSP selection, Euclidean alignment (uses unlabeled target data - not zero-shot), baseline-normalised features, a measured-only layout grid, channel-group and ocular diagnostics.

Further controls added in `ocular_robust_v1` (see `docs/OCULAR_ROBUST_EVALUATION.md`): a strictly causal filter option (`bandpass_filter(..., causal=True)`), EOG-only diagnostics, EOG regression fitted on the
dataset's eye-movement calibration block only, the pre-cue negative control with within-run and previous-class-conditioned permutation nulls, and a check of the cue-sequence structure (only 4 distinct sequences exist in the
18 recordings; consecutive cues are negatively dependent). Saved experiment folders can no longer be overwritten and the baseline runs are protected.

## 7. Metrics

Positive class = right hand. Accuracy, balanced accuracy, precision, recall (sensitivity), specificity, F1, confusion matrix, ROC-AUC (only when scores exist and both classes
are present). Undefined values are `NaN` with a note. Per participant: Wilson 95 % interval for accuracy and the accuracy needed to beat 50 % at p < 0.05 for that test-set
size. Across participants: mean, sample standard deviation, min, max; paired exact Wilcoxon tests for model differences (n = 9, uncorrected, indicative only).

## 8. Reproducibility

Seeds: config `evaluation.seed` (+ participant ID for the CNN); `numpy.random.default_rng` for batch order. Full config, software versions, split manifests and rejected trials are stored in
`outputs/<run>/`. Classical models are deterministic; CNN results may differ slightly between PyTorch versions/CPUs/thread counts.
