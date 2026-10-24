# Testing

## Run

```bash
pip install -r requirements.txt
python -m pytest                    # everything (real-data tests are skipped automatically if the dataset is missing)
python -m pytest -m "not realdata"  # synthetic-only: no dataset needed
python -m pytest -m realdata        # only the checks against the real BCI IV 2a files
```

## Last full run (2026-10-24, Windows 11, Python 3.11.9, CPU)

* `python -m pytest` → **183 passed, 0 failed, 0 skipped** (≈ 2.3 min) — includes 5 real-data tests that executed because the dataset was present.
* Before the research audit the suite had 139 tests (all passing). The audit added 20 and the ocular-robust evaluation 24 (`tests/test_ocular_precue.py`); none of the original tests was changed or weakened.
  (`tests/conftest.py` gained two optional, default-off fixture parameters - `calibration_s`, `eog_leak` - without changing any existing fixture output.)
* Passing tests were **not** treated as proof of correct methodology: the runtime leakage audit (`scripts/audit_leakage.py`, 18 checks per outer fold) and the negative controls described in `docs/RESEARCH_AUDIT.md` are separate evidence.

## What is tested

All synthetic fixtures are generated in `tests/conftest.py` and are **artificial noise plus a planted 10 Hz rhythm** whose only purpose is to exercise the
software (file structure imitates BCI IV 2a: 25 channels, 768/769/770… events, −1600 µV run gaps). Test results on them say nothing about real EEG.

| File | Tests | Covers |
|---|---|---|
| `test_validation.py` | 22 | missing file, wrong extension, empty file, corrupted GDF, unknown subject/session, wrong sampling rate, wrong channel count, shifted anchor channels, unknown / non-numeric event codes, partial NaN, run-gap → NaN conversion, event index / order / empty-table checks, label-file errors (missing, corrupt, no `classlabel`, unknown class) |
| `test_preprocessing.py` | 12 | output shape and channel preservation, filtering along the *time* axis (10 Hz passes, 1 Hz/80 Hz rejected), zero phase, determinism, invalid band edges / Nyquist, invalid sampling rate, short / empty / wrong-dimension input, gaps preserved and segments filtered independently, NaN/∞ rejection, re-referencing |
| `test_epoching.py` | 16 | trial table, run numbering, hidden-class cues + external labels (length mismatch, conflict, unknown class), missing cues / trial starts, epoch boundaries and gap overlap, window validation, binary selection and counting of other classes, **label ↔ trial alignment** (planted lateralised rhythm), expert-artifact and amplitude rejection with recorded reasons, identical trials for both filter bands, `TrialSet` alignment / concat / select, total-rejection edge case |
| `test_features.py` | 14 | PSD / band-power shapes, band-major order, correct band and channel for a pure sinusoid, amplitude² scaling, invalid shapes / sampling rates / bands, NaN/∞, short trials, band without a frequency bin, flat channel, channel independence, determinism, log-variance and Hjorth known values |
| `test_spatial_representation.py` | 20 | montage lookup (unknown channel = error), coordinate validation (shape, count, NaN, ∞, duplicates), projection geometry on a sphere, anatomical left/right and front/back sanity, output shapes, mask / NaN outside the hull, constant field, **linear field reproduced ⇒ channel-to-position mapping verified (and broken by permuting channels)**, agreement with direct `griddata`, orientation of electrode pixels, determinism, invalid features, scaler fitted on training data only, scaler state round-trip, scaler errors |
| `test_models.py` | 16 | transformer shape, logistic-regression with grouped inner CV, CSP+LDA, scaler fitted on training data, CNN initialisation / shapes / size, invalid architecture, a small valid training run, reproducibility with equal seed, odd batch sizes and single-sample inference, 6 invalid-input cases for training (labels, NaN, shape, single class, empty, channel mismatch), prediction input validation, CPU inference determinism, save/load round trip, missing / corrupted checkpoint |
| `test_splits.py` | 8 | LOSO partitions disjoint, complete and subject-pure; each participant held out once; deterministic validation rule; argument errors; cross-session integrity; overlap detector catches subject / trial / row overlap; manifest contents |
| `test_metrics.py` | 11 | known confusion matrix, perfect/worst, single-class and no-positive-prediction cases (NaN + note), F1 = 0 case, missing scores, invalid inputs, Wilson interval, chance threshold, summaries with NaN |
| `test_integration.py` | 5 | full workflow on synthetic recordings: load/validate → preprocess → epoch → features → images → majority/logreg/CSP-LDA/CNN → both protocols → save results → export artifact → hash-checked load → predict (trial sets and whole recordings, labelled and unlabelled) → tamper detection and input checks |
| `test_real_data.py` | 4 | *real files*: A01T structure (22 channels, 8 gaps of 100 samples, 288 trial starts), `.mat` labels equal event codes, 72 trials per class, 15 expert-marked trials, E session cue 783 + external labels, binary trial set sanity |

## Tests added by the research audit (`tests/test_audit_regressions.py`, 19; plus 1 in `test_real_data.py`)

| Topic | What is asserted |
|---|---|
| Runtime leakage audit | passes on the real code path for both protocols; **fails** when CSP or the logistic-regression scaler is additionally fitted on test-participant trials (injected leaks); reports duplicated epochs |
| CSP / scaler / image-scaler fitting | fitted statistics equal recomputation from training trials only and differ from train + test statistics (inside the audit) |
| Gaps and boundaries | no kept epoch overlaps a missing-data gap; epoch samples equal the preprocessed signal at cue + 0.5 s (label/time alignment) |
| Same trials for all models | identical test-trial sets for every model |
| Aggregation | mean, sample SD, n and pooled confusion matrix equal recomputation from per-participant rows |
| Negative control | permuted training labels leave test labels untouched and remove the signal |
| Alignment | identity mean covariance per group, row order preserved, error cases |
| ERD features | equals task minus baseline log power, offset-invariant, length check |
| Nested selection | grid has 24 unique configurations with the default first; flipping the held-out participant's labels changes neither inner scores nor the selection; ablation default equals the primary model |
| Layout grid | positions follow the official figure; value i lands on electrode i's pixel |
| Real data (all 18 files) | event codes = labels, 72 per class, cue 500 samples after start, gap structure |

## Tests added by the ocular-robust evaluation (`tests/test_ocular_precue.py`, 24)

| Topic | What is asserted |
|---|---|
| Causal filtering | a forward-only filter cannot see later samples; the zero-phase filter does (short range); impulse profile reports 0 energy before the event for the causal filter |
| Pre-cue window | lies 0.5 s after trial start and 0.5 s before the cue; never crosses gaps; the control is blind to a post-cue-only effect yet detects a window that overlaps it |
| EOG regression | recovers planted coefficients; coefficients depend **only** on the calibration block; gaps preserved; shape/length errors; calibration block must exist, be long enough and end before the first trial |
| Frozen design | frontal exclusion in code equals the frozen YAML; CNN not part of it; train-only strategy flagged zero-shot, target-calibrated flagged not |
| Paired comparisons | require identical test trials; Holm/bootstrap/permutation helpers behave as specified |
| Held-out labels | flipping all labels of the held-out participant leaves that fold's predictions and scores bit-identical (both protocols, all classical models) |
| Adaptation data | fold-specific data may not change labels/metadata; per-fold dataset hook called once per held-out participant |
| Baseline protection | `final_v1`, `repro_v1`, `repro_v2` can never be written (even with `overwrite=True`); existing run folders need explicit overwrite; provenance recorded |

## Bugs that the tests found (and what was done)

* A synthetic fixture generated a different spatial pattern for every recording, so a CSP filter trained on one recording could not transfer; the fixture
  (not the model) was wrong and was fixed – the assertion (≥ 0.75 accuracy on the planted effect) was **not** loosened.
* `TrialSet.select` failed for integer index arrays (found by the misalignment test) – fixed to support both boolean masks and index arrays.
* Two incorrect test expectations (an off-by-design prediction-row count and an error-message pattern) were corrected to match the intended behaviour.

## Not covered by automated tests

* The Streamlit UI beyond a headless smoke run (`streamlit.testing.v1.AppTest`: load, switch recording, run prediction; no exceptions — executed manually, not part of `pytest`).
* The notebooks (their code cells were executed once via a script; not part of `pytest`).
* The Dockerfile (never built), GPU execution, other operating systems / Python versions.
* `scripts/*` (figures, reports, audits, diagnostics, the ocular-robust runner, pre-cue investigation) were run once; they have no automated tests (the audit logic itself is tested through `src/evaluation/leakage_audit.py`).
* Statistical properties of the models on real data (that is the job of `docs/EXPERIMENTS.md`, not of unit tests).
