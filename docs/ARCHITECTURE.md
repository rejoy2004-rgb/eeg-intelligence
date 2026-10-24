# Architecture

## Layout

```
eeg-intelligence/
├── app.py                    Streamlit dashboard (UI only; no science in here)
├── configs/default.yaml      ALL scientific settings (one file, saved with every experiment)
├── src/
│   ├── config.py             load/validate YAML, software-version record
│   ├── data/
│   │   ├── validation.py     DataValidationError, ValidationReport, array/event/label/channel checks
│   │   └── dataset.py        GDF → Recording (22 EEG channels, µV, NaN gaps, event table); official label files
│   ├── signal/
│   │   ├── preprocessing.py  gap-aware zero-phase Butterworth band-pass, optional re-reference
│   │   ├── epoching.py       trial table from event codes, epoch cutting, rejections, TrialSet, dual-band sets
│   │   └── quality.py        peak-to-peak screening, per-channel quality table
│   ├── features/
│   │   ├── spectral.py       Welch PSD → band power (n, bands, channels)
│   │   ├── statistical.py    log-variance, Hjorth (implemented + tested; NOT used in the reported experiments)
│   │   └── spatial.py        EEGLearn-inspired electrode → image mapping + training-only ImageScaler
│   ├── models/
│   │   ├── baseline.py       band-power logistic regression, CSP + LDA (scikit-learn / MNE pipelines)
│   │   └── eeg_cnn.py        compact PyTorch CNN, training loop, save/load
│   ├── evaluation/
│   │   ├── splits.py         cross-session and leave-one-subject-out splits + overlap assertions + manifests
│   │   ├── metrics.py        metrics with explicit undefined cases, Wilson CI, chance threshold
│   │   └── experiments.py    run protocols, aggregate, save auditable results
│   ├── training/
│   │   ├── train.py          CLI: `evaluate`, `export-demo`
│   │   └── predict.py        artifact loading (hash-checked) and prediction (CLI + dashboard share it)
│   └── visualization/        Matplotlib figures: signals, spectra, topomaps
├── scripts/                  make_figures, make_experiment_report, verify_channel_layout
├── tests/                    pytest suite (synthetic fixtures + a few real-data checks)
├── notebooks/                4 walkthrough notebooks (call the same library)
├── docs/                     this documentation
├── data/                     README only; raw data are downloaded by the user (git-ignored)
├── models/                   exported artifacts (git-ignored)
└── outputs/                  experiment folders (final_v1 is kept as evidence)
```

The original repository is cloned separately in `../reference/EEGLearn/` and is never imported.

## Data flow

```
 A0xT.gdf / A0xE.gdf ─▶ load_gdf_file ─▶ Recording
 (+ A0xE.mat labels)        │              data (22, N) µV, NaN gaps, events DataFrame, warnings
                            ▼
                preprocess_recording        band-pass per gap-free segment (4–40 Hz; CSP: 8–30 Hz)
                            ▼
              locate_trials ▶ extract_epochs ▶ rejections (expert 1023, amplitude, gap overlap)
                            ▼
                         TrialSet            X (n, 22, 500), y (0 left / 1 right), meta, rejections
                  ┌─────────┼───────────────────────┐
                  ▼         ▼                       ▼
        BandPowerFeatures  CSP (8–30 Hz set)   band_power_from_config  (n, 3 bands, 22)
        → StandardScaler      → LDA                    ▼
        → LogisticRegression                    SpatialImageBuilder  (n, 3, 16, 16, NaN outside hull)
                                                       ▼
                                                  ImageScaler (fit on training images only)
                                                       ▼
                                                  EEGSpatialCNN → logits (n, 2)
```

Two epoch sets (`bp`: 4–40 Hz, `csp`: 8–30 Hz) are always built together by `build_dual_band_trials` and contain **identical trials**
so that all models are compared on the same data.

## Key array shapes

| Object | Shape | Meaning |
|---|---|---|
| `Recording.data` | (22, N) | µV; NaN in documented gaps |
| `TrialSet.X` | (n, 22, 500) | 2 s at 250 Hz, cue + 0.5 … 2.5 s |
| band power | (n, 3, 22) | mu, beta_low, beta_high; band-major |
| images | (n, 3, 16, 16) | row 0 front, column 0 left; NaN outside hull before scaling, 0 after |
| CNN output | (n, 2) | logits; score = logit_right − logit_left |

## Design decisions (and why)

* **Pipelines own the learned steps.** Feature extraction, scaling, CSP and the classifier live in one scikit-learn `Pipeline`, so
  fitting on training data automatically fits *everything* on training data, and prediction replays the same steps.
* **One configuration object, stored with every result** (`summary.json` contains the full config, versions and a split manifest).
* **Fail loudly.** Loaders and feature code raise typed errors (`DataValidationError`, `PreprocessingError`, `FeatureError`, `SpatialError`,
  `ModelInputError`, `MetricsError`, `SplitError`) instead of guessing; rejected trials are listed with a reason.
* **UI separated from science.** `app.py` only calls library functions; the CLI prediction and the dashboard prediction call the same
  `predict_recording`.
* **No heavy infrastructure**: no database, no web API, no experiment-tracking service. Results are CSV/JSON files.
* **Memory**: one GDF file (~120 MB as float64) is held at a time during loading; epochs (2346 × 22 × 500 float64 ≈ 200 MB for all data) are kept.
