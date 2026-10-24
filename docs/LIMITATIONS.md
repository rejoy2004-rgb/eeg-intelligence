# Limitations

Read together with `docs/EXPERIMENTS.md` (numbers) and `docs/MODEL_CARD.md` (intended use).

## Data

* **Nine participants.** Any statement about "people in general" rests on nine individuals recorded in one lab on two days each. Between-participant
  differences are large (protocol A balanced accuracy of CSP+LDA ranges from 0.61 to 0.95), so means have wide uncertainty.
* **Labels are cues, not verified mental events.** We know which arrow was shown, not whether the participant performed the imagery.
* **Channel names and positions**: only 5 of 22 channel names are in the files. The official Figure 3 confirms the numbering/row structure and the five named anchors, and
  the other 17 names are inferred from their position in the 10-20 grid (see `DATASET.md`, `RESEARCH_AUDIT.md` §4.3). Electrode coordinates come from a template head, not from the participants.
* **Ocular / gaze leakage is possible** (research audit): the cue is an arrow visible until 3.25 s and EOG-only classifiers (diagnostic, never used as model input) reach balanced
  accuracy 0.67-0.76 for S05, S07, S09 (protocol B: 9/9 participants above chance, mean 0.637). The primary pipeline does not remove ocular activity and its CSP uses frontal channels. This threatens *validity of the interpretation as motor imagery*, not the split integrity.
  Frontal-channel exclusion and EOG regression (train-only and target-calibrated) did **not** change CSP + LDA results (all Holm p ≥ 0.66) - compatible with no ocular reliance, but not proof: the EOG electrodes also record brain activity, the regression quality is very uneven (R² 0.01-0.65), volume-conducted ocular activity remains in the other channels, and n = 9.
* **Cue-order structure in the dataset**: only 4 distinct 288-trial cue sequences exist across the 18 recordings (13 share one identical order) and consecutive left/right cues are negatively dependent (P(next differs) = 0.571).
  The pre-cue window (cue−1.5 … cue−0.5 s) still gives CSP + LDA 0.542 / 0.521 (causal filter; A / B), which is largely - not entirely - explained by carry-over of the previous trial's class (a residual ≈ 0.03 remains in A, unresolved).
  History-related information may therefore add a few hundredths to task-window scores (`OCULAR_ROBUST_EVALUATION.md` §3.3-3.4); it cannot be separated from true decoding with the present data.
* **The origin of S08's strong signal is undetermined** (broad across regions; not the classical C3/C4 mu effect; not ocular, not one training participant).
* **Alignment-based gains need unlabeled target data**: the improved protocol B score of CSP + LDA after Euclidean alignment (0.730) is a calibrated, not zero-shot, result and exploratory.
* **Artifact handling is minimal.** We drop expert-marked trials and trials above 200 µV peak-to-peak; we do not remove ocular or muscle artifacts
  (no ICA/regression), and do not claim the remaining trials are artifact-free. Rejecting artifact trials also means results describe *artifact-free* trials,
  as in the official competition evaluation, not continuous/asynchronous use.
* **Rejection is uneven**: S06 loses 67 of 288 left/right trials (23 %), S04 43, S09 42; S01/S02 lose 9/10 (see `outputs/final_v1/rejected_trials.csv`).
* **The E-session labels are public** and have been used by many studies since 2008; there is no truly untouched test set in this dataset.
* **Offline analysis.** Zero-phase filtering and whole-trial windows are non-causal; the competition required causal algorithms. The pipeline is not a real-time BCI.

## Generalisation

* Protocol A (same person, other day) and protocol B (unseen person) are *different questions*. Good A results say nothing about B.
* Per-participant confidence intervals and chance thresholds treat the trials of a session as independent; they are optimistic because trials are serially correlated. Inference across
  participants uses paired participant-level tests (n = 9) with Holm correction.
* Protocol B results are weak for all models: mean balanced accuracy 0.55–0.63, and the mean of CSP+LDA is carried by one participant (S08, 0.955; without them
  0.59). Only about half the participants exceed the per-participant chance threshold for the best models. Cross-participant transfer for motor imagery is known to be hard.
* A single dataset, a single task (left vs right hand), a single recording setup. Nothing here supports generalisation to other tasks, hardware or populations.

## Scientific conclusions that are *not* supported

* "Deep learning is better than classical methods" – in this experiment the CNN was **not** better; a different design or more data might change that, which was not tested.
* "The model detects/reads thought, intention or mental state" – it separates two cued imagery conditions in a laboratory setting.
* Any clinical statement (diagnosis, screening, assessment of ability). None was evaluated and none should be inferred.
* Any claim of reproducing EEGLearn: different data, task, features, architecture and evaluation (`ORIGINAL_EEGLEARN_ANALYSIS.md`).
* Spatial maps are descriptive and interpolated; they are not source localisation and carry no significance test.
* Comparison with published state-of-the-art: not attempted (different task framing, splits and metrics).

## Model behaviour

* Scores are decision values and are **not calibrated probabilities**; thresholding at 0 assumes balanced classes.
* The binary model assigns "left" or "right" to *every* trial it is given. For unlabeled recordings containing feet/tongue cues it cannot say "neither".
* Logistic regression and the CNN use power only (no phase, no spatial filtering); CSP+LDA, which does, performs best.
* The CNN was not tuned (a-priori settings) and early stopping uses a small validation set (about one run or two participants), so its selected epoch is noisy.
* Model files are pickles/PyTorch weights: load only artifacts you created (`weights_only=True` is used for PyTorch, joblib has no such protection; SHA-256 hashes detect accidental or
  post-export modification, not a malicious author).

## Engineering

* Docker image not built or run (Docker unavailable on the development machine).
* Notebooks were verified by executing their code cells in a script, not through Jupyter; outputs are not stored in them.
* The dashboard was smoke-tested with Streamlit's headless `AppTest` (no exceptions when loading, switching recordings and predicting), not manually in a browser, and not on phones/other browsers.
* Exact CNN reproducibility across machines is not guaranteed.
* Tested on Windows 11, Python 3.11.9, CPU only.
