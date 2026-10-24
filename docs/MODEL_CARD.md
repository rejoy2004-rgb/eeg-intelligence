# Model card — EEG Intelligence motor-imagery classifiers

## Summary

Three small binary classifiers (left-hand vs right-hand *cued motor imagery*) trained on EEG from BCI Competition IV 2a:
(1) band-power logistic regression, (2) CSP + LDA, (3) a 3-channel-image CNN inspired by EEGLearn. A majority-class model is the floor.
Artifacts are produced by `python -m src.training.train export-demo --heldout 1` into `models/demo_heldout_S01/` (git-ignored; participant 1 is excluded from training).

## Intended use

* Education, interview/portfolio demonstration, methodological research on EEG pipelines and evaluation (leakage-aware splits, spatial EEG images).
* Exploring how a motor-imagery decoder behaves on the public dataset it was built for.

## Out-of-scope / do not use for

* Any clinical, diagnostic, screening or rehabilitation decision.
* Inferring emotions, thoughts, intentions, mental health or cognitive ability of a person.
* Controlling devices (the pipeline is offline, non-causal and was never validated for real-time use).
* Other EEG hardware, channel sets, sampling rates or tasks (the loader rejects files that do not match the 22-channel / 250 Hz layout).

## Training data

BCI Competition IV 2a, 9 healthy participants (the description does not state further demographics), 22 EEG channels, 250 Hz, two sessions on different days
(`docs/DATASET.md`). Left/right trials only; 2346 trials after rejecting 246 (expert-marked artifacts). The demo artifact is trained on participants 2–9 (both sessions);
logistic-regression C by subject-grouped CV; the CNN is validated on participants 8–9 (fixed rule: the last two training participants).

## Evaluation (see `docs/EXPERIMENTS.md` for full tables)

Run `final_v1` (balanced accuracy, mean ± SD over 9 participants):

| Protocol | Majority | Band-power LogReg | CSP + LDA | CNN |
|---|---|---|---|---|
| A: same participant, unseen session | 0.500 | 0.645 ± 0.117 | **0.783 ± 0.143** | 0.606 ± 0.112 |
| B: unseen participant (LOSO) | 0.500 | 0.575 ± 0.054 | **0.632 ± 0.150** | 0.545 ± 0.042 |

Reading: CSP + LDA was the best model in this study under both protocols; the CNN was not better than the classical baselines. In protocol B the
average of CSP + LDA is driven by one participant (S08), and the best models exceed per-participant chance level for only 5 of 9 participants.
**The demo artifact was applied once as an illustration** to participant 1's evaluation session (never seen in training): CSP+LDA accuracy 0.546 and
CNN accuracy 0.539 on 141 trials, i.e. *below* the 0.574 needed to distinguish a result from chance for that many trials. Participant 1 is a case where the
unseen-participant models do not work (protocol B, CSP+LDA: 0.529); the demo is for showing the workflow, not performance.

## Additional validity caveats from the research audit

* EOG-only classifiers reach 0.67-0.76 balanced accuracy for some participants (S05, S07, S09), so ocular/gaze information may contribute to the left-vs-right decision for them; the pipeline does not remove it.
* A pre-cue control gave CSP + LDA a small above-chance score (0.537 / 0.521); the reported task-window scores are far higher, but a small non-specific component cannot be excluded.
* Euclidean alignment improves cross-participant CSP + LDA to 0.730 but needs unlabeled data from the target participant; the exported demo artifact does **not** use it.

## Inputs and outputs

* Input: a BCI IV 2a-style GDF recording (22 EEG + 3 EOG channels, 250 Hz). Trials are cut 0.5–2.5 s after the cue; trials with artifact markers are skipped.
* Output per trial: predicted class (left hand / right hand) and a score (decision value). The score is **not a probability** and is not calibrated.

## Risks and caveats

* Over-interpretation: a labelled output like “left hand” can be mistaken for a reading of a person's mind. It only reflects the statistical similarity of the band-power /
  spatial pattern to training trials of a cued task.
* Fairness/representation: nine participants of unknown demographics; no analysis across sex, age, hair/cap type or BCI aptitude. Performance varies hugely between participants
  (0.49–0.96 balanced accuracy), so any real use would need per-person validation.
* Privacy: EEG recordings are personal data. The dashboard processes uploads locally and in a temporary folder; no data are sent anywhere or logged. Uploaded GDF files are validated
  and size-limited (100 MB) but should come from trusted sources.
* Security: model files (`joblib`, PyTorch weights) must only be loaded from trusted origins. `load_cnn` uses `torch.load(weights_only=True)`; joblib files are pickles (hash check only detects modification after export).

## Maintenance

Research code without a maintenance plan. Configuration, software versions and split manifests are stored with each run for reproducibility.
