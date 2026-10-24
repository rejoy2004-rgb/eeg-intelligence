# Dataset: BCI Competition IV, Data set 2a (Graz)

Everything below was checked against the official pages
(<https://www.bbci.de/competition/iv/>, download page, results page, `desc_2a.pdf`) **and** against the real files
downloaded for this project. Items that could not be checked are marked **[unverified]**.

## Source and access

* Provider: Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology
  (C. Brunner, R. Leeb, G. Müller-Putz, A. Schlögl, G. Pfurtscheller).
* Training/evaluation signals: `https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip` (439 968 864 bytes;
  SHA-256 of the file used here: `65fe93cb766e4b00ece69a200312d81f54bba17c642406bd922d913a8aedc024`).
* True labels of the evaluation sessions: `https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip`
  (SHA-256 `a5e21dbd7e92d2959a3579d18de0defa9ca1a85eec17931118f81dbe141c6302`).
* The project does **not** download data automatically. To reproduce:

```bash
mkdir -p data/raw/bci_iv_2a data/raw/labels_2a
curl -L -o bci.zip   https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip
curl -L -o labels.zip https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip
unzip bci.zip -d data/raw/bci_iv_2a && unzip labels.zip -d data/raw/labels_2a
```

## Terms of use

The only terms stated on the competition site are in its download agreement: *users must give reference to the
group(s) which recorded the data*, and publications analysing a data set should cite one of the papers listed in the
data-set description (for 2a: the description lists Naeem et al. 2006 and Brunner et al. 2007), and report such
publications to the organisers. **No licence text and no statement about redistribution was found [unverified whether
redistribution is permitted]**, therefore this repository does **not** contain the data (`data/raw/` is git-ignored) and
users must download it themselves. Please cite: Brunner, C., Leeb, R., Müller-Putz, G. R., Schlögl, A., Pfurtscheller, G.
*BCI Competition 2008 – Graz data set A.*

## Files

* 18 GDF files: `A01T.gdf … A09T.gdf` (training sessions, class labels stored in the events) and `A01E.gdf … A09E.gdf`
  (evaluation sessions, cue class **hidden**, event 783). Loaded with `mne.io.read_raw_gdf` (MNE 1.13.2); no extra library needed.
* 18 label files `A0xT.mat`, `A0xE.mat`, each with variable `classlabel` (288×1, values 1–4). **[verified]** For the T files the
  `.mat` labels are identical to the event codes (checked for A01T and A04T and asserted in `tests/test_real_data.py`).
  For the E files the `.mat` file is the only source of the labels.

## Recording facts (official description + files)

| Item | Value | Verified on files |
|---|---|---|
| Participants | 9 (`A01`–`A09`) | ✔ |
| Sessions | 2 per participant, recorded on different days (T, then E) | ✔ (file dates differ, e.g. A01T 2005-01-17, A01E 2005-01-19) |
| Runs | 6 per session, 48 trials each (12 per class) = 288 trials | ✔ 288 trial-start events per file |
| Channels | 22 EEG + 3 EOG (monopolar) | ✔ 25 channels |
| Reference / ground | left mastoid / right mastoid | official description |
| Sampling | 250 Hz | ✔ |
| Amplifier filter | 0.5–100 Hz band-pass + 50 Hz notch | ✔ (header: highpass 0.5, lowpass 100) |
| EOG | Provided but **must not be used for classification** | not loaded by this project |
| Classes | 1 left hand, 2 right hand, 3 both feet, 4 tongue | ✔ |
| Recording length | ≈ 45 min per file (e.g. A01T: 672 528 samples = 2690 s) | ✔ |

## Trial timing (official Fig. 2 and text, cue offset ✔ on all inspected files)

| Time | Event |
|---|---|
| 0 s | trial start: fixation cross + short beep (event **768**, duration 7.5 s) |
| 2 s | cue arrow appears for 1.25 s (**769** left, **770** right, **771** foot, **772** tongue; **783** = unknown in E files) |
| 2 s → 6 s | participant imagines the movement until the cross disappears (t = 6 s); no feedback |
| 6 s → | short break |

Other events: **276** idle EEG eyes open, **277** eyes closed, **1072** eye movements (first ≈ 5 min of each session, used for EOG
estimation), **32766** start of a new run, **1023** trial marked as containing an artifact by a human expert.

Verified on A01T, A04T, A01E: every 768 is followed by exactly one cue event exactly 500 samples (2.0 s) later; every 1023 starts at exactly
the same sample as a 768 (so it flags that trial); A01T has 15 flagged trials, A01E 7.

## Missing data between runs

The description states that runs are separated by 100 missing values. In the GDF files these samples appear as
**−1600 µV on every channel** (the amplifier overflow value). Verified: in A01T, A01E, A04T, A09E the −1600 µV stretches are
exactly 100 samples long and begin exactly at each run-start event (8 gaps per file with 9 run starts, 6 gaps in A04T).
`src/data/dataset.py` converts them to `NaN`, records them in `Recording.missing_segments`, reports them as a warning, and the
filter never runs across them. Note: A04T has fewer EOG blocks (technical problem, per the description), so it has 7 runs-starts.

## Channel names and electrode locations — read this carefully

* The GDF files name only five EEG channels: **Fz, C3, Cz, C4, Pz** (positions 0, 7, 9, 11, 19 in file order). The others are
  labelled `EEG-0 … EEG-16`. **[verified]**
* This project assigns the remaining names by the standard 22-channel montage of this dataset:
  `Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz`.
  * the five named positions match this order in every file (checked at load time – a mismatch raises an error);
  * **Verified against the official description (research audit, rendered Fig. 3):** the figure numbers the 22 electrodes 1-22 in rows {1}, {2-6}, {7-13},
    {14-18}, {19-21}, {22} and prints the names #1 Fz, #8 C3, #10 Cz, #12 C4, #20 Pz - exactly the five named channels at 0-based file positions 0, 7, 9, 11, 19, so
    file order = figure numbering. The figure does **not print** the other 17 names; they are inferred from the electrodes' position in the 10-20 grid
    (e.g. #2-#6 = FC3 FC1 FCz FC2 FC4 above C3...C4). They are therefore *inferred from figure geometry*, not read from printed labels;
  * a data-driven plausibility check (`scripts/verify_channel_layout.py`) shows that the correlation between channels falls with template
    inter-electrode distance (Spearman −0.86 … −0.95 across the nine T recordings) far more than for random relabellings of the 17 unnamed
    channels (mean ≈ −0.05, best of 300 shuffles −0.26 … −0.42, none as good as the assumed order). This supports the order as a whole;
    it cannot exclude a swap of two neighbouring unnamed channels.
* Electrode **positions** are not part of the data. They are taken from MNE's `colin27_1020` template montage (a standard head model),
  *not* measured on these participants. Individual head geometry and cap placement differ.

## Known limitations

* 9 participants only; 2 sessions each; 72 trials per class per session (before artifact rejection).
* Motor-imagery ability and signal quality vary strongly between participants; "BCI illiteracy" is a known phenomenon.
* Labels are the *cued* class, not a verified mental event: we cannot know whether the participant actually performed the imagery.
* The E sessions were originally the held-out competition test data; their labels became public after the deadline (2008). They have been
  public for many years; no claim can be made that methods in the wider literature never saw them.
* The competition required causal (real-time) algorithms; this project is an offline analysis and uses a zero-phase filter.
