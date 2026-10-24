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
