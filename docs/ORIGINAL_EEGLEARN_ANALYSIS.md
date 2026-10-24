# Analysis of the original EEGLearn repository

This document records what was found by **reading the actual code and sample files** of
<https://github.com/pbashivan/EEGLearn> (commit `b217414919d2a38c9e32d78fbc4516230a6806cd`, dated 2020-07-02,
cloned into `reference/EEGLearn/`, never modified). Statements are tagged:

* **[verified]** – read directly in the code / checked on the sample files in this session.
* **[uncertain]** – not determinable from the repository; stated as a question, not as a fact.

The paper itself was *not* re-read in full during this session; claims about it are limited to what the
repository README states.

---

## 1. What the repository contains

| File | Role |
|---|---|
| `README.md` | Overview, install notes, two usage tips, pointers to TensorFlow/PyTorch ports, citation request |
| `requirements.txt` | `numpy>=1.9, scipy>=0.16, scikit-learn>=0.16, theano>=0.8, Lasagne>=0.1` |
| `setup.py` | `distutils` install script; pins `numpy==1.13.1, scipy==0.19.1, scikit-learn==0.18.2, theano==0.8` and a Lasagne git revision |
| `eeglearn/eeg_cnn_lib.py` (514 lines) | Projection, image generation (`gen_images`), five network builders, mini-batching, `train`, and a `__main__` demo script |
| `eeglearn/utils.py` (129 lines) | `cart2sph`, `pol2cart`, noise augmentation, `load_data`, `reformatInput` |
| `Sample data/` | Three `.mat` files + README describing a visual working-memory experiment |
| `LICENSE` | **GNU GPL v2** |

## 2. Answers to the Phase-1 questions

**1. What EEG data does it expect?** *Not raw EEG.* It expects a **pre-computed feature matrix**
`features` of shape `(n_trials, n_features)` plus, separately, electrode coordinates. **[verified]**
`FeatureMat_timeWin.mat` → variable `features`, shape `(2670, 1345)`, `float64`.

**2. How are recordings/features represented?** Each trial is one row. The feature vector is built from
FFT power in three bands (theta, alpha, beta) for 64 electrodes in a **band-major** order
`theta_1…theta_64, alpha_1…alpha_64, beta_1…beta_64` = 192 values per time window. **[verified: sample README + `gen_images` slicing `features[:, c*nElectrodes:(c+1)*nElectrodes]`]**
The exact FFT settings (window length, scaling, any log transform) are **[uncertain]**: the sample features
contain negative values (min −1.05, max 51.65 **[verified]**), so they are not plain non-negative power.

**3. How are temporal windows processed?** The sample has **7 windows**; their feature blocks are stored
side by side: columns `0:191` = window 1, `192:383` = window 2, … **[verified: sample README; 7×192 = 1344, plus one label column = 1345 ✔]**.
`__main__` then either (a) averages the 7 windows into one image per trial (single-frame "cnn" model) or (b)
builds one image per window and feeds the 7-image sequence to the recurrent/convolutional models.

**4. How is channel-location information used?** 3-D coordinates (`Neuroscan_locs_orig.mat`, variable `A`, shape
`(64, 3)` **[verified]**) are projected to 2-D with `azim_proj` (azimuthal equidistant projection via `cart2sph` then `pol2cart`). The
coordinate unit and frame are **[uncertain]**; the vector norms of the 64 positions range from 92.7 to 161.3 **[verified]**,
i.e. the coordinate origin is *not* the centre of a sphere through the electrodes, so projecting "around the origin" is only approximate.

**5. How are spatial EEG images built?** `gen_images(locs, features, n_gridpoints, normalize, augment, pca, std_mult, n_components, edgeless)`:
split the feature columns into `n_colors = n_features / n_electrodes` groups (one per band); build a regular
`n_gridpoints × n_gridpoints` grid spanning min..max of the projected x and y **independently**; for every trial
and every band call `scipy.interpolate.griddata(locs, values, grid, method='cubic', fill_value=nan)`;
standardise each band with `sklearn.preprocessing.scale` over **all non-NaN pixels of all samples**; replace remaining NaN
(outside the convex hull of the electrodes) by 0. Optional: Gaussian/PCA noise augmentation *before* interpolation, and an
"edgeless" mode that adds four zero-valued virtual electrodes at the bounding-box corners.

**6. What do the image dimensions represent?** Output shape `[samples, colors, W, H]`
(`np.swapaxes(np.asarray(temp_interp), 0, 1)`). `W×H` is a 2-D map of the scalp (default 32×32 in the demo); a pixel is an
interpolated feature value at that scalp position.

**7. How are feature channels represented?** As the second axis, called "colors" in the code. It is an analogy: the
channels are frequency bands, not colours.

**8. What does `gen_images` do?** See 5. **[verified]** Notes on its behaviour: it loops over trials in pure Python
and calls `griddata` (which re-triangulates) for every trial × band; `normalize=True` uses all samples at once
(this happens *before* any train/test split in the demo → test data influence the scaling statistics; the effect is
unsupervised, uses no labels, and is probably small, but it is not leakage-free by construction).

**9. How is the classification pipeline structured?** `__main__` of `eeg_cnn_lib.py` (only runs as a script):
load locations/features/subject numbers → project locations → leave-one-subject-out folds (shuffled index
lists) → generate images → train a model per fold → average test accuracy. **[verified]**

**10. What does `train` expect?** `train(images, labels, fold, model_type, batch_size=32, num_epochs=5)`:
`images` 4-D `[n, colors, W, H]` (single frame) or 5-D `[windows, n, colors, W, H]`; `labels` 0-based integers;
`fold` = `(train_indices, test_indices)`; `model_type ∈ {'cnn','1dconv','maxpool','lstm','mix'}`.

**11. How are labels provided?** In the sample, as the **last column** of `features`, values 1–4 (working-memory load
levels; class counts 757 / 715 / 640 / 558 **[verified]**); the demo subtracts 1. `utils.load_data` documents a different MAT layout
(`featMat` + `labels`) that does not match its own code (it reads `features`) – inconsistent. **[verified]**

**12. What does the model do?** `build_cnn`: VGG-style stacks of 3×3 convolutions (`n_layers=(4,2,1)`, filters 32→64→128,
`pad='same'`), 2×2 max-pool after each stack. Single-frame model: + dropout 0.5 → dense 256 (ReLU) → dropout 0.5 → softmax.
Multi-window models (`maxpool`, `1dconv`, `lstm`, `mix`): one weight-sharing CNN per time window, then temporal integration by
element-wise max across windows, a 1-D convolution, an LSTM (128 units), or both, then dense 512/256 and softmax.
Training: categorical cross-entropy + L2 (1e-4), Adam lr 1e-3, batch 32, dropout 0.5. **[verified]**

**13. Assumptions.** Features are already computed and band-major; 64 electrodes with known 3-D positions; equal
number of features per window; class labels 1..K; one subject id per trial; a spherical-ish head so that azimuthal
projection is meaningful; Python 2 semantics (see below).

**14. Legacy dependencies.** Theano ≥0.8 (development ended in 2017) and Lasagne (unmaintained) are used for every network
and the training loop. `setup.py` pins `numpy==1.13.1`/`scipy==0.19.1`/`scikit-learn==0.18.2` and uses `distutils` (removed in Python 3.12).
The code is **Python 2**: `from utils import …` (implicit relative import) and `n_colors = features.shape[1] / nElectrodes`
used in `range(...)` (true division in Python 3 → `TypeError`). **[verified]**

**15. Reusable vs. redesign.**

| Reuse (idea) | Redesign |
|---|---|
| Band-power-per-electrode feature maps | Start from raw EEG with a documented pipeline instead of an opaque feature file |
| Azimuthal equidistant projection of 3-D electrode positions | Fit the sphere centre first; isotropic grid |
| Cubic scattered-data interpolation onto a grid; zero outside the hull | Precompute the interpolation matrix; return the hull mask; document that pixels are interpolated |
| Stack band maps as image channels | Fit normalisation on **training data only** |
| VGG-style small-kernel CNN with dropout, Adam, L2 | PyTorch, compact network, BatchNorm, binary head, deterministic seeds |
| Leave-subject-out evaluation | Subject-pure validation partition, saved split manifests, overlap tests |
| – | Multi-window recurrent models: not reimplemented (needs sequences of windows per trial; out of scope) |

## 3. Defects and methodological issues found in the original code

All **[verified]** by reading the source.

1. `reformatInput` is imported only inside `if __name__ == '__main__':` (line 457) but used by `train` (line 344). Calling
   `eeglib.train(...)` as a library, exactly as the README shows, raises `NameError`.
2. `augment_EEG_image` refers to an undefined variable `data` (it takes `image`) – dead/broken code.
3. `load_data` prints `dataMat['featMat']` but returns `dataMat['features']`.
4. **Validation split is not subject-independent**: `reformatInput` takes the first `len(test)` indices of the *shuffled*
   training index list as validation, i.e. random trials from the same training participants. Model selection by validation
   accuracy therefore does not measure generalisation to unseen participants.
5. `train` reports the test accuracy measured when validation accuracy last improved (selection by validation, not by test –
   acceptable) but the validation sets leak participant information as in (4).
6. Image normalisation uses statistics over all samples before the folds are built (see 8 above).
7. `iterate_minibatches(..., shuffle=False)` is used for training: the sample order is fixed (shuffled once per fold), not
   reshuffled every epoch.
8. Random seeds are set at import time in two modules (`123`, `1234`); fold shuffling is not seeded explicitly.
9. No automated tests; no saved configuration besides weights (`np.savez`).

## 4. How this project relates to EEGLearn

See `docs/METHODOLOGY.md` for the implemented methods. In short: **same idea** (EEG features → electrode-position-aware
images → CNN), **different implementation** (written from scratch in NumPy/SciPy/PyTorch, no EEGLearn source copied), **different
data** (raw BCI Competition IV 2a motor-imagery recordings instead of the working-memory feature file), **different task**
(binary hand imagery) and **no claim of reproducing the paper's results**.

## 5. Licence and attribution

EEGLearn is distributed under **GPL-2.0**. No source code from it was copied or translated into this project; the mathematics
(azimuthal equidistant projection, Clough–Tocher cubic interpolation, per-band standardisation) are standard methods that were
re-implemented from their definitions, and the *idea* of "EEG images" is credited to Bashivan et al. If code from EEGLearn is
ever incorporated, the combined work would have to comply with GPL-2.0.

## 6. References

Items marked ✔ were fetched/read during this work; items marked ○ are cited from memory of the literature and were **not**
re-checked online in this session – verify before citing formally.

* ✔ Bashivan, P., Rish, I., Yeasin, M., Codella, N. *Learning Representations from EEG with Deep Recurrent-Convolutional
  Neural Networks.* ICLR 2016. arXiv:1511.06448 (as cited by the repository README).
* ✔ Brunner, C., Leeb, R., Müller-Putz, G. R., Schlögl, A., Pfurtscheller, G. *BCI Competition 2008 – Graz data set A.*
  Institute for Knowledge Discovery, Graz University of Technology (official description, `desc_2a.pdf`).
* ✔ Naeem, M. et al. (2006), *J. Neural Eng.* 3, 208–216 and Brunner, C. et al. (2007), *Pattern Recognition Letters* 28, 957–964
  (the two analyses of this dataset named in the official description).
* ○ Ramoser, H., Müller-Gerking, J., Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined
  hand movement. *IEEE Trans. Rehab. Eng.* 8(4), 441–446. (CSP)
* ○ Pfurtscheller, G., Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic
  principles. *Clin. Neurophysiol.* 110, 1842–1857. (mu/beta ERD)
* ○ Lotte, F. et al. (2018). A review of classification algorithms for EEG-based brain–computer interfaces: a 10-year update.
  *J. Neural Eng.* 15, 031005.
* ○ Schirrmeister, R. T. et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization.
  *Hum. Brain Mapp.* 38, 5391–5420.
