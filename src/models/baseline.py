"""Classical baselines built as scikit-learn pipelines that consume raw epochs (n, C, T).

Putting feature extraction *inside* the pipeline guarantees that training, evaluation and prediction
use identical steps, and that every learned step (scaler, CSP filters, classifier) is fitted on the
training data of the current split only.

Baseline 1  ``logreg``   : log band power (mu / low-beta / high-beta per channel)
                           -> StandardScaler -> L2 logistic regression (C chosen by grouped inner CV)
Baseline 2  ``csp_lda``  : 8-30 Hz epochs -> Common Spatial Patterns (log-variance of 4 spatial filters)
                           -> shrinkage LDA. A standard motor-imagery method (Ramoser et al., 2000).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.spectral import flatten_band_major, log_band_power


class BandPowerFeatures(BaseEstimator, TransformerMixin):
    """(n, C, T) epochs -> (n, B*C) log band power, band-major. Stateless (nothing is learned)."""

    def __init__(self, sfreq=250.0, bands=None, nperseg=250, overlap=0.5):
        self.sfreq = sfreq
        self.bands = bands
        self.nperseg = nperseg
        self.overlap = overlap

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return flatten_band_major(log_band_power(np.asarray(X), self.sfreq, self.bands, self.nperseg, self.overlap))


def erd_band_power(X, sfreq, bands, nperseg=250, overlap=0.5, base=(0, 250), task=(500, 1000)):
    """Event-related (de)synchronisation features: log band power in the task window minus log band power in a
    pre-cue baseline window of the SAME trial. Shape (n, bands, channels).

    X must be long epochs; with the audit window (cue-1.5 ... cue+2.5 s at 250 Hz, 1000 samples) the baseline
    samples [0, 250) are cue-1.5 ... cue-0.5 s (fixation period, after the beep) and the task samples [500, 1000)
    are cue+0.5 ... cue+2.5 s (the original task window). Uses only the trial's own samples: no label, no
    training-set or test-set statistics, so it cannot leak.
    The baseline has a single 1 s Welch segment (noisier than the 3-segment task estimate).
    """
    X = np.asarray(X)
    if X.shape[-1] < max(base[1], task[1]):
        raise ValueError(f"Epochs have {X.shape[-1]} samples; ERD needs at least {max(base[1], task[1])}")
    t = log_band_power(X[..., task[0]:task[1]], sfreq, bands, nperseg, overlap)
    b = log_band_power(X[..., base[0]:base[1]], sfreq, bands, nperseg, overlap)
    return t - b


class ERDFeatures(BaseEstimator, TransformerMixin):
    """(n, C, T_long) epochs -> (n, B*C) baseline-normalised log band power. Stateless."""

    def __init__(self, sfreq=250.0, bands=None, nperseg=250, overlap=0.5, base=(0, 250), task=(500, 1000)):
        self.sfreq = sfreq
        self.bands = bands
        self.nperseg = nperseg
        self.overlap = overlap
        self.base = base
        self.task = task

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return flatten_band_major(erd_band_power(X, self.sfreq, self.bands, self.nperseg, self.overlap, self.base, self.task))


def build_logreg_pipeline(cfg, sfreq: float, c: float = 1.0, features: str = "power") -> Pipeline:
    f = cfg.features
    if features == "power":
        first = BandPowerFeatures(sfreq, dict(f["bands_hz"]), f["welch_nperseg"], f["welch_overlap"])
    elif features == "erd":
        first = ERDFeatures(sfreq, dict(f["bands_hz"]), f["welch_nperseg"], f["welch_overlap"])
    else:
        raise ValueError("features must be 'power' or 'erd'")
    return Pipeline(
        [
            ("features", first),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=2000, solver="lbfgs")),
        ]
    )


def build_csp_lda_pipeline(cfg, n_components: int | None = None, csp_reg="ledoit_wolf", lda_shrinkage="auto",
                           scale_features: bool = False) -> Pipeline:
    """CSP -> (optional StandardScaler) -> LDA. Defaults reproduce the primary (final_v1) model exactly."""
    import mne
    from mne.decoding import CSP

    mne.set_log_level("ERROR")

    n_comp = cfg.models["csp_lda"]["n_components"] if n_components is None else n_components
    steps = [("csp", CSP(n_components=n_comp, reg=csp_reg, log=True, norm_trace=False))]
    if scale_features:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=lda_shrinkage)))
    return Pipeline(steps)


def fit_logreg(cfg, sfreq: float, X, y, groups, seed: int = 0, features: str = "power"):
    """Fit the band-power logistic regression, choosing C by grouped inner CV on TRAINING data only.

    ``groups`` must be a group label per training trial (run id within a session, or subject id),
    so inner folds never split a group - mirroring the outer split's independence assumption.
    Returns (fitted_pipeline, {"best_C": ..., "inner_cv_balanced_accuracy": ..., "n_groups": ...}).
    """
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for inner cross-validation")
    k = min(cfg.evaluation["inner_cv_folds"], n_groups)
    search = GridSearchCV(
        build_logreg_pipeline(cfg, sfreq, features=features),
        param_grid={"clf__C": list(cfg.models["logreg"]["c_grid"])},
        scoring="balanced_accuracy",
        cv=GroupKFold(n_splits=k),
        refit=True,
    )
    search.fit(X, y, groups=groups)
    info = {
        "best_C": float(search.best_params_["clf__C"]),
        "inner_cv_balanced_accuracy": float(search.best_score_),
        "n_groups": int(n_groups),
        "inner_folds": int(k),
    }
    return search.best_estimator_, info


def fit_csp_lda(cfg, X, y):
    """CSP+LDA has no hyper-parameter search here (fixed a priori in the config)."""
    pipe = build_csp_lda_pipeline(cfg)
    pipe.fit(X, y)
    return pipe


def decision_scores(model, X) -> np.ndarray:
    """Signed decision value; positive favours class 1 (right hand). NOT a calibrated probability."""
    return np.asarray(model.decision_function(X))
