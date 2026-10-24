"""Nested model selection for CSP + LDA, and descriptive sensitivity analyses.

Design (so that selection never sees the outer test data)
    for every OUTER split (protocol A or B, exactly the splits of the primary evaluation):
        for every candidate configuration in a small pre-defined grid:
            inner score = mean balanced accuracy of grouped K-fold CV INSIDE the outer training set
        pick the configuration with the best inner score (ties -> earlier in the grid; the a-priori default is first)
        refit on the whole outer training set, predict the outer test set ONCE
    The nested estimate is the average of those outer results.
The outer test performance of every non-selected configuration is also recorded, but ONLY as a descriptive
sensitivity table: choosing the best row of it would be selection on test data and its score would be optimistic.

Candidate grid (fixed here, before looking at any result of this analysis):
    CSP components {2, 4, 6, 8}  x  band {8-30, 8-13, 13-30 Hz}  x  window after cue {0.5-2.5, 1.0-3.0 s}
Regularisation (Ledoit-Wolf) and LDA shrinkage ('auto') are kept fixed in the grid and varied only in the
descriptive ablation, to keep the search small and interpretable.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import compute_metrics
from src.evaluation.splits import Split, cross_session_split, loso_splits
from src.models.baseline import build_csp_lda_pipeline, decision_scores

BANDS = {"8-30": (8.0, 30.0), "8-13": (8.0, 13.0), "13-30": (13.0, 30.0)}
WINDOWS = {"0.5-2.5": (0.5, 2.5), "1.0-3.0": (1.0, 3.0)}
N_COMPONENTS = (2, 4, 6, 8)
DEFAULT = ("4", "8-30", "0.5-2.5")        # the primary (final_v1) configuration


@dataclass(frozen=True)
class CSPConfig:
    n_components: int
    band: str
    window: str

    @property
    def name(self) -> str:
        return f"c{self.n_components}_b{self.band}_w{self.window}"

    @property
    def bank_key(self) -> str:
        return f"b{self.band}_w{self.window}"


def candidate_grid() -> list[CSPConfig]:
    """All 24 configurations; the a-priori default (4 components, 8-30 Hz, 0.5-2.5 s) is first, so ties go to it."""
    grid = [CSPConfig(c, b, w) for c, b, w in itertools.product(N_COMPONENTS, BANDS, WINDOWS)]
    default = CSPConfig(int(DEFAULT[0]), DEFAULT[1], DEFAULT[2])
    return [default] + [g for g in grid if g != default]


def bank_specs() -> dict:
    """Specs for ``build_epoch_bank``: one entry per (band, window) pair used by the grid."""
    return {f"b{b}_w{w}": (BANDS[b], *WINDOWS[w]) for b, w in itertools.product(BANDS, WINDOWS)}


def _fit_score(cfg, Xtr, ytr, Xte, yte, n_components, **pipe_kw):
    model = build_csp_lda_pipeline(cfg, n_components=n_components, **pipe_kw).fit(np.asarray(Xtr, dtype=np.float64), ytr)
    s = decision_scores(model, np.asarray(Xte, dtype=np.float64))
    return compute_metrics(yte, (s > 0).astype(int), s), s


def inner_score(cfg, X, y, groups, train_idx, n_components, k, **pipe_kw) -> float:
    """Mean balanced accuracy of grouped K-fold CV restricted to ``train_idx``."""
    g = groups[train_idx]
    kf = GroupKFold(n_splits=min(k, len(np.unique(g))))
    scores = []
    for a, b in kf.split(train_idx, y[train_idx], g):
        m, _ = _fit_score(cfg, X[train_idx[a]], y[train_idx[a]], X[train_idx[b]], y[train_idx[b]], n_components, **pipe_kw)
        scores.append(m["balanced_accuracy"])
    return float(np.nanmean(scores))


def make_splits(meta, y, protocol, cfg) -> list[Split]:
    if protocol == "cross_session":
        return [cross_session_split(meta, y, int(s)) for s in sorted(meta["subject"].unique())]
    return loso_splits(meta, y, cfg.models["cnn"]["n_val_subjects"])


def nested_selection(cfg, bank: dict, meta, y, protocol: str, n_jobs: int = 4, log=print):
    """Run nested selection for one protocol. Returns (rows DataFrame, selection DataFrame)."""
    grid = candidate_grid()
    groups = (meta["run"] if protocol == "cross_session" else meta["subject"]).to_numpy()
    k = cfg.evaluation["inner_cv_folds"]
    rows, picks = [], []
    for split in make_splits(meta, y, protocol, cfg):
        tr, te = split.train_idx, split.test_idx
        test_subject = int(meta.iloc[te[0]]["subject"])

        def one(conf):
            X = bank[conf.bank_key]
            inner = inner_score(cfg, X, y, groups, tr, conf.n_components, k)
            m, _ = _fit_score(cfg, X[tr], y[tr], X[te], y[te], conf.n_components)
            return conf, inner, m

        results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(one)(c) for c in grid)
        best = max(range(len(results)), key=lambda i: (round(results[i][1], 12), -i))     # earlier grid index wins ties
        for i, (conf, inner, m) in enumerate(results):
            rows.append({"protocol": protocol, "test_subject": test_subject, "config": conf.name,
                         "n_components": conf.n_components, "band": conf.band, "window": conf.window,
                         "inner_balanced_accuracy": inner, "selected": i == best,
                         "outer_balanced_accuracy": m["balanced_accuracy"], "outer_accuracy": m["accuracy"], "n_test": m["n"]})
        picks.append({"protocol": protocol, "test_subject": test_subject, "selected_config": results[best][0].name,
                      "inner_balanced_accuracy": results[best][1], "outer_balanced_accuracy": results[best][2]["balanced_accuracy"],
                      "default_outer_balanced_accuracy": results[0][2]["balanced_accuracy"]})
        log(f"  [{protocol}] held-out S{test_subject:02d}: selected {results[best][0].name} "
            f"(inner {results[best][1]:.3f}) -> outer {results[best][2]['balanced_accuracy']:.3f}; default -> {results[0][2]['balanced_accuracy']:.3f}")
    return pd.DataFrame(rows), pd.DataFrame(picks)


ABLATIONS = {
    "default (ledoit_wolf CSP, LDA auto)": {},
    "CSP unregularised": {"csp_reg": None},
    "CSP oas": {"csp_reg": "oas"},
    "LDA no shrinkage": {"lda_shrinkage": None},
    "feature StandardScaler before LDA": {"scale_features": True},
}


def ablation(cfg, X, meta, y, protocol: str, log=print) -> pd.DataFrame:
    """Descriptive sensitivity of the default configuration to regularisation choices (NOT used for selection)."""
    rows = []
    for split in make_splits(meta, y, protocol, cfg):
        tr, te = split.train_idx, split.test_idx
        s = int(meta.iloc[te[0]]["subject"])
        for name, kw in ABLATIONS.items():
            m, _ = _fit_score(cfg, X[tr], y[tr], X[te], y[te], cfg.models["csp_lda"]["n_components"], **kw)
            rows.append({"protocol": protocol, "test_subject": s, "variant": name, "balanced_accuracy": m["balanced_accuracy"],
                         "accuracy": m["accuracy"], "n_test": m["n"]})
    return pd.DataFrame(rows)
