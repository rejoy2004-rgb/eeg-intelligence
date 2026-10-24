"""Regression tests added by the research audit (synthetic data; see conftest.py warning).

They check the *mechanisms* that protect the evaluation: what is actually passed to each fit, that the leakage audit
really detects an injected leak, that epochs never touch missing-data gaps, that summaries are recomputable from
per-participant results, and the new components (alignment, ERD features, nested selection, negative control).
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import csp_selection as cs
from src.evaluation import experiments as exp
from src.evaluation.epoch_bank import with_epochs
from src.evaluation.experiments import aggregate, run_protocol
from src.evaluation.leakage_audit import audit_protocol
from src.evaluation.metrics import summarize
from src.features.alignment import align_by_group, euclidean_alignment
from src.models.baseline import ERDFeatures, erd_band_power
from src.signal.epoching import TrialSet, build_dual_band_trials


@pytest.fixture
def small_cfg(cfg):
    cfg["models"]["cnn"].update(max_epochs=2, patience=2, batch_size=16, n_val_subjects=1)
    return cfg


@pytest.fixture
def dataset(small_cfg, make_recording):
    bps, csps = [], []
    for subject in (1, 2, 3):
        for session in ("T", "E"):
            rec, label4 = make_recording(subject, session, n_trials=32, n_runs=2)
            bp, csp = build_dual_band_trials(rec, small_cfg, label4)
            bps.append(bp)
            csps.append(csp)
    return TrialSet.concat(bps), TrialSet.concat(csps)


# ---- leakage audit machinery ---------------------------------------------------------------
@pytest.mark.parametrize("protocol", ["cross_session", "loso"])
def test_runtime_audit_passes_on_the_real_code_path(dataset, small_cfg, protocol):
    bp, csp = dataset
    report = audit_protocol(small_cfg, bp, csp, protocol, progress=lambda *_: None)
    assert report["problems"] == [] and report["all_passed"]
    assert report["duplicate_epochs_in_dataset"] == 0
    assert report["n_folds"] == 3
    names = set(report["folds"][0]["checks"])
    assert {"csp_fit_ids_equal_train", "csp_filters_equal_refit_on_training_only", "logreg_scaler_mean_equals_training_mean",
            "image_scaler_mean_equals_cnn_fit_part_mean", "cnn_fit_val_test_pairwise_disjoint"} <= names


def test_runtime_audit_detects_an_injected_leak(dataset, small_cfg, monkeypatch):
    """If CSP were (wrongly) also fitted on a few test-participant trials, the audit must fail."""
    bp, csp = dataset
    original = exp.fit_csp_lda

    def leaky_fit(cfg, X, y):
        extra = np.flatnonzero((bp.meta["subject"] == 1).to_numpy())[:6]       # trials of participant 1 = test in fold 1
        return original(cfg, np.concatenate([X, csp.X[extra]]), np.concatenate([y, bp.y[extra]]))

    monkeypatch.setattr(exp, "fit_csp_lda", leaky_fit)
    report = audit_protocol(small_cfg, bp, csp, "loso", models=("csp_lda",), progress=lambda *_: None)
    assert not report["all_passed"]
    # the leak is hidden INSIDE the fit function, so it is caught by the numerical refit check
    assert any("csp_filters_equal_refit_on_training_only" in p for p in report["problems"])


def test_runtime_audit_detects_scaler_fitted_on_test_data(dataset, small_cfg, monkeypatch):
    bp, csp = dataset
    original = exp.fit_logreg

    def leaky(cfg, sfreq, X, y, groups, seed=0, features="power"):
        extra = np.flatnonzero((bp.meta["subject"] == 1).to_numpy())[:6]
        return original(cfg, sfreq, np.concatenate([X, bp.X[extra]]), np.concatenate([y, bp.y[extra]]),
                        np.concatenate([groups, np.full(6, 99)]), seed, features)

    monkeypatch.setattr(exp, "fit_logreg", leaky)
    report = audit_protocol(small_cfg, bp, csp, "loso", models=("logreg",), progress=lambda *_: None)
    assert not report["all_passed"]


def test_duplicate_epochs_are_reported(dataset, small_cfg):
    bp, csp = dataset
    dup = TrialSet(np.concatenate([bp.X, bp.X[:1]]), np.concatenate([bp.y, bp.y[:1]]),
                   pd.concat([bp.meta, bp.meta.iloc[:1]], ignore_index=True), bp.rejections, bp.sfreq, bp.ch_names)
    dupc = TrialSet(np.concatenate([csp.X, csp.X[:1]]), np.concatenate([csp.y, csp.y[:1]]), dup.meta, csp.rejections, csp.sfreq, csp.ch_names)
    # the duplicate sits at index 0's identity; audit counts identical epoch content
    report = audit_protocol(small_cfg, dup, dupc, "cross_session", models=("csp_lda",), progress=lambda *_: None)
    assert report["duplicate_epochs_in_dataset"] == 1 and not report["all_passed"]


# ---- epochs, gaps, labels --------------------------------------------------------------------
def test_no_kept_epoch_touches_a_missing_gap(synth, cfg):
    rec, _ = synth
    bp, _ = build_dual_band_trials(rec, cfg)
    t0, t1 = cfg.preprocessing["epoch_tmin_s"], cfg.preprocessing["epoch_tmax_s"]
    for cue in bp.meta["cue_sample"]:
        a, b = cue + round(t0 * rec.sfreq), cue + round(t1 * rec.sfreq)
        assert all(b <= g0 or a >= g1 for g0, g1 in rec.missing_segments)
    assert np.isfinite(bp.X).all()


def test_epoch_data_equals_filtered_signal_at_the_cue(synth, cfg):
    """Label/time alignment: the epoch must be the preprocessed signal at cue+0.5..cue+2.5 s, sample for sample."""
    from src.signal.preprocessing import preprocess_recording

    rec, _ = synth
    bp, _ = build_dual_band_trials(rec, cfg)
    sig = preprocess_recording(rec, cfg)
    for i in (0, 5, bp.n_trials - 1):
        a = int(bp.meta.loc[i, "cue_sample"]) + 125
        assert np.array_equal(bp.X[i], sig[:, a:a + 500])


def test_each_model_is_scored_on_identical_trials(dataset, small_cfg):
    bp, csp = dataset
    _, _, preds = run_protocol(small_cfg, bp, csp, "loso", log=lambda *_: None)
    sets = [frozenset(zip(g["test_subject"], g["session"], g["trial"])) for _, g in preds.groupby("model")]
    assert len(set(sets)) == 1 and len(sets[0]) == bp.n_trials


# ---- aggregation / summaries -------------------------------------------------------------------
def test_summary_statistics_match_recomputation_from_rows(dataset, small_cfg):
    bp, csp = dataset
    res, _, _ = run_protocol(small_cfg, bp, csp, "cross_session", models=("logreg", "csp_lda"), log=lambda *_: None)
    agg = aggregate(res)["cross_session"]
    for model in ("logreg", "csp_lda"):
        v = res[res["model"] == model]["balanced_accuracy"].to_numpy()
        assert agg[model]["balanced_accuracy"]["mean"] == pytest.approx(v.mean())
        assert agg[model]["balanced_accuracy"]["std"] == pytest.approx(v.std(ddof=1))
        assert agg[model]["balanced_accuracy"]["n"] == len(v) == 3
        cm = np.array(agg[model]["pooled_confusion_matrix"])
        sub = res[res["model"] == model]
        assert cm.sum() == sub["n_test"].sum() and cm[1, 1] == sub["tp"].sum() and cm[0, 0] == sub["tn"].sum()


def test_summarize_uses_sample_std_and_ignores_nan():
    s = summarize([0.5, 0.7, 0.9, np.nan])
    assert s["n"] == 3 and s["mean"] == pytest.approx(0.7) and s["std"] == pytest.approx(0.2)


# ---- negative control ---------------------------------------------------------------------------
def test_permutation_control_keeps_test_labels_and_removes_signal(dataset, small_cfg):
    bp, csp = dataset
    y_before = bp.y.copy()
    real, _, _ = run_protocol(small_cfg, bp, csp, "cross_session", models=("csp_lda",), log=lambda *_: None)
    perm, _, preds = run_protocol(small_cfg, bp, csp, "cross_session", models=("csp_lda",), log=lambda *_: None,
                                  permute_training_labels=True)
    assert np.array_equal(bp.y, y_before)                                   # data object untouched
    # the evaluated labels are the TRUE labels of the E-session trials
    truth = {(int(s), int(t)): int(y) for s, t, y, sess in zip(bp.meta["subject"], bp.meta["trial"], bp.y, bp.meta["session"]) if sess == "E"}
    assert len(preds) == len(truth)
    assert all(truth[(int(r.test_subject), int(r.trial))] == int(r.y_true) for r in preds.itertuples())
    assert real["balanced_accuracy"].mean() > perm["balanced_accuracy"].mean() + 0.15


# ---- Euclidean alignment ------------------------------------------------------------------------
def test_alignment_gives_identity_mean_covariance_and_is_label_free():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    X = np.einsum("cd,ndt->nct", A, rng.normal(size=(40, 6, 200)))
    Z = euclidean_alignment(X)
    cov = np.einsum("nct,ndt->cd", Z, Z) / (Z.shape[0] * Z.shape[2])
    assert np.allclose(cov, np.eye(6), atol=1e-8)
    assert Z.shape == X.shape


def test_alignment_is_applied_per_group_and_keeps_row_order():
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(size=(10, 4, 100)) * 3, rng.normal(size=(10, 4, 100)) * 0.2])
    g = np.array(["a"] * 10 + ["b"] * 10)
    Z = align_by_group(X, g)
    for lab in ("a", "b"):
        m = g == lab
        cov = np.einsum("nct,ndt->cd", Z[m], Z[m]) / (m.sum() * 100)
        assert np.allclose(cov, np.eye(4), atol=1e-8)
    assert np.allclose(Z[:10], euclidean_alignment(X[:10]))


def test_alignment_rejects_bad_input():
    with pytest.raises(ValueError):
        euclidean_alignment(np.zeros((0, 3, 10)))
    with pytest.raises(ValueError):
        euclidean_alignment(np.full((2, 3, 10), np.nan))
    with pytest.raises(ValueError, match="singular"):
        euclidean_alignment(np.ones((3, 3, 10)))
    with pytest.raises(ValueError):
        align_by_group(np.zeros((3, 2, 5)), np.array([1, 2]))


# ---- ERD features ---------------------------------------------------------------------------------
def test_erd_equals_task_minus_baseline_log_power_and_needs_long_epochs():
    from src.features.spectral import log_band_power

    X = np.random.default_rng(0).normal(0, 3, (4, 5, 1000))
    bands = {"mu": [8, 12], "beta": [13, 30]}
    erd = erd_band_power(X, 250.0, bands)
    assert erd.shape == (4, 2, 5) and np.isfinite(erd).all()
    assert np.allclose(erd, log_band_power(X[..., 500:], 250.0, bands) - log_band_power(X[..., :250], 250.0, bands))
    # a pure amplitude scaling of the whole epoch cancels (offset invariance - the point of the baseline)
    assert np.allclose(erd_band_power(10 * X, 250.0, bands), erd, atol=1e-9)
    with pytest.raises(ValueError, match="needs at least"):
        erd_band_power(X[..., :500], 250.0, bands)
    assert ERDFeatures(250.0, bands).fit_transform(X).shape == (4, 10)


# ---- nested selection --------------------------------------------------------------------------------
def test_candidate_grid_is_small_unique_and_default_first():
    g = cs.candidate_grid()
    assert len(g) == 24 and len({c.name for c in g}) == 24
    assert (str(g[0].n_components), g[0].band, g[0].window) == cs.DEFAULT
    assert set(cs.bank_specs()) == {c.bank_key for c in g}


def test_selection_never_depends_on_test_labels(dataset, small_cfg, monkeypatch):
    """Flip all labels of the held-out participant: inner scores and the selected configuration must not change."""
    bp, csp = dataset
    four = cs.candidate_grid()[:4]
    monkeypatch.setattr(cs, "candidate_grid", lambda: four)
    bank = {c.bank_key: csp.X for c in four}
    r1, k1 = cs.nested_selection(small_cfg, bank, bp.meta, bp.y, "loso", n_jobs=1, log=lambda *_: None)
    y2 = bp.y.copy()
    y2[(bp.meta["subject"] == 1).to_numpy()] ^= 1                           # corrupt ONLY the held-out participant's labels
    r2, k2 = cs.nested_selection(small_cfg, bank, bp.meta, y2, "loso", n_jobs=1, log=lambda *_: None)
    s1 = k1[k1["test_subject"] == 1]["selected_config"].iloc[0]
    s2 = k2[k2["test_subject"] == 1]["selected_config"].iloc[0]
    assert s1 == s2
    a = r1[r1.test_subject == 1]["inner_balanced_accuracy"].to_numpy()
    b = r2[r2.test_subject == 1]["inner_balanced_accuracy"].to_numpy()
    assert np.allclose(a, b)                                                # inner scores identical: test labels unused
    assert len(k1) == 3 and k1["selected_config"].notna().all()
    # ...whereas the OUTER (test) score does change, proving the labels really were used for scoring only
    o1 = r1[(r1.test_subject == 1) & r1.selected]["outer_balanced_accuracy"].iloc[0]
    o2 = r2[(r2.test_subject == 1) & r2.selected]["outer_balanced_accuracy"].iloc[0]
    assert o1 != o2


def test_ablation_variants_include_the_default(dataset, small_cfg):
    bp, csp = dataset
    ab = cs.ablation(small_cfg, csp.X, bp.meta, bp.y, "cross_session", log=lambda *_: None)
    assert set(ab["variant"]) == set(cs.ABLATIONS) and len(ab) == 3 * len(cs.ABLATIONS)
    res, _, _ = run_protocol(small_cfg, bp, csp, "cross_session", models=("csp_lda",), log=lambda *_: None)
    default = ab[ab["variant"].str.startswith("default")].sort_values("test_subject")["balanced_accuracy"].to_numpy()
    assert np.allclose(default, res.sort_values("test_subject")["balanced_accuracy"].to_numpy())   # same as the primary model


# ---- measured-only layout grid ------------------------------------------------------------------------
def test_layout_grid_follows_the_official_figure(cfg):
    from src.features.spatial import LayoutGridBuilder, SpatialError

    b = LayoutGridBuilder()
    names = cfg.dataset["eeg_channels"]
    pos = {n: (int(b.rows[i]), int(b.cols[i])) for i, n in enumerate(names)}
    assert b.mask.sum() == 22 and b.mask.shape == (8, 8)
    assert pos["C3"][0] == pos["Cz"][0] == pos["C4"][0] and pos["C3"][1] < pos["Cz"][1] < pos["C4"][1]
    assert pos["Fz"][0] < pos["FCz"][0] < pos["Cz"][0] < pos["CPz"][0] < pos["Pz"][0] < pos["POz"][0]
    assert pos["Fz"][1] == pos["FCz"][1] == pos["Cz"][1] == pos["CPz"][1] == pos["Pz"][1] == pos["POz"][1]
    feats = np.arange(22, dtype=float)[None, None, :]
    img = b.transform(feats)
    assert img.shape == (1, 1, 8, 8) and np.isnan(img[0, 0][~b.mask]).all()
    assert all(img[0, 0, r, c] == i for i, (r, c) in enumerate(zip(b.rows, b.cols)))      # value i sits at electrode i's pixel
    with pytest.raises(SpatialError):
        b.transform(np.zeros((1, 1, 21)))
    with pytest.raises(SpatialError):
        b.transform(np.full((1, 1, 22), np.nan))
