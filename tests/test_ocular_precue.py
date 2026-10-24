"""Tests for the ocular-robust evaluation and the pre-cue investigation (synthetic data; see conftest.py warning).

Covers: EOG regression (fit on calibration data only), frontal-exclusion definition tied to the frozen config, causal filtering,
pre-cue control sensitivity, paired-comparison hygiene, held-out-label independence, baseline-overwrite protection, statistics helpers.
"""
import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import PROJECT_ROOT
from src.data.dataset import recording_from_raw
from src.data.validation import DataValidationError
from src.evaluation import precue as pc
from src.evaluation import stats as st
from src.evaluation.experiments import PROTECTED_RUNS, run_protocol, save_experiment
from src.evaluation.ocular_bank import FRONTAL
from src.models.baseline import build_csp_lda_pipeline, decision_scores
from src.signal.epoching import TrialSet, build_dual_band_trials, extract_epochs, locate_trials
from src.signal.ocular import (apply_eog_regression, calibration_slice, eog_features, extract_eog, fit_eog_regression)
from src.signal.preprocessing import PreprocessingError, bandpass_filter, preprocess_recording
from tests.conftest import make_synthetic_raw


# ------------------------------------------------------------------------------ causal filter / pre-cue window
def test_causal_filter_cannot_see_the_future_but_zero_phase_can():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 4000))
    x2 = x.copy()
    x2[:, 2500:] += 50.0                                   # change ONLY later samples
    for causal, must_change in ((True, False), (False, True)):
        a = bandpass_filter(x, 250.0, 8, 30, 4, causal=causal)
        b = bandpass_filter(x2, 250.0, 8, 30, 4, causal=causal)
        near = not np.allclose(a[:, 2300:2500], b[:, 2300:2500], atol=1e-6)        # the 0.8 s just before the change
        assert near == must_change
    # the zero-phase backward spread is short: 2 s earlier nothing is left (this is what the impulse profile quantifies)
    z0, z1 = bandpass_filter(x, 250.0, 8, 30, 4), bandpass_filter(x2, 250.0, 8, 30, 4)
    assert np.allclose(z0[:, :2000], z1[:, :2000], atol=1e-6)


def test_impulse_profile_shows_backward_spread_only_for_zero_phase():
    for band in ((4.0, 40.0), (8.0, 30.0)):
        prof = pc.impulse_leakage_profile(band)
        assert prof["causal"]["energy_fraction_before_pre_window_end"] == 0.0
        assert prof["zero_phase"]["energy_fraction_before_pre_window_end"] >= 0.0
        assert prof["zero_phase"]["earliest_sample_with_1pct_of_peak_s_before_event"] > 0.0
        assert prof["causal"]["earliest_sample_with_1pct_of_peak_s_before_event"] == 0.0


def test_preprocess_recording_causal_option(synth, cfg):
    rec, _ = synth
    z, c = preprocess_recording(rec, cfg), preprocess_recording(rec, cfg, causal=True)
    assert z.shape == c.shape and np.array_equal(np.isnan(z), np.isnan(c)) and not np.allclose(np.nan_to_num(z), np.nan_to_num(c))


def test_precue_window_lies_before_the_cue_and_inside_the_trial(synth, cfg):
    rec, _ = synth
    tr = locate_trials(rec, cfg)
    a = pc.timing_audit(rec, tr, -1.5, -0.5)
    assert a["window_end_before_cue_s[min,median,max]"] == [0.5, 0.5, 0.5]
    assert a["window_start_after_trial_start_s[min,median,max]"] == [0.5, 0.5, 0.5]
    assert a["windows_crossing_trial_start_or_cue"] == 0 and a["windows_overlapping_missing_gap"] == 0


def test_precue_control_is_blind_to_post_cue_information_but_detects_a_leaky_window(cfg):
    """Sensitivity of the negative control: planted effect exists only AFTER the cue (cue+0.5..cue+2.5 s)."""
    raw, label4 = make_synthetic_raw(cfg, n_trials=160, n_runs=4, seed=3)      # 40 test trials; chance level checked over 6 seeds: 0.38-0.55
    rec = recording_from_raw(raw, cfg, 1, "T")
    bp, _ = build_dual_band_trials(rec, cfg, label4)
    sig = preprocess_recording(rec, cfg, (8.0, 30.0), causal=True)          # causal: nothing from after the cue can enter
    y = bp.y
    tr, te = np.arange(len(y)) < len(y) // 2, np.arange(len(y)) >= len(y) // 2

    def acc(tmin, tmax):
        ep, why = extract_epochs(sig, rec.sfreq, bp.meta["cue_sample"].to_numpy(), tmin, tmax)
        assert not any(why)
        m = build_csp_lda_pipeline(cfg).fit(ep[tr], y[tr])
        return ((decision_scores(m, ep[te]) > 0).astype(int) == y[te]).mean()

    pre, task, leaky = acc(-1.5, -0.5), acc(0.5, 2.5), acc(-0.5, 1.5)
    assert pre < 0.65                        # pre-cue: no class information in the data -> near chance
    assert task > 0.9                        # the task window carries the planted effect
    assert leaky > 0.9 and leaky > pre + 0.25   # a window overlapping the effect (a 'leaky' control) is detected as informative


def test_label_sequence_statistics_and_previous_trial_labels():
    rng = np.random.default_rng(0)
    same = rng.integers(1, 5, 288)
    labels = {(1, "T"): same, (1, "E"): same.copy(), (2, "T"): rng.integers(1, 5, 288), (2, "E"): rng.integers(1, 5, 288)}
    s = pc.label_sequence_stats(labels)
    assert s["position_agreement_between_sessions_of_same_participant[min,mean,max]"][2] == 1.0       # identical sequences detected
    assert 0.15 < s["position_agreement_between_different_participants[min,mean,max]"][1] < 0.35        # ~chance 0.25 otherwise
    meta = pd.DataFrame({"subject": [1, 1, 1], "session": ["T"] * 3, "trial": [0, 1, 2]})
    prev = pc.previous_trial_labels(meta, {(1, "T"): np.array([1, 2, 1])}, {(1, "T"): np.array([0, 0, 1])})
    assert np.isnan(prev[0]) and prev[1] == 1 and np.isnan(prev[2])         # first trial of a run has no previous trial


# ------------------------------------------------------------------------------ EOG regression
def test_regression_recovers_known_coefficients_and_removes_the_eog_component():
    rng = np.random.default_rng(0)
    eog = rng.normal(size=(3, 20000))
    B_true = rng.normal(size=(5, 3))
    eeg = B_true @ eog + 0.2 * rng.normal(size=(5, 20000))
    B, d = fit_eog_regression(eeg, eog, (0, 10000 * 1))
    assert np.allclose(B, B_true, atol=0.02)
    clean = apply_eog_regression(eeg, eog, B)
    assert np.abs(np.corrcoef(np.vstack([clean[0], eog]))[0, 1:]).max() < 0.02
    assert d["r2_split_half"].min() > 0.9


def test_regression_uses_only_the_calibration_block():
    """Changing any data outside the block (e.g. the trials) must not change the coefficients."""
    rng = np.random.default_rng(1)
    eog, eeg = rng.normal(size=(3, 12000)), rng.normal(size=(4, 12000))
    B1, _ = fit_eog_regression(eeg, eog, (0, 6000))
    eeg2, eog2 = eeg.copy(), eog.copy()
    eeg2[:, 6000:] += 100 * rng.normal(size=(4, 6000))
    eog2[:, 6000:] *= 7
    B2, _ = fit_eog_regression(eeg2, eog2, (0, 6000))
    assert np.array_equal(B1, B2)


def test_regression_keeps_gaps_and_validates_shapes():
    eeg, eog = np.random.default_rng(2).normal(size=(4, 9000)), np.random.default_rng(3).normal(size=(3, 9000))
    eeg[:, 5000:5100] = np.nan
    eog[:, 5000:5100] = np.nan
    B, _ = fit_eog_regression(eeg, eog, (0, 9000))
    out = apply_eog_regression(eeg, eog, B)
    assert np.isnan(out[:, 5000:5100]).all() and np.isfinite(np.delete(out, np.s_[5000:5100], axis=1)).all()
    with pytest.raises(PreprocessingError):
        apply_eog_regression(eeg, eog, np.zeros((3, 3)))
    with pytest.raises(PreprocessingError):
        fit_eog_regression(eeg[:, :500], eog[:, :500], (0, 500))             # too few samples


def test_calibration_block_is_found_and_separate_from_trials(cfg):
    raw, _ = make_synthetic_raw(cfg, calibration_s=60.0)
    rec = recording_from_raw(raw, cfg, 1, "T")
    a, b = calibration_slice(rec)
    assert (a, b) == (0, 15000) and b <= rec.events.loc[rec.events["code"] == 768, "sample"].min()
    eog = extract_eog(raw, rec)
    assert eog.shape == (3, rec.n_samples) and np.isnan(eog[:, rec.missing_segments[0][0]:rec.missing_segments[0][1]]).all()


def test_calibration_block_errors(cfg, synth):
    rec, _ = synth                                                           # synthetic file without a 1072 block
    with pytest.raises(DataValidationError, match="exactly one"):
        calibration_slice(rec)
    raw, _ = make_synthetic_raw(cfg, calibration_s=10.0)
    with pytest.raises(DataValidationError, match="shorter"):
        calibration_slice(recording_from_raw(raw, cfg, 1, "T"))
    raw, _ = make_synthetic_raw(cfg, calibration_s=60.0)
    r = recording_from_raw(raw, cfg, 1, "T")
    r.events.loc[r.events["code"] == 1072, "duration"] = r.n_samples          # block would swallow the trials
    with pytest.raises(DataValidationError, match="overlaps the trial"):
        calibration_slice(r)


def test_regression_on_a_recording_reduces_planted_eog_leakage(cfg):
    raw, label4 = make_synthetic_raw(cfg, calibration_s=60.0, eog_leak=0.6)
    rec = recording_from_raw(raw, cfg, 1, "T")
    eog = extract_eog(raw, rec)
    from src.signal.ocular import filter_eog

    band = (8.0, 30.0)
    eeg_f, eog_f = preprocess_recording(rec, cfg, band), filter_eog(eog, rec.sfreq, band, 4)
    B, d = fit_eog_regression(eeg_f, eog_f, calibration_slice(rec))
    clean = apply_eog_regression(eeg_f, eog_f, B)
    ok = np.isfinite(eeg_f).all(axis=0) & np.isfinite(eog_f).all(axis=0)
    corr = lambda a: np.abs([np.corrcoef(a[c, ok], eog_f[e, ok])[0, 1] for c in range(22) for e in range(3)]).mean()
    assert corr(eeg_f) > 0.15 and corr(clean) < 0.02                         # planted EOG component removed (correlation with EOG ~ 0)
    assert clean[:, ok].var() < eeg_f[:, ok].var()
    assert d["r2_in_sample"].mean() > 0.1


def test_eog_features_shape_and_gap_handling():
    eog = np.random.default_rng(0).normal(size=(3, 5000))
    eog[:, 2000:2100] = np.nan
    f = eog_features(eog, np.array([1000, 2050, 4900]), 250.0, 0.5, 2.5)
    assert f.shape == (3, 6) and np.isfinite(f[0]).all()
    assert np.isnan(f[1]).all() and np.isnan(f[2]).all()                     # window over a gap / beyond the recording


# ------------------------------------------------------------------------------ frozen design
def test_frontal_exclusion_matches_the_frozen_configuration():
    plan = yaml.safe_load((PROJECT_ROOT / "configs" / "ocular_robust_v1.yaml").read_text(encoding="utf-8"))
    assert tuple(plan["strategies"]["frontal_exclusion"]["channels_excluded"]) == FRONTAL
    assert plan["strategies"]["eog_regression_train_only"]["zero_shot_with_respect_to_target_participant"] is True
    assert plan["strategies"]["eog_regression_target_calibrated"]["zero_shot_with_respect_to_target_participant"] is False
    assert plan["eog_regression"]["calibration_event_code"] == 1072 and plan["eog_regression"]["regulariser"] == "none"
    assert plan["models"] == ["csp_lda", "logreg"] and "cnn" not in plan["models"]


# ------------------------------------------------------------------------------ pairing, statistics
def _preds(keys, model="m"):
    return pd.DataFrame({"protocol": "loso", "model": model, "test_subject": [k[0] for k in keys], "session": [k[1] for k in keys], "trial": [k[2] for k in keys]})


def test_paired_comparisons_require_identical_test_trials():
    a = _preds([(1, "T", 0), (1, "T", 1), (2, "E", 0)])
    assert st.assert_same_test_trials(a, a.copy()) == 3
    with pytest.raises(st.PairingError, match="identical"):
        st.assert_same_test_trials(a, _preds([(1, "T", 0), (1, "T", 1), (2, "E", 1)]))
    with pytest.raises(st.PairingError, match="Duplicate"):
        st.assert_same_test_trials(pd.concat([a, a]), a)
    with pytest.raises(st.PairingError):
        st.assert_same_test_trials(a)


def test_holm_and_paired_statistics():
    assert np.allclose(st.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(st.holm([0.5, np.nan, 0.01])[[0, 2]], [0.5, 0.02]) and np.isnan(st.holm([0.5, np.nan, 0.01])[1])
    a = pd.Series(np.linspace(0.55, 0.75, 9), index=range(1, 10))
    b = a - 0.05 + np.array([0.01, -0.01, 0.0, 0.02, -0.02, 0.01, 0.0, 0.01, -0.01])
    r = st.paired_comparison(a, b)
    assert r["n_participants"] == 9 and r["ci_low"] <= r["mean_diff"] <= r["ci_high"] and 0 < r["p_wilcoxon"] < 0.05
    with pytest.raises(st.PairingError):
        st.paired_comparison(a, b.drop(9))


def test_within_run_permutation_test_controls_false_positives():
    rng = np.random.default_rng(0)
    n = 600
    parts = np.repeat([1, 2, 3], n // 3)
    groups = np.tile(np.repeat(np.arange(4), n // 12), 3) + 10 * parts
    y = rng.integers(0, 2, n)
    random_pred = rng.integers(0, 2, n)
    assert st.within_group_permutation_test(y, random_pred, groups, parts, n_perm=400)["p_one_sided"] > 0.05
    perfect = st.within_group_permutation_test(y, y.copy(), groups, parts, n_perm=400)
    assert perfect["p_one_sided"] < 0.01 and perfect["observed_mean_ba"] == 1.0


# ------------------------------------------------------------------------------ run_protocol hygiene
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


@pytest.mark.parametrize("protocol", ["loso", "cross_session"])
def test_held_out_labels_never_influence_predictions(dataset, small_cfg, protocol):
    """Flip every label of the held-out participant (and nothing else): that fold's predictions and scores must be identical."""
    bp, csp = dataset
    flipped = bp.y.copy()
    held = (bp.meta["subject"] == 1).to_numpy() & ((bp.meta["session"] == "E").to_numpy() if protocol == "cross_session" else True)
    flipped[held] ^= 1
    bp2 = TrialSet(bp.X, flipped, bp.meta, bp.rejections, bp.sfreq, bp.ch_names)
    csp2 = TrialSet(csp.X, flipped, csp.meta, csp.rejections, csp.sfreq, csp.ch_names)
    _, _, p1 = run_protocol(small_cfg, bp, csp, protocol, log=lambda *_: None)
    _, _, p2 = run_protocol(small_cfg, bp2, csp2, protocol, log=lambda *_: None)
    a, b = p1[p1.test_subject == 1].sort_values(["model", "trial", "session"]), p2[p2.test_subject == 1].sort_values(["model", "trial", "session"])
    assert np.array_equal(a["y_pred"], b["y_pred"]) and np.allclose(a["score"], b["score"], atol=0, rtol=0)
    assert not np.array_equal(a["y_true"], b["y_true"])                      # the labels really differ; only the scoring uses them


def test_strategies_are_scored_on_identical_test_trials(dataset, small_cfg):
    bp, csp = dataset
    keep = [i for i, c in enumerate(bp.ch_names) if c not in FRONTAL]
    sub = lambda ts: TrialSet(ts.X[:, keep], ts.y, ts.meta, ts.rejections, ts.sfreq, [ts.ch_names[i] for i in keep])
    for p in ("loso", "cross_session"):
        _, _, a = run_protocol(small_cfg, bp, csp, p, models=("csp_lda",), log=lambda *_: None)
        _, _, b = run_protocol(small_cfg, sub(bp), sub(csp), p, models=("csp_lda",), log=lambda *_: None)
        assert st.assert_same_test_trials(a, b) > 0
        assert np.array_equal(a.sort_values(["test_subject", "session", "trial"])["y_true"], b.sort_values(["test_subject", "session", "trial"])["y_true"])


def test_per_split_data_cannot_change_labels_or_trials(dataset, small_cfg):
    bp, csp = dataset
    bad = TrialSet(bp.X, bp.y ^ 1, bp.meta, bp.rejections, bp.sfreq, bp.ch_names)
    with pytest.raises(ValueError, match="identical labels"):
        run_protocol(small_cfg, bp, csp, "loso", models=("csp_lda",), log=lambda *_: None, per_split_data=lambda s: (bad, csp))
    _, _, ok = run_protocol(small_cfg, bp, csp, "loso", models=("csp_lda",), log=lambda *_: None, per_split_data=lambda s: (bp, csp))
    assert len(ok) == bp.n_trials


def test_target_adaptation_data_are_separate_from_evaluation_trials(dataset, small_cfg):
    """A fold-specific dataset may replace the held-out participant's epochs by corrected versions, but only THEIR rows."""
    bp, csp = dataset
    seen = []

    def per_split(test_subject):
        seen.append(test_subject)
        X = csp.X.copy()
        X[(csp.meta["subject"] == test_subject).to_numpy()] *= 1.0        # same rows only (identity here)
        return bp, TrialSet(X, csp.y, csp.meta, csp.rejections, csp.sfreq, csp.ch_names)

    run_protocol(small_cfg, bp, csp, "loso", models=("csp_lda",), log=lambda *_: None, per_split_data=per_split)
    assert sorted(seen) == [1, 2, 3]


# ------------------------------------------------------------------------------ baseline protection
def test_baseline_runs_can_never_be_overwritten(tmp_path, dataset, small_cfg):
    bp, csp = dataset
    res, spl, pred = run_protocol(small_cfg, bp, csp, "cross_session", models=("csp_lda",), log=lambda *_: None)
    for name in sorted(PROTECTED_RUNS):
        for ow in (False, True):
            with pytest.raises(FileExistsError, match="protected"):
                save_experiment(small_cfg, res, spl, pred, bp, name, "final", out_root=tmp_path, overwrite=ow)
    assert not any(tmp_path.iterdir())                                       # nothing was written
    save_experiment(small_cfg, res, spl, pred, bp, "new_run", "exploratory", out_root=tmp_path)
    with pytest.raises(FileExistsError, match="already holds"):
        save_experiment(small_cfg, res, spl, pred, bp, "new_run", "exploratory", out_root=tmp_path)
    save_experiment(small_cfg, res, spl, pred, bp, "new_run", "exploratory", out_root=tmp_path, overwrite=True)
    assert {"final_v1", "repro_v1", "repro_v2"} <= PROTECTED_RUNS


def test_extra_provenance_is_recorded(tmp_path, dataset, small_cfg):
    bp, csp = dataset
    res, spl, pred = run_protocol(small_cfg, bp, csp, "cross_session", models=("csp_lda",), log=lambda *_: None)
    out = save_experiment(small_cfg, res, spl, pred, bp, "prov", "exploratory", out_root=tmp_path, extra={"config_sha256": "abc", "seed": 42})
    import json

    s = json.loads((out / "summary.json").read_text())
    assert s["config_sha256"] == "abc" and s["seed"] == 42 and s["config"]["evaluation"]["seed"] == 42
