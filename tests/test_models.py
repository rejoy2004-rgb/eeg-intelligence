"""Model tests: baselines, CNN architecture, small training run, save/load (synthetic data)."""
import json

import numpy as np
import pytest
import torch

from src.models.baseline import (BandPowerFeatures, build_csp_lda_pipeline, build_logreg_pipeline, decision_scores,
                                 fit_csp_lda, fit_logreg)
from src.models.eeg_cnn import (EEGSpatialCNN, ModelInputError, load_cnn, predict_scores, save_cnn, train_cnn)
from src.signal.epoching import build_dual_band_trials


@pytest.fixture
def trials(synth, cfg):
    rec, _ = synth
    return build_dual_band_trials(rec, cfg)


# ---- baselines ------------------------------------------------------------------------
def test_bandpower_transformer_shape(trials):
    bp, _ = trials
    f = BandPowerFeatures(250.0, {"mu": [8, 12], "b": [13, 30]}).fit_transform(bp.X)
    assert f.shape == (bp.n_trials, 2 * 22)


def test_logreg_pipeline_fits_and_predicts(trials, cfg):
    bp, _ = trials
    model, info = fit_logreg(cfg, bp.sfreq, bp.X, bp.y, bp.meta["run"].to_numpy())
    assert info["best_C"] in cfg.models["logreg"]["c_grid"]
    assert info["n_groups"] == 2
    s = decision_scores(model, bp.X)
    assert s.shape == (bp.n_trials,) and np.isfinite(s).all()


def test_logreg_needs_two_groups(trials, cfg):
    bp, _ = trials
    with pytest.raises(ValueError, match="at least 2 groups"):
        fit_logreg(cfg, bp.sfreq, bp.X, bp.y, np.zeros(bp.n_trials))


def test_csp_lda_pipeline(trials, cfg):
    _, csp = trials
    m = fit_csp_lda(cfg, csp.X, csp.y)
    s = decision_scores(m, csp.X)
    assert s.shape == (csp.n_trials,)
    assert ((s > 0).astype(int) == csp.y).mean() > 0.8        # planted synthetic effect, training data


def test_pipelines_are_unfitted_until_fit(cfg):
    p = build_logreg_pipeline(cfg, 250.0)
    with pytest.raises(Exception):
        p.predict(np.zeros((2, 22, 500)))
    assert build_csp_lda_pipeline(cfg) is not None


def test_scaler_inside_pipeline_is_fitted_on_training_data_only(trials, cfg):
    bp, _ = trials
    pipe = build_logreg_pipeline(cfg, 250.0).fit(bp.X[:10], bp.y[:10])
    feats_train = pipe.named_steps["features"].transform(bp.X[:10])
    assert np.allclose(pipe.named_steps["scale"].mean_, feats_train.mean(0))


# ---- CNN ---------------------------------------------------------------------------------
def test_cnn_initialisation_and_shapes():
    net = EEGSpatialCNN(in_channels=3, grid_size=16)
    out = net(torch.zeros(5, 3, 16, 16))
    assert out.shape == (5, 2)
    assert sum(p.numel() for p in net.parameters()) < 200_000          # compact
    assert EEGSpatialCNN(in_channels=1, grid_size=8, channels=(8, 8))(torch.zeros(2, 1, 8, 8)).shape == (2, 2)


def test_cnn_rejects_invalid_architecture():
    with pytest.raises(ModelInputError):
        EEGSpatialCNN(in_channels=0)
    with pytest.raises(ModelInputError):
        EEGSpatialCNN(grid_size=2)


def _toy_images(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    X = rng.normal(size=(n, 3, 16, 16)).astype(np.float32)
    X[y == 1, 0, :8, :] += 1.5               # simple planted spatial pattern (synthetic)
    return X, y


def test_small_valid_training_run(cfg):
    cfg["models"]["cnn"].update(max_epochs=6, batch_size=16, patience=6)
    X, y = _toy_images()
    Xv, yv = _toy_images(30, seed=1)
    net, hist = train_cnn(X, y, Xv, yv, cfg, seed=0)
    assert 1 <= hist["best_epoch"] <= 6 and len(hist["train_loss"]) == len(hist["val_loss"])
    assert hist["best_val_balanced_accuracy"] > 0.7                    # learnable synthetic pattern
    assert np.isfinite(hist["train_loss"]).all()


def test_training_is_reproducible_with_same_seed(cfg):
    cfg["models"]["cnn"].update(max_epochs=3, batch_size=16, patience=3)
    X, y = _toy_images()
    Xv, yv = _toy_images(30, seed=1)
    a, _ = train_cnn(X, y, Xv, yv, cfg, seed=5)
    b, _ = train_cnn(X, y, Xv, yv, cfg, seed=5)
    assert np.allclose(predict_scores(a, Xv), predict_scores(b, Xv), atol=1e-6)


def test_odd_batch_sizes_do_not_crash(cfg):
    cfg["models"]["cnn"].update(max_epochs=2, batch_size=32, patience=2)
    X, y = _toy_images(33 if False else 34)          # last batch holds 2 samples with batch 16 below
    cfg["models"]["cnn"]["batch_size"] = 16
    net, _ = train_cnn(X, y, X[:6], y[:6], cfg)
    assert predict_scores(net, X[:1]).shape == (1,)   # single-sample inference works in eval mode


@pytest.mark.parametrize("bad", ["labels", "nan", "shape", "class", "empty", "channels"])
def test_training_rejects_invalid_inputs(cfg, bad):
    cfg["models"]["cnn"].update(max_epochs=1)
    X, y = _toy_images()
    Xv, yv = _toy_images(10, 1)
    if bad == "labels":
        y = y + 1
    elif bad == "nan":
        X = X.copy(); X[0, 0, 0, 0] = np.nan
    elif bad == "shape":
        X = X[:, 0]
    elif bad == "class":
        y = np.zeros_like(y)
    elif bad == "empty":
        X, y = X[:0], y[:0]
    elif bad == "channels":
        Xv = Xv[:, :2]
    with pytest.raises(ModelInputError):
        train_cnn(X, y, Xv, yv, cfg)


def test_predict_input_validation():
    net = EEGSpatialCNN()
    with pytest.raises(ModelInputError):
        predict_scores(net, np.zeros((3, 16, 16), dtype=np.float32))
    with pytest.raises(ModelInputError):
        predict_scores(net, np.zeros((2, 2, 16, 16), dtype=np.float32))
    bad = np.zeros((2, 3, 16, 16), dtype=np.float32)
    bad[0, 0, 0, 0] = np.inf
    with pytest.raises(ModelInputError):
        predict_scores(net, bad)


def test_cpu_inference_is_deterministic_in_eval_mode():
    net = EEGSpatialCNN()
    X = np.random.default_rng(0).normal(size=(4, 3, 16, 16)).astype(np.float32)
    assert all(p.device.type == "cpu" for p in net.parameters())
    assert np.array_equal(predict_scores(net, X), predict_scores(net, X))


def test_save_and_load_roundtrip(tmp_path):
    net = EEGSpatialCNN(channels=(8, 16, 16))
    X = np.random.default_rng(0).normal(size=(4, 3, 16, 16)).astype(np.float32)
    save_cnn(tmp_path, net, {"note": "test"})
    loaded, meta = load_cnn(tmp_path)
    assert meta["note"] == "test" and meta["architecture"]["channels"] == [8, 16, 16]
    assert np.allclose(predict_scores(net, X), predict_scores(loaded, X), atol=1e-6)
    assert json.loads((tmp_path / "cnn_meta.json").read_text())["architecture"]["in_channels"] == 3


def test_load_missing_or_tampered(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cnn(tmp_path)
    net = EEGSpatialCNN()
    save_cnn(tmp_path, net, {})
    (tmp_path / "cnn.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(Exception):
        load_cnn(tmp_path)
