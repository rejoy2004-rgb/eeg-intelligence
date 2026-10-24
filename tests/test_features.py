"""Spectral / statistical feature tests (synthetic signals; see conftest.py warning)."""
import numpy as np
import pytest

from src.features.spectral import (FeatureError, band_power, band_power_from_config, flatten_band_major,
                                   log_band_power, welch_psd)
from src.features.statistical import hjorth, log_variance

FS = 250.0
BANDS = {"mu": [8.0, 12.0], "beta_low": [13.0, 20.0], "beta_high": [21.0, 30.0]}


def _epochs(n=6, c=4, t=500, seed=0):
    return np.random.default_rng(seed).normal(0, 5, (n, c, t))


def test_psd_shapes():
    f, p = welch_psd(_epochs(), FS, 250)
    assert p.shape == (6, 4, 126) and f[1] - f[0] == pytest.approx(1.0)


def test_band_power_shape_finite_and_band_major():
    bp = band_power(_epochs(), FS, BANDS)
    assert bp.shape == (6, 3, 4) and np.isfinite(bp).all() and (bp > 0).all()
    lbp = log_band_power(_epochs(), FS, BANDS)
    assert np.allclose(lbp, np.log(bp))
    flat = flatten_band_major(lbp)
    assert flat.shape == (6, 12)
    assert np.array_equal(flat[:, :4], lbp[:, 0, :])        # first band's channels come first


def test_band_power_picks_the_right_band_and_channel():
    t = np.arange(500) / FS
    x = np.zeros((2, 3, 500))
    x[:, 1, :] = 10 * np.sin(2 * np.pi * 10 * t)            # 10 Hz only on channel 1
    x += np.random.default_rng(1).normal(0, 0.1, x.shape)
    bp = band_power(x, FS, BANDS)
    assert bp[0, 0, 1] > 100 * bp[0, 1, 1]                  # mu >> beta_low on that channel
    assert bp[0, 0, 1] > 100 * bp[0, 0, 0]                  # channel 1 >> channel 0 in mu


def test_sinusoid_power_scales_with_amplitude():
    t = np.arange(500) / FS
    a = (np.sin(2 * np.pi * 10 * t))[None, None, :]
    p1 = band_power(a, FS, {"mu": [8, 12]})[0, 0, 0]
    p2 = band_power(2 * a, FS, {"mu": [8, 12]})[0, 0, 0]
    assert p2 / p1 == pytest.approx(4.0, rel=1e-6)


def test_invalid_inputs():
    with pytest.raises(FeatureError, match="3-D"):
        band_power(np.zeros((4, 500)), FS, BANDS)
    with pytest.raises(FeatureError, match="empty"):
        band_power(np.zeros((0, 4, 500)), FS, BANDS)
    with pytest.raises(FeatureError, match="short trial"):
        band_power(np.zeros((2, 4, 100)) + 1, FS, BANDS, nperseg=250)
    with pytest.raises(FeatureError, match="Invalid sampling"):
        band_power(_epochs(), 0.0, BANDS)
    with pytest.raises(FeatureError, match="Invalid sampling"):
        band_power(_epochs(), float("nan"), BANDS)
    with pytest.raises(FeatureError, match="invalid for Nyquist"):
        band_power(_epochs(), FS, {"x": [100.0, 130.0]})
    with pytest.raises(FeatureError, match="invalid for Nyquist"):
        band_power(_epochs(), FS, {"x": [12.0, 8.0]})
    with pytest.raises(FeatureError, match="At least one band"):
        band_power(_epochs(), FS, {})
    with pytest.raises(FeatureError, match="overlap"):
        welch_psd(_epochs(), FS, 250, overlap=1.0)


def test_non_finite_input_rejected():
    x = _epochs()
    x[0, 0, 0] = np.nan
    with pytest.raises(FeatureError, match="NaN"):
        band_power(x, FS, BANDS)
    x[0, 0, 0] = np.inf
    with pytest.raises(FeatureError, match="NaN"):
        log_band_power(x, FS, BANDS)


def test_band_without_frequency_bin_rejected():
    with pytest.raises(FeatureError, match="no frequency bin"):
        band_power(_epochs(), FS, {"narrow": [10.2, 10.8]}, nperseg=250)


def test_flat_channel_gives_finite_log_power():
    x = _epochs()
    x[:, 2, :] = 0.0
    assert np.isfinite(log_band_power(x, FS, BANDS)).all()


def test_missing_channel_changes_output_width_consistently():
    full = band_power(_epochs(c=4), FS, BANDS)
    sub = band_power(_epochs(c=4)[:, :3], FS, BANDS)
    assert sub.shape[2] == 3 and np.allclose(sub, full[:, :, :3])       # channels are processed independently


def test_features_from_config(cfg):
    out = band_power_from_config(_epochs(c=22), FS, cfg)
    assert out.shape == (6, 3, 22) and np.isfinite(out).all()


def test_flatten_requires_3d():
    with pytest.raises(FeatureError):
        flatten_band_major(np.zeros((3, 4)))


def test_deterministic_features():
    x = _epochs()
    assert np.array_equal(log_band_power(x, FS, BANDS), log_band_power(x, FS, BANDS))


def test_log_variance_known_value():
    x = _epochs()
    assert np.allclose(log_variance(2 * x) - log_variance(x), np.log(4.0))
    assert log_variance(x).shape == (6, 4)


def test_hjorth_shapes_and_known_sine():
    t = np.arange(500) / FS
    s = np.sin(2 * np.pi * 5 * t)[None, None, :]
    h = hjorth(s)
    assert h.shape == (1, 3, 1)
    assert h[0, 1, 0] == pytest.approx(2 * np.pi * 5 / FS, rel=0.02)    # mobility of a sine = omega
    assert h[0, 2, 0] == pytest.approx(1.0, abs=0.05)                   # complexity of a pure sine ~ 1
    with pytest.raises(FeatureError):
        hjorth(np.zeros((1, 1, 2)) + 1.0)
