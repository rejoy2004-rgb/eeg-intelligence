"""Preprocessing tests (synthetic signals; see conftest.py warning)."""
import numpy as np
import pytest

from src.signal.preprocessing import (PreprocessingError, bandpass_filter, finite_segments, preprocess_recording,
                                      rereference, validate_band)

FS = 250.0


def _sines(freqs, n=2500):
    t = np.arange(n) / FS
    return np.vstack([np.sin(2 * np.pi * f * t) for f in freqs])


def test_output_shape_and_channel_count_preserved():
    x = np.random.default_rng(0).normal(size=(5, 2000))
    y = bandpass_filter(x, FS, 4, 40)
    assert y.shape == x.shape and np.isfinite(y).all()


def test_filters_along_time_axis_not_channel_axis():
    # channel 0: 10 Hz (pass), channel 1: 1 Hz (stop), channel 2: 80 Hz (stop)
    x = _sines([10, 1, 80])
    y = bandpass_filter(x, FS, 4, 40)
    mid = slice(500, 2000)
    rms = lambda a: np.sqrt(np.mean(a[mid] ** 2))
    assert rms(y[0]) > 0.9 * rms(x[0])
    assert rms(y[1]) < 0.05
    assert rms(y[2]) < 0.05


def test_zero_phase():
    x = _sines([10])
    y = bandpass_filter(x, FS, 4, 40)
    assert np.corrcoef(x[0, 500:2000], y[0, 500:2000])[0, 1] > 0.999        # no time shift


def test_deterministic():
    x = np.random.default_rng(1).normal(size=(3, 1500))
    assert np.array_equal(bandpass_filter(x, FS, 4, 40), bandpass_filter(x, FS, 4, 40))


@pytest.mark.parametrize("low,high", [(0, 40), (-1, 40), (10, 10), (20, 10), (4, 125), (4, 200)])
def test_invalid_band_rejected(low, high):
    with pytest.raises(PreprocessingError):
        validate_band(low, high, FS)


def test_invalid_sampling_frequency():
    with pytest.raises(PreprocessingError):
        validate_band(4, 40, 0)
    with pytest.raises(PreprocessingError):
        validate_band(4, 40, float("nan"))
    with pytest.raises(PreprocessingError, match="Nyquist"):
        bandpass_filter(np.zeros((1, 500)), 60.0, 4, 40)       # 40 Hz >= Nyquist of 30 Hz


def test_short_and_empty_signals():
    with pytest.raises(PreprocessingError, match="too short"):
        bandpass_filter(np.zeros((2, 10)), FS, 4, 40)
    with pytest.raises(PreprocessingError, match="empty"):
        bandpass_filter(np.zeros((2, 0)), FS, 4, 40)
    with pytest.raises(PreprocessingError, match="2-D"):
        bandpass_filter(np.zeros(100), FS, 4, 40)
    with pytest.raises(PreprocessingError):
        bandpass_filter(np.zeros((2, 500)), FS, 4, 40, order=0)


def test_gaps_are_preserved_and_segments_filtered_separately():
    x = np.random.default_rng(2).normal(size=(3, 3000))
    x[:, 1000:1100] = np.nan
    y = bandpass_filter(x, FS, 4, 40)
    assert np.isnan(y[:, 1000:1100]).all()
    assert np.isfinite(np.delete(y, np.s_[1000:1100], axis=1)).all()
    assert finite_segments(x) == [(0, 1000), (1100, 3000)]
    # the second segment must be independent of data before the gap
    x2 = x.copy()
    x2[:, :1000] += 50.0
    y2 = bandpass_filter(x2, FS, 4, 40)
    assert np.allclose(y[:, 1100:], y2[:, 1100:])


def test_nan_inside_one_channel_or_inf_rejected():
    x = np.random.default_rng(3).normal(size=(3, 1500))
    x[1, 700] = np.nan
    with pytest.raises(PreprocessingError, match="Non-finite"):
        bandpass_filter(x, FS, 4, 40)
    x = np.random.default_rng(3).normal(size=(3, 1500))
    x[0, 5] = np.inf
    with pytest.raises(PreprocessingError, match="Non-finite"):
        bandpass_filter(x, FS, 4, 40)


def test_all_nan_rejected():
    with pytest.raises(PreprocessingError):
        bandpass_filter(np.full((2, 500), np.nan), FS, 4, 40)


def test_rereference():
    x = np.random.default_rng(4).normal(size=(4, 100))
    assert rereference(x, "none") is x
    avg = rereference(x, "average")
    assert np.allclose(avg.sum(axis=0), 0)
    with pytest.raises(PreprocessingError):
        rereference(x, "laplacian")


def test_preprocess_recording_uses_config_and_keeps_gap(synth, cfg):
    rec, _ = synth
    out = preprocess_recording(rec, cfg)
    assert out.shape == rec.data.shape
    a, b = rec.missing_segments[0]
    assert np.isnan(out[:, a:b]).all()
    out2 = preprocess_recording(rec, cfg, band_hz=(8.0, 30.0))
    assert not np.allclose(np.nan_to_num(out), np.nan_to_num(out2))
