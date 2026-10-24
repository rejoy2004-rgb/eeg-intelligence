"""Tests for the EEGLearn-inspired spatial representation (synthetic / hand-checkable inputs)."""
import numpy as np
import pytest
from scipy.interpolate import griddata

from src.features.spatial import (ImageScaler, SpatialError, SpatialImageBuilder, azimuthal_equidistant,
                                  builder_from_config, get_electrode_xyz, images_from_epochs,
                                  validate_coordinates)

# five electrodes on a plane: four corners + centre (hand-checkable)
XY = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])


# ---- electrode locations -------------------------------------------------------------
def test_real_montage_positions_for_dataset_channels(cfg):
    xyz = get_electrode_xyz(cfg.dataset["eeg_channels"], cfg.spatial["montage"])
    assert xyz.shape == (22, 3) and np.isfinite(xyz).all()
    validate_coordinates(xyz, 22)


def test_unknown_channel_is_an_error_not_an_invented_position():
    with pytest.raises(SpatialError, match="not found"):
        get_electrode_xyz(["Cz", "NotAnElectrode"])


def test_coordinate_validation():
    with pytest.raises(SpatialError, match="shape"):
        validate_coordinates(np.zeros((5, 4)))
    with pytest.raises(SpatialError, match="shape"):
        validate_coordinates(np.zeros(5))
    with pytest.raises(SpatialError, match="coordinates for 22"):
        validate_coordinates(XY, 22)
    with pytest.raises(SpatialError, match="At least 4"):
        validate_coordinates(XY[:3])
    bad = XY.copy()
    bad[1, 0] = np.nan
    with pytest.raises(SpatialError, match="NaN"):
        validate_coordinates(bad)
    bad = XY.copy()
    bad[1] = np.inf
    with pytest.raises(SpatialError):
        validate_coordinates(bad)
    dup = XY.copy()
    dup[1] = dup[0]
    with pytest.raises(SpatialError, match="same"):
        validate_coordinates(dup)


def test_projection_geometry_on_a_sphere():
    # vertex -> origin; equator points -> distance pi/2; four equator points are 90 degrees apart
    pts = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0.6, 0.0, 0.8]])
    xy = azimuthal_equidistant(pts, center=np.zeros(3))
    assert np.allclose(xy[0], 0, atol=1e-12)
    assert np.allclose(np.linalg.norm(xy[1:5], axis=1), np.pi / 2)
    assert np.allclose(xy[1], [np.pi / 2, 0]) and np.allclose(xy[2], [0, np.pi / 2])
    assert np.linalg.norm(xy[5]) == pytest.approx(np.arccos(0.8))       # angular distance from the vertex


def test_projection_matches_neuroscience_expectations(cfg):
    names = cfg.dataset["eeg_channels"]
    xy = azimuthal_equidistant(get_electrode_xyz(names, cfg.spatial["montage"]))
    p = {n: xy[i] for i, n in enumerate(names)}
    assert p["C3"][0] < p["Cz"][0] < p["C4"][0]          # left -> right
    assert p["Fz"][1] > p["Cz"][1] > p["Pz"][1]          # front -> back
    assert np.linalg.norm(p["Cz"]) < 0.25 * np.linalg.norm(p["C4"] - p["C3"]) + 0.2


def test_projection_requires_3d():
    with pytest.raises(SpatialError):
        azimuthal_equidistant(XY)


# ---- image builder ---------------------------------------------------------------------
def test_output_shape_and_mask():
    b = SpatialImageBuilder(XY, grid_size=9, method="linear")
    img = b.transform(np.ones((7, 3, 5)))
    assert img.shape == (7, 3, 9, 9)
    assert b.mask.shape == (9, 9) and b.mask.all()                      # square hull fills the square grid
    assert np.isfinite(img).all()


def test_pixels_outside_electrode_hull_are_nan():
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.1, 0.1]])
    b = SpatialImageBuilder(tri, grid_size=8, method="linear")
    img = b.transform(np.ones((1, 1, 4)))
    assert np.isnan(img[0, 0][~b.mask]).all() and np.isfinite(img[0, 0][b.mask]).all()
    assert 0 < (~b.mask).sum() < 64


def test_constant_input_gives_constant_image_inside_hull():
    for method in ("linear", "cubic"):
        b = SpatialImageBuilder(XY, grid_size=10, method=method)
        img = b.transform(np.full((2, 1, 5), 3.7))
        assert np.allclose(img[:, :, b.mask], 3.7, atol=1e-6)


def test_linear_field_is_reproduced_and_channels_map_to_correct_positions():
    """If every electrode carries its own x coordinate, the image must equal the pixel x coordinate."""
    b = SpatialImageBuilder(XY, grid_size=11, method="linear")
    img = b.transform(XY[:, 0][None, None, :])[0, 0]
    assert np.allclose(img[b.mask], b.grid_x[b.mask], atol=1e-9)
    img_y = b.transform(XY[:, 1][None, None, :])[0, 0]
    assert np.allclose(img_y[b.mask], b.grid_y[b.mask], atol=1e-9)
    # permuting the channels must break the correspondence (guards against silent channel mix-ups)
    perm = b.transform(XY[::-1, 0][None, None, :])[0, 0]
    assert not np.allclose(perm[b.mask], b.grid_x[b.mask], atol=1e-3)


def test_cubic_reproduces_linear_field_closely():
    b = SpatialImageBuilder(XY, grid_size=11, method="cubic")
    img = b.transform(XY[:, 0][None, None, :])[0, 0]
    assert np.allclose(img[b.mask], b.grid_x[b.mask], atol=1e-3)


def test_matrix_implementation_matches_direct_griddata(cfg):
    names = cfg.dataset["eeg_channels"]
    b = builder_from_config(names, cfg)
    vals = np.random.default_rng(0).normal(size=(3, 1, 22))
    ours = b.transform(vals)
    for i in range(3):
        direct = griddata(b.xy, vals[i, 0], (b.grid_x, b.grid_y), method="cubic", fill_value=np.nan)
        both = b.mask & np.isfinite(direct)
        assert both.sum() == b.mask.sum()
        assert np.allclose(ours[i, 0][b.mask], direct[b.mask], atol=1e-4)


def test_orientation_front_is_top_left_is_left(cfg):
    names = cfg.dataset["eeg_channels"]
    b = builder_from_config(names, cfg)
    px = b.electrode_pixels
    col = {n: px[i, 0] for i, n in enumerate(names)}
    row = {n: px[i, 1] for i, n in enumerate(names)}
    assert row["Fz"] < row["Pz"]            # row 0 = front
    assert col["C3"] < col["C4"]            # column 0 = left


def test_deterministic_output(cfg):
    b = builder_from_config(cfg.dataset["eeg_channels"], cfg)
    x = np.random.default_rng(3).normal(size=(4, 3, 22))
    assert np.array_equal(b.transform(x), b.transform(x), equal_nan=True)


def test_transform_rejects_bad_features():
    b = SpatialImageBuilder(XY, grid_size=8, method="linear")
    with pytest.raises(SpatialError, match="\\(n, F, C\\)"):
        b.transform(np.ones((3, 5)))
    with pytest.raises(SpatialError, match="electrodes"):
        b.transform(np.ones((2, 1, 4)))
    with pytest.raises(SpatialError, match="no trials"):
        b.transform(np.ones((0, 1, 5)))
    bad = np.ones((2, 1, 5))
    bad[0, 0, 2] = np.nan
    with pytest.raises(SpatialError, match="NaN"):
        b.transform(bad)
    bad[0, 0, 2] = np.inf
    with pytest.raises(SpatialError):
        b.transform(bad)


def test_builder_argument_validation():
    with pytest.raises(SpatialError):
        SpatialImageBuilder(XY, grid_size=2)
    with pytest.raises(SpatialError):
        SpatialImageBuilder(XY, method="nearest")
    with pytest.raises(SpatialError):
        SpatialImageBuilder(np.zeros((5, 3)) + np.arange(15).reshape(5, 3))     # 3-D coordinates not allowed


def test_images_from_epochs(cfg):
    b = builder_from_config(cfg.dataset["eeg_channels"], cfg)
    x = np.random.default_rng(0).normal(0, 5, (4, 22, 500))
    img = images_from_epochs(x, 250.0, cfg, b)
    assert img.shape == (4, 3, 16, 16)
    assert np.isfinite(img[:, :, b.mask]).all()


# ---- scaler (leakage guard) -----------------------------------------------------------------
def test_scaler_uses_training_statistics_only():
    b = SpatialImageBuilder(XY, grid_size=8, method="linear")
    rng = np.random.default_rng(0)
    train = b.transform(rng.normal(0, 1, (30, 2, 5)))
    test = b.transform(rng.normal(5, 1, (10, 2, 5)))             # deliberately shifted distribution
    sc = ImageScaler().fit(train, b.mask)
    z_train, z_test = sc.transform(train), sc.transform(test)
    assert z_train[:, :, b.mask].mean() == pytest.approx(0.0, abs=1e-5)
    assert z_test[:, :, b.mask].mean() > 3.0                      # NOT re-centred on test data
    assert z_train.dtype == np.float32


def test_scaler_outside_hull_is_zero_and_state_roundtrips():
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.1, 0.1]])
    b = SpatialImageBuilder(tri, grid_size=8, method="linear")
    imgs = b.transform(np.random.default_rng(1).normal(size=(6, 1, 4)))
    sc = ImageScaler().fit(imgs, b.mask)
    z = sc.transform(imgs)
    assert (z[:, :, ~b.mask] == 0).all()
    sc2 = ImageScaler.from_state(sc.state())
    assert np.allclose(sc2.transform(imgs), z)


def test_scaler_errors():
    b = SpatialImageBuilder(XY, grid_size=8, method="linear")
    imgs = b.transform(np.ones((3, 1, 5)))
    with pytest.raises(SpatialError, match="constant"):
        ImageScaler().fit(imgs, b.mask)
    with pytest.raises(SpatialError, match="fitted"):
        ImageScaler().transform(imgs)
    sc = ImageScaler().fit(b.transform(np.random.default_rng(0).normal(size=(3, 1, 5))), b.mask)
    with pytest.raises(SpatialError, match="feature channels"):
        sc.transform(np.zeros((2, 2, 8, 8)))
    with pytest.raises(SpatialError):
        ImageScaler().fit(np.zeros((2, 8, 8)), b.mask)
