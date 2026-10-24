"""Spatial EEG images - a modern re-implementation of the idea behind EEGLearn's ``gen_images``.

Idea (Bashivan et al., 2016)
    Take one feature value per electrode (e.g. mu-band power), place it at the electrode's 2-D
    scalp position, interpolate between electrodes on a regular pixel grid, and stack several
    feature types as image channels. A CNN can then exploit the *spatial* arrangement of the sensors.

What is the same as EEGLearn
    * 3-D electrode positions -> 2-D with an azimuthal equidistant projection (same maths).
    * ``scipy.interpolate.griddata`` with ``method='cubic'`` on the pixel grid.
    * Pixels outside the electrodes' convex hull are undefined (NaN in EEGLearn) and set to 0 after
      normalisation. Several feature types become several image channels (EEGLearn: theta/alpha/beta).

What is deliberately different (see docs/ORIGINAL_EEGLEARN_ANALYSIS.md)
    * Normalisation statistics are learned on TRAINING data only (``ImageScaler.fit``). EEGLearn
      normalised over all samples before splitting, which lets test data influence training inputs.
    * The grid is isotropic (same metres-per-pixel in x and y) so the head is not stretched.
    * The sphere centre is fitted before projecting, so the vertex of the projection is the
      centre of the electrode cloud rather than the origin of an arbitrary coordinate frame.
    * Interpolation is precomputed as a matrix, so converting thousands of trials is one matmul.
    * Electrode positions come from a *template* montage (not measured on these participants).

Array shapes (n trials, F feature channels, C electrodes, H = W = grid size)
    features : (n, F, C)        one value per electrode and feature type
    images   : (n, F, H, W)     row 0 = front of the head, column 0 = left side
    mask     : (H, W) bool      True where the pixel lies inside the electrode convex hull

Every in-hull pixel is *interpolated* (no pixel is a direct measurement); only the electrode
locations, shown separately, are measured positions of real sensors.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata


class SpatialError(ValueError):
    """Raised for invalid electrode coordinates or feature arrays."""


# ----------------------------------------------------------------------------- coordinates
def get_electrode_xyz(ch_names, montage: str = "colin27_1020") -> np.ndarray:
    """3-D electrode positions (metres, MNE head frame) for the given channel names.

    Raises if a channel is missing from the montage (we never invent positions).
    """
    import mne

    mont = mne.channels.make_standard_montage(montage)
    pos = mont.get_positions()["ch_pos"]
    missing = [c for c in ch_names if c not in pos]
    if missing:
        raise SpatialError(f"Channels not found in montage '{montage}': {missing}")
    return np.array([pos[c] for c in ch_names], dtype=float)


def validate_coordinates(xyz: np.ndarray, n_channels: int | None = None, min_separation: float = 1e-6) -> None:
    """Shape, finiteness, channel count and duplicate-location checks."""
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] not in (2, 3):
        raise SpatialError(f"Coordinates must have shape (n_channels, 2 or 3); got {xyz.shape}")
    if n_channels is not None and xyz.shape[0] != n_channels:
        raise SpatialError(f"{xyz.shape[0]} coordinates for {n_channels} channels")
    if xyz.shape[0] < 4:
        raise SpatialError("At least 4 electrodes are needed for 2-D interpolation")
    if not np.isfinite(xyz).all():
        raise SpatialError("Coordinates contain NaN or infinite values (missing electrode position?)")
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    d[np.diag_indices(len(d))] = np.inf
    if (d < min_separation).any():
        raise SpatialError("Two electrodes share (almost) the same location")


def fit_sphere_center(xyz: np.ndarray) -> np.ndarray:
    """Least-squares centre of the sphere closest to the 3-D points (linear formulation)."""
    A = np.hstack([2 * xyz, np.ones((len(xyz), 1))])
    b = (xyz**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:3]


def azimuthal_equidistant(xyz: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    """Project 3-D points to 2-D, preserving distance from the vertex (+z axis).

    theta = atan2(y, x)          (azimuth around the vertical axis)
    rho   = pi/2 - elevation     (angular distance from the vertex)
    x2d, y2d = rho cos(theta), rho sin(theta)

    The same formula as EEGLearn's ``azim_proj`` (cart2sph followed by pol2cart). In the MNE head
    frame +x points to the right ear and +y to the nose, so the nose ends up at the top of the image.
    """
    xyz = np.asarray(xyz, dtype=float)
    validate_coordinates(xyz)
    if xyz.shape[1] != 3:
        raise SpatialError("azimuthal_equidistant needs 3-D coordinates")
    c = fit_sphere_center(xyz) if center is None else np.asarray(center, dtype=float)
    p = xyz - c
    r_xy = np.hypot(p[:, 0], p[:, 1])
    elev = np.arctan2(p[:, 2], r_xy)
    rho = np.pi / 2 - elev
    theta = np.arctan2(p[:, 1], p[:, 0])
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


# ----------------------------------------------------------------------------- image builder
class SpatialImageBuilder:
    """Turns per-electrode feature values into interpolated 2-D images.

    The interpolation of one feature map is linear in the electrode values, so we compute once,
    for each electrode, the pixel pattern produced by a unit value at that electrode (``weights``).
    Then ``images = features @ weights.T``. For method='linear' this is exact; for 'cubic'
    (Clough-Tocher, gradients estimated iteratively with a finite tolerance) it matches a direct
    ``griddata`` call to ~1e-6 relative error (verified in tests/test_spatial_representation.py).
    """

    def __init__(self, xy: np.ndarray, grid_size: int = 16, method: str = "cubic"):
        xy = np.asarray(xy, dtype=float)
        validate_coordinates(xy)
        if xy.shape[1] != 2:
            raise SpatialError("SpatialImageBuilder needs 2-D electrode coordinates")
        if grid_size < 4:
            raise SpatialError("grid_size must be >= 4")
        if method not in ("cubic", "linear"):
            raise SpatialError("method must be 'cubic' or 'linear'")
        self.xy = xy
        self.grid_size = int(grid_size)
        self.method = method
        self.n_channels = len(xy)

        lo, hi = xy.min(axis=0), xy.max(axis=0)
        center, half = (lo + hi) / 2, float((hi - lo).max()) / 2
        gx = np.linspace(center[0] - half, center[0] + half, grid_size)          # left -> right
        gy = np.linspace(center[1] + half, center[1] - half, grid_size)          # front -> back
        self.grid_x, self.grid_y = np.meshgrid(gx, gy)                           # (H, W)

        weights = np.empty((grid_size * grid_size, self.n_channels))
        for j in range(self.n_channels):
            unit = np.zeros(self.n_channels)
            unit[j] = 1.0
            img = griddata(xy, unit, (self.grid_x, self.grid_y), method=method, fill_value=np.nan)
            weights[:, j] = img.ravel()
        self.mask = ~np.isnan(weights).any(axis=1).reshape(grid_size, grid_size)
        self.weights = np.nan_to_num(weights, nan=0.0)
        if not self.mask.any():
            raise SpatialError("No grid pixel lies inside the electrode hull")

    @property
    def electrode_pixels(self) -> np.ndarray:
        """Electrode positions in fractional pixel units (col, row) for plotting overlays."""
        gx, gy = self.grid_x[0], self.grid_y[:, 0]
        col = (self.xy[:, 0] - gx[0]) / (gx[1] - gx[0])
        row = (self.xy[:, 1] - gy[0]) / (gy[1] - gy[0])
        return np.column_stack([col, row])

    def transform(self, features: np.ndarray) -> np.ndarray:
        """(n, F, C) -> (n, F, H, W). Pixels outside the hull are NaN (see ``mask``)."""
        f = np.asarray(features)
        if f.ndim != 3:
            raise SpatialError(f"features must be (n, F, C); got shape {f.shape}")
        if f.shape[2] != self.n_channels:
            raise SpatialError(f"features have {f.shape[2]} electrodes but the layout has {self.n_channels}")
        if f.shape[0] == 0:
            raise SpatialError("features contain no trials")
        if not np.isfinite(f).all():
            raise SpatialError("features contain NaN or infinite values")
        n, F, _ = f.shape
        flat = f.reshape(n * F, self.n_channels) @ self.weights.T
        images = flat.reshape(n, F, self.grid_size, self.grid_size)
        images[:, :, ~self.mask] = np.nan
        return images


class ImageScaler:
    """Per-feature-channel standardisation fitted on TRAINING images only.

    mean/std are computed over all in-hull pixels of all training trials, separately for each
    feature channel (as EEGLearn did per colour, but without peeking at test data).
    Outside-hull pixels become 0 (= the training mean after scaling).
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, images: np.ndarray, mask: np.ndarray) -> "ImageScaler":
        if images.ndim != 4 or images.shape[2:] != mask.shape:
            raise SpatialError(f"images {images.shape} do not match mask {mask.shape}")
        vals = images[:, :, mask]                         # (n, F, n_pixels_in_hull)
        if not np.isfinite(vals).all():
            raise SpatialError("Training images contain non-finite values inside the hull")
        self.mean_ = vals.mean(axis=(0, 2))
        self.std_ = vals.std(axis=(0, 2))
        if (self.std_ < 1e-12).any():
            raise SpatialError("A feature channel is constant in the training data; cannot standardise")
        self.mask_ = mask.copy()
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise SpatialError("ImageScaler must be fitted before use")
        if images.ndim != 4 or images.shape[1] != len(self.mean_):
            raise SpatialError(f"Expected images with {len(self.mean_)} feature channels; got {images.shape}")
        out = (images - self.mean_[None, :, None, None]) / self.std_[None, :, None, None]
        out[:, :, ~self.mask_] = 0.0
        return out.astype(np.float32)

    def fit_transform(self, images, mask):
        return self.fit(images, mask).transform(images)

    def state(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist(), "mask": self.mask_.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "ImageScaler":
        s = cls()
        s.mean_, s.std_ = np.array(state["mean"]), np.array(state["std"])
        s.mask_ = np.array(state["mask"], dtype=bool)
        return s


def builder_from_config(ch_names, cfg) -> SpatialImageBuilder:
    """Electrode names -> template 3-D positions -> projected 2-D -> image builder."""
    sp = cfg.spatial
    xyz = get_electrode_xyz(ch_names, sp["montage"])
    validate_coordinates(xyz, len(ch_names))
    return SpatialImageBuilder(azimuthal_equidistant(xyz), sp["grid_size"], sp["interpolation"])


def images_from_epochs(X: np.ndarray, sfreq: float, cfg, builder: SpatialImageBuilder) -> np.ndarray:
    """Epochs (n, C, T) -> log band power (n, B, C) -> unscaled interpolated images (n, B, H, W)."""
    from src.features.spectral import band_power_from_config

    return builder.transform(band_power_from_config(X, sfreq, cfg))


# ----------------------------------------------------------------------------- measured-only layout grid
# Row/column of each electrode NUMBER (1-22, = position in the GDF file) read from Figure 3 (left) of the official
# data-set description: rows hold electrodes {1}, {2-6}, {7-13}, {14-18}, {19-21}, {22}; columns are aligned so that
# 1, 4, 10, 16, 20, 22 sit in the middle column. This encodes the figure's TOPOLOGY only (no distances, no template).
OFFICIAL_FIGURE_LAYOUT = {1: (0, 3)}
OFFICIAL_FIGURE_LAYOUT.update({n: (1, 1 + n - 2) for n in range(2, 7)})
OFFICIAL_FIGURE_LAYOUT.update({n: (2, n - 7) for n in range(7, 14)})
OFFICIAL_FIGURE_LAYOUT.update({n: (3, 1 + n - 14) for n in range(14, 19)})
OFFICIAL_FIGURE_LAYOUT.update({n: (4, 2 + n - 19) for n in range(19, 22)})
OFFICIAL_FIGURE_LAYOUT[22] = (5, 3)


class LayoutGridBuilder:
    """Places each electrode's feature in ONE pixel of a small grid following the official montage figure.

    No interpolation: every non-empty pixel is a real electrode measurement, every other pixel is empty (0 after
    scaling). The 6 x 7 layout is zero-padded into a square ``size`` x ``size`` grid (rows shifted down by 1) so the
    same CNN class can be used. Same interface as ``SpatialImageBuilder`` (``transform``, ``mask``).
    """

    def __init__(self, n_channels: int = 22, size: int = 8):
        if n_channels != 22 or size < 7:
            raise SpatialError("LayoutGridBuilder supports the 22-channel montage and size >= 7")
        self.n_channels, self.grid_size = n_channels, size
        self.rows = np.array([OFFICIAL_FIGURE_LAYOUT[i + 1][0] + 1 for i in range(22)])
        self.cols = np.array([OFFICIAL_FIGURE_LAYOUT[i + 1][1] for i in range(22)])
        self.mask = np.zeros((size, size), dtype=bool)
        self.mask[self.rows, self.cols] = True
        if self.mask.sum() != 22:
            raise SpatialError("Two electrodes map to the same pixel")

    def transform(self, features: np.ndarray) -> np.ndarray:
        f = np.asarray(features)
        if f.ndim != 3 or f.shape[2] != self.n_channels or f.shape[0] == 0:
            raise SpatialError(f"features must be (n, F, 22); got {f.shape}")
        if not np.isfinite(f).all():
            raise SpatialError("features contain NaN or infinite values")
        out = np.full((f.shape[0], f.shape[1], self.grid_size, self.grid_size), np.nan)
        out[:, :, self.rows, self.cols] = f
        return out
