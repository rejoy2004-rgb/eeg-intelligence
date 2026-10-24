"""Electrode layout and spatial feature maps.

Reading these figures correctly
    * Electrode markers are template positions (colin27 10-20 montage), not measured on the participant.
    * Coloured pixels are INTERPOLATED between electrodes (cubic). Only the electrode markers mark
      places where a sensor actually measured something. Pixels outside the electrode hull are blank.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.features.spatial import SpatialImageBuilder


def plot_electrode_layout(xy: np.ndarray, ch_names, builder: SpatialImageBuilder | None = None):
    """Projected 2-D electrode positions (nose at top), optionally with the interpolation hull."""
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    if builder is not None:
        ax.imshow(builder.mask, cmap="Greys", alpha=0.15,
                  extent=(builder.grid_x[0, 0], builder.grid_x[0, -1], builder.grid_y[-1, 0], builder.grid_y[0, 0]))
    ax.scatter(xy[:, 0], xy[:, 1], c="tab:blue", s=30, zorder=3)
    for (x, y), n in zip(xy, ch_names):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("Projected x (rad from vertex; left → right)")
    ax.set_ylabel("Projected y (rad; back → front)")
    ax.set_title("Electrode layout (azimuthal projection)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_spatial_map(image: np.ndarray, builder: SpatialImageBuilder, title: str, cbar_label: str = "z-score",
                     show_electrodes: bool = True, ax=None):
    """One interpolated feature map (H, W). Pixels outside the hull are masked (white)."""
    img = np.where(builder.mask, image, np.nan)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4))
    else:
        fig = ax.figure
    lim = np.nanmax(np.abs(img)) or 1.0
    im = ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim)
    if show_electrodes:
        e = builder.electrode_pixels
        ax.scatter(e[:, 0], e[:, 1], c="k", s=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)
    return fig


def plot_class_difference_maps(images: np.ndarray, y: np.ndarray, builder: SpatialImageBuilder, feature_names):
    """Mean(right) - mean(left) for each feature channel. Descriptive only - not a significance test."""
    if len(images) == 0 or not (y == 0).any() or not (y == 1).any():
        raise ValueError("Need trials from both classes")
    diff = images[y == 1].mean(0) - images[y == 0].mean(0)
    fig, axes = plt.subplots(1, len(feature_names), figsize=(3.6 * len(feature_names), 3.6))
    axes = np.atleast_1d(axes)
    for ax, d, name in zip(axes, diff, feature_names):
        plot_spatial_map(d, builder, f"{name}: right − left", "mean difference (z)", ax=ax)
    fig.tight_layout()
    return fig
