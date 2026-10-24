"""Signal-exploration plots (Matplotlib). Every function returns a Figure; none calls plt.show().

What these plots can show: amplitude, drift, gaps, obvious artifacts, class balance.
What they cannot show: whether a person is performing motor imagery, their mental state, or any
clinical condition. Do not read cognition into raw traces.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset import Recording


def _check_channels(rec_or_names, channels):
    names = rec_or_names.ch_names
    bad = [c for c in channels if c not in names]
    if bad:
        raise ValueError(f"Unknown channels {bad}; available: {names}")
    return [names.index(c) for c in channels]


def plot_traces(rec: Recording, channels, t_start_s: float, t_stop_s: float, data: np.ndarray | None = None,
                title: str | None = None):
    """Stacked time series of selected channels for [t_start_s, t_stop_s). ``data`` overrides rec.data."""
    if t_stop_s <= t_start_s:
        raise ValueError("t_stop_s must be greater than t_start_s")
    x = rec.data if data is None else data
    idx = _check_channels(rec, channels)
    a, b = int(t_start_s * rec.sfreq), min(int(t_stop_s * rec.sfreq), x.shape[1])
    if a >= b:
        raise ValueError("Requested interval lies outside the recording")
    t = np.arange(a, b) / rec.sfreq
    seg = x[idx, a:b]
    spread = np.nanpercentile(np.abs(seg), 99) * 2 or 1.0
    fig, ax = plt.subplots(figsize=(10, 0.8 + 0.7 * len(idx)))
    for k, (name, row) in enumerate(zip(channels, seg)):
        ax.plot(t, row - k * spread, lw=0.7, label=name)
    ax.set_yticks([-k * spread for k in range(len(idx))])
    ax.set_yticklabels(channels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel (offset traces, µV)")
    ax.set_title(title or f"EEG traces, {t_start_s:.0f}-{t_stop_s:.0f} s (gaps shown as breaks)")
    ax.text(0.99, 0.01, f"spacing = {spread:.0f} µV", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def plot_amplitude_distribution(rec: Recording, channels, bins: int = 100):
    """Histogram of sample amplitudes per channel (finite samples only)."""
    idx = _check_channels(rec, channels)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, i in zip(channels, idx):
        v = rec.data[i][np.isfinite(rec.data[i])]
        ax.hist(v, bins=bins, histtype="step", density=True, label=name)
    ax.set_xlabel("Amplitude (µV)")
    ax.set_ylabel("Probability density")
    ax.set_title("Signal amplitude distribution (finite samples)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_class_counts(meta: pd.DataFrame, y: np.ndarray, rejections: pd.DataFrame | None = None):
    """Kept trials per class (and per session), plus rejected trials by reason."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1, 1.3]})
    names = {0: "left hand", 1: "right hand"}
    sessions = sorted(meta["session"].unique())
    width = 0.8 / max(len(sessions), 1)
    for j, s in enumerate(sessions):
        counts = [int(((y == c) & (meta["session"].to_numpy() == s)).sum()) for c in (0, 1)]
        axes[0].bar(np.arange(2) + j * width, counts, width, label=f"session {s}")
    axes[0].set_xticks(np.arange(2) + width * (len(sessions) - 1) / 2)
    axes[0].set_xticklabels([names[0], names[1]])
    axes[0].set_ylabel("Trials kept")
    axes[0].set_title("Trials per class")
    axes[0].legend(fontsize=8)
    if rejections is not None and len(rejections):
        counts = rejections["reason"].value_counts()
        axes[1].barh(range(len(counts)), counts.values)
        axes[1].set_yticks(range(len(counts)))
        axes[1].set_yticklabels([r[:42] for r in counts.index], fontsize=7)
        axes[1].set_xlabel("Number of rejected trials")
    else:
        axes[1].text(0.5, 0.5, "No rejected trials", ha="center", va="center")
        axes[1].set_axis_off()
    axes[1].set_title("Rejections by reason")
    fig.tight_layout()
    return fig


def plot_quality(quality: pd.DataFrame):
    """Per-channel RMS and max |amplitude| (bars) with flat channels highlighted."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(quality))
    colors = ["tab:red" if f else "tab:blue" for f in quality["flat"]]
    ax.bar(x, quality["rms_uV"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(quality["channel"], rotation=60, fontsize=8)
    ax.set_ylabel("RMS amplitude (µV)")
    ax.set_title("Per-channel RMS (red = flat channel)")
    fig.tight_layout()
    return fig
