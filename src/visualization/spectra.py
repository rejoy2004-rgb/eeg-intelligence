"""Power-spectral-density plots.

A PSD peak (for example near 10 Hz) describes the signal's frequency content. It is NOT proof of a
particular mental state, and absolute power differs strongly between people and recording sessions.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.features.spectral import welch_psd


def plot_psd_by_class(X: np.ndarray, y: np.ndarray, sfreq: float, ch_names, channel: str,
                      bands: dict | None = None, nperseg: int = 250, fmax: float = 45.0):
    """Mean PSD (with a +-1 standard-error band) of one channel for left vs right trials."""
    if channel not in ch_names:
        raise ValueError(f"Unknown channel '{channel}'")
    ci = list(ch_names).index(channel)
    freqs, psd = welch_psd(X, sfreq, nperseg)
    sel = freqs <= fmax
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for cls, label in ((0, "left hand"), (1, "right hand")):
        p = psd[y == cls][:, ci][:, sel]
        if len(p) == 0:
            continue
        mean, se = p.mean(0), p.std(0, ddof=1) / np.sqrt(len(p)) if len(p) > 1 else np.zeros(sel.sum())
        ax.semilogy(freqs[sel], mean, label=f"{label} (n={len(p)})")
        ax.fill_between(freqs[sel], mean - se, mean + se, alpha=0.2)
    for name, (lo, hi) in (bands or {}).items():
        ax.axvspan(lo, hi, alpha=0.06, color="gray")
        ax.text((lo + hi) / 2, ax.get_ylim()[1], name, ha="center", va="top", fontsize=7)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(f"Welch PSD at {channel}, cue +0.5 to +2.5 s")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_recording_psd(data: np.ndarray, sfreq: float, ch_names, channels, fmax: float = 60.0):
    """PSD of the continuous recording (finite stretches only) for chosen channels."""
    from scipy.signal import welch

    idx = [list(ch_names).index(c) for c in channels]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for c, i in zip(channels, idx):
        v = data[i][np.isfinite(data[i])]
        if v.size < 512:
            continue
        f, p = welch(v, fs=sfreq, nperseg=1024)
        ax.semilogy(f[f <= fmax], p[f <= fmax], label=c)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title("Whole-recording PSD (missing gaps excluded)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
