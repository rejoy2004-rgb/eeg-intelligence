"""Shared test fixtures.

!!! SYNTHETIC DATA WARNING !!!
Everything generated in this file is artificial noise plus a deliberately planted 10 Hz rhythm.
It exists ONLY to test software behaviour (shapes, alignment, error handling, leakage guards).
Nothing computed from it says anything about real EEG or about motor-imagery decoding.
Real-data checks live in tests/test_real_data.py and are skipped when the dataset is absent.
"""
from __future__ import annotations

import copy

import mne
import numpy as np
import pytest

from src.config import load_config
from src.data.dataset import recording_from_raw

SFREQ = 250.0
TRIAL_SAMPLES = 2000            # 8 s per trial
RUN_LEAD = 350                  # samples between run start and first trial (as in the real files)
GAP = 100                       # documented missing samples at each run start (except the first)
mne.set_log_level("ERROR")


@pytest.fixture(scope="session")
def base_cfg():
    return load_config()


@pytest.fixture
def cfg(base_cfg):
    """A private, mutable copy of the default configuration."""
    c = copy.deepcopy(base_cfg)
    return c


def make_synthetic_raw(cfg, n_trials=32, n_runs=2, seed=0, unknown_cue=False,
                       expert_artifact_trials=(), spike_trials=(), effect=0.15, calibration_s=0.0, eog_leak=0.0):
    """Synthetic 25-channel MNE Raw that mimics the *file structure* of BCI IV 2a.

    Class cycle 1,2,3,4,...; left (1) attenuates the planted 10 Hz rhythm at C4, right (2) at C3.
    Returns (raw, label4) where label4 holds the true class (1-4) of every trial.
    calibration_s > 0 prepends an "eye movements" block (event 1072) before the first run; eog_leak > 0 mixes the EOG channels
    into the EEG with a fixed random mixing matrix (so a regression has something to remove). Defaults leave the data unchanged.
    """
    rng = np.random.default_rng(seed)
    ds = cfg["dataset"]
    anchors = {int(k): v for k, v in ds["anchor_channels"].items()}
    names, k = [], 0
    for i, n in enumerate(ds["eeg_channels"]):
        if i in anchors:
            names.append(f"EEG-{n}")
        else:
            names.append(f"EEG-{k}")
            k += 1
    names += ["EOG-left", "EOG-central", "EOG-right"]
    per_run = n_trials // n_runs
    lead = int(round(calibration_s * SFREQ))
    n_samples = lead + n_runs * (RUN_LEAD + per_run * TRIAL_SAMPLES) + 500
    t = np.arange(n_samples) / SFREQ
    eeg = rng.normal(0, 3.0, (22, n_samples))
    # Fixed spatial phase pattern shared by all synthetic recordings (a stable 'source' topography),
    # so a spatial filter learned on one recording can transfer to another, as in real data.
    phase = np.random.default_rng(12345).uniform(0, 2 * np.pi, (22, 1))
    amp = np.full((22, n_samples), 8.0)
    label4 = np.array([(i % 4) + 1 for i in range(n_trials)])

    onset, dur, desc = [], [], []
    gap_starts = []
    pos = lead
    if calibration_s > 0:
        onset.append(0.0); dur.append(calibration_s); desc.append("1072")
    trial = 0
    for r in range(n_runs):
        onset.append(pos / SFREQ); dur.append(0.004); desc.append("32766")
        if r > 0:
            gap_starts.append(pos)
        pos += RUN_LEAD
        for _ in range(per_run):
            cue = pos + 500
            onset.append(pos / SFREQ); dur.append(7.5); desc.append("768")
            lab = label4[trial]
            onset.append(cue / SFREQ); dur.append(0.0)
            desc.append("783" if unknown_cue else str(768 + lab))
            if trial in expert_artifact_trials:
                onset.append(pos / SFREQ); dur.append(7.5); desc.append("1023")
            a, b = cue + 125, cue + 625
            if lab == 1:
                amp[11, a:b] *= effect
            elif lab == 2:
                amp[7, a:b] *= effect
            if trial in spike_trials:
                eeg[3, cue + 300] += 3000.0
            pos += TRIAL_SAMPLES
            trial += 1
    eeg += amp * np.sin(2 * np.pi * 10.0 * t + phase)
    for g in gap_starts:
        eeg[:, g:g + GAP] = -1600.0
    eog = rng.normal(0, 5.0, (3, n_samples))
    if eog_leak > 0:
        eeg = eeg + eog_leak * (np.random.default_rng(777).normal(size=(22, 3)) @ eog)
        for g in gap_starts:
            eeg[:, g:g + GAP] = -1600.0
    data = np.vstack([eeg, eog]) * 1e-6   # volts
    info = mne.create_info(names, SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onset, dur, desc))
    return raw, label4


@pytest.fixture
def synth(cfg):
    raw, label4 = make_synthetic_raw(cfg)
    return recording_from_raw(raw, cfg, subject=1, session="T"), label4


@pytest.fixture
def make_recording(cfg):
    def _make(subject=1, session="T", **kw):
        raw, label4 = make_synthetic_raw(cfg, seed=subject * 10 + (session == "E"), **kw)
        return recording_from_raw(raw, cfg, subject, session), label4
    return _make
