"""Loading BCI Competition IV Dataset 2a (GDF files) into plain NumPy / pandas objects.

What this module guarantees
---------------------------
* Only the 22 EEG channels are kept. The 3 EOG channels are *not* loaded as features
  (the dataset providers state they "must not be used for classification").
* Signals are in microvolts, shape ``(22, n_samples)``.
* The data providers encode the 100-sample gap between runs as the amplifier's overflow value
  (-1600 uV on every channel; verified on real files, it starts exactly at every run-start marker).
  We convert these gaps to NaN, keep their positions in ``Recording.missing_segments`` and write a
  warning. Nothing is invented to fill them.
* Event codes follow Table 2 of the official description (see ``configs/default.yaml``).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from src.config import Config
from src.data.validation import (
    DataValidationError,
    ValidationReport,
    check_finite,
    validate_channels,
    validate_events,
    validate_labels,
    validate_sfreq,
    validate_signal_array,
)

OVERFLOW_UV = -1600.0  # documented amplifier overflow encoding of "missing" samples


@dataclass
class Recording:
    """One GDF file (one subject, one session), EEG channels only."""

    subject: int | None
    session: str | None
    sfreq: float
    ch_names: list[str]
    data: np.ndarray                      # (22, n_samples), microvolts, NaN in missing segments
    events: pd.DataFrame                  # columns: sample, code, duration (samples)
    missing_segments: list[tuple[int, int]] = field(default_factory=list)  # [start, stop) samples
    source_path: str = ""
    report: ValidationReport = field(default_factory=ValidationReport)

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sfreq

    def sha256(self) -> str | None:
        """Hash of the source file (for the experiment record). None if there is no file."""
        p = Path(self.source_path)
        if not p.is_file():
            return None
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


def recording_path(cfg: Config, subject: int, session: str) -> Path:
    if subject not in cfg.dataset["subjects"]:
        raise DataValidationError(f"Unknown subject {subject}; valid: {cfg.dataset['subjects']}")
    if session not in cfg.dataset["sessions"]:
        raise DataValidationError(f"Unknown session '{session}'; valid: {cfg.dataset['sessions']}")
    return cfg.resolve_path(cfg.dataset["raw_dir"]) / f"A{subject:02d}{session}.gdf"


def find_missing_segments(data: np.ndarray, overflow_uv: float = OVERFLOW_UV) -> list[tuple[int, int]]:
    """Find runs of samples where *every* channel equals the overflow value. Returns [start, stop)."""
    is_gap = np.all(data <= overflow_uv + 1e-3, axis=0)
    if not is_gap.any():
        return []
    padded = np.concatenate([[0], is_gap.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def recording_from_raw(raw, cfg: Config, subject: int | None, session: str | None, source: str = "") -> Recording:
    """Convert an ``mne.io.Raw`` (already in memory) into a validated ``Recording``.

    Separated from file reading so it can be unit-tested with synthetic ``RawArray`` objects.
    """
    ds = cfg.dataset
    validate_sfreq(float(raw.info["sfreq"]), ds["expected_sfreq"])
    if len(raw.ch_names) != ds["n_total_channels"]:
        raise DataValidationError(
            f"Expected {ds['n_total_channels']} channels (22 EEG + 3 EOG), found {len(raw.ch_names)}"
        )
    eeg_idx = [i for i, n in enumerate(raw.ch_names) if n.upper().startswith("EEG")]
    if len(eeg_idx) != ds["n_eeg_channels"]:
        raise DataValidationError(f"Expected {ds['n_eeg_channels']} 'EEG-*' channels, found {len(eeg_idx)}")

    gdf_names = [raw.ch_names[i].split("-", 1)[1] for i in eeg_idx]
    anchors = {int(k): v for k, v in ds["anchor_channels"].items()}
    # Only 5 channels are named in the file; they must sit where the positional mapping expects them.
    for pos, anchor in anchors.items():
        if gdf_names[pos] != anchor:
            raise DataValidationError(
                f"Channel at position {pos} is '{gdf_names[pos]}', expected '{anchor}'. "
                "The positional channel mapping cannot be trusted for this file."
            )
    validate_channels(list(ds["eeg_channels"]), ds["eeg_channels"])  # config itself must be consistent
    data = raw.get_data(picks=eeg_idx) * 1e6                         # volts -> microvolts
    validate_signal_array(data, ds["n_eeg_channels"], "EEG data")

    report = ValidationReport()
    segments = find_missing_segments(data)
    if segments:
        gap = np.zeros(data.shape[1], dtype=bool)
        for a, b in segments:
            gap[a:b] = True
        data[:, gap] = np.nan
        report.warn(
            f"{len(segments)} documented missing segment(s) ({int(gap.sum())} samples, encoded as "
            f"{OVERFLOW_UV:.0f} uV overflow) converted to NaN; positions kept in Recording.missing_segments"
        )
    # Any non-finite value must belong to a documented gap (all channels NaN at once).
    check_finite(data, allowed_missing=np.isnan(data).all(axis=0), name="EEG data")

    ann = raw.annotations
    samples = np.round((ann.onset - raw.first_time) * raw.info["sfreq"]).astype(int)
    durations = np.round(ann.duration * raw.info["sfreq"]).astype(int)
    try:
        codes = np.array([int(d) for d in ann.description])
    except ValueError as exc:
        raise DataValidationError(f"Non-numeric event annotation found: {exc}") from exc
    order = np.argsort(samples, kind="stable")
    events = pd.DataFrame({"sample": samples[order], "code": codes[order], "duration": durations[order]})
    known = set(ds["events"].values()) | {276, 277, 1072}  # idle-EEG and eye-movement blocks
    validate_events(events, data.shape[1], known)

    return Recording(
        subject=subject, session=session, sfreq=float(raw.info["sfreq"]), ch_names=list(ds["eeg_channels"]),
        data=data, events=events, missing_segments=segments, source_path=source, report=report,
    )


_NAME_RE = re.compile(r"^A(\d{2})([TE])$", re.IGNORECASE)


def load_gdf_file(path: str | Path, cfg: Config, subject: int | None = None, session: str | None = None) -> Recording:
    """Read one GDF file with MNE. Subject/session are parsed from names like ``A03T.gdf`` if not given."""
    import mne

    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"EEG file not found: {path}")
    if path.suffix.lower() != ".gdf":
        raise DataValidationError(f"Unsupported file format '{path.suffix}'; only .gdf is supported")
    if path.stat().st_size == 0:
        raise DataValidationError(f"EEG file is empty: {path}")
    m = _NAME_RE.match(path.stem)
    if m:
        subject = int(m.group(1)) if subject is None else subject
        session = m.group(2).upper() if session is None else session
    try:
        raw = mne.io.read_raw_gdf(str(path), preload=True, verbose="ERROR")
    except Exception as exc:  # MNE raises many exception types for corrupt files
        raise DataValidationError(f"Could not read GDF file {path.name} (corrupted or invalid?): {exc}") from exc
    return recording_from_raw(raw, cfg, subject, session, source=str(path))


def load_recording(cfg: Config, subject: int, session: str) -> Recording:
    """Load one participant/session of the dataset (path built from the config)."""
    return load_gdf_file(recording_path(cfg, subject, session), cfg, subject, session)


def load_external_labels(cfg: Config, subject: int, session: str) -> np.ndarray:
    """Official 'true labels' (classes 1-4, one per trial) from ``A0xT.mat`` / ``A0xE.mat``."""
    path = cfg.resolve_path(cfg.dataset["labels_dir"]) / f"A{subject:02d}{session}.mat"
    if not path.is_file():
        raise DataValidationError(f"Label file not found: {path}")
    try:
        mat = sio.loadmat(str(path))
    except Exception as exc:
        raise DataValidationError(f"Could not read label file {path.name}: {exc}") from exc
    if "classlabel" not in mat:
        raise DataValidationError(
            f"{path.name} has no 'classlabel' variable (found {[k for k in mat if not k.startswith('__')]})"
        )
    labels = np.asarray(mat["classlabel"]).ravel().astype(int)
    validate_labels(labels, allowed=[1, 2, 3, 4], name=f"{path.name} classlabel")
    return labels
