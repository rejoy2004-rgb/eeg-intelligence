"""Validation helpers for EEG data.

Design rule: *never silently repair or drop data*. Every problem either raises a
``DataValidationError`` (the input cannot be used) or is written to a
``ValidationReport`` (the input is usable, but the user must be told).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when EEG data or metadata cannot be used safely."""


@dataclass
class ValidationReport:
    """Collects warnings and rejection records produced while loading / processing data."""

    warnings: list[str] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def reject(self, **record) -> None:
        """Record one rejected item, e.g. ``reject(trial=12, reason="expert artifact marker (1023)")``."""
        self.rejections.append(record)

    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rejections:
            counts[r["reason"]] = counts.get(r["reason"], 0) + 1
        return counts


def validate_signal_array(data: np.ndarray, n_channels: int, name: str = "signal") -> None:
    """Check a (channels, samples) array: type, dimensionality, size. NaNs are checked elsewhere."""
    if not isinstance(data, np.ndarray):
        raise DataValidationError(f"{name} must be a numpy array, got {type(data).__name__}")
    if data.ndim != 2:
        raise DataValidationError(f"{name} must be 2-D (channels, samples); got shape {data.shape}")
    if data.shape[0] != n_channels:
        raise DataValidationError(f"{name} must have {n_channels} channels; got {data.shape[0]}")
    if data.shape[1] == 0:
        raise DataValidationError(f"{name} is empty (0 samples)")
    if not np.issubdtype(data.dtype, np.floating):
        raise DataValidationError(f"{name} must be floating point; got dtype {data.dtype}")


def check_finite(data: np.ndarray, allowed_missing: np.ndarray | None = None, name: str = "signal") -> int:
    """Raise if non-finite values exist outside ``allowed_missing`` (boolean mask over samples).

    Returns the number of non-finite time samples that *are* allowed (documented missing data).
    """
    bad = ~np.isfinite(data).all(axis=0)  # per time sample
    if allowed_missing is not None:
        if allowed_missing.shape != bad.shape:
            raise DataValidationError("allowed_missing mask has the wrong length")
        unexpected = bad & ~allowed_missing
    else:
        unexpected = bad
    if unexpected.any():
        raise DataValidationError(
            f"{name} contains {int(unexpected.sum())} non-finite samples outside documented missing segments"
        )
    return int(bad.sum())


def validate_sfreq(sfreq: float, expected: float | None = None) -> None:
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise DataValidationError(f"Invalid sampling frequency: {sfreq}")
    if expected is not None and abs(sfreq - expected) > 1e-6:
        raise DataValidationError(
            f"Unexpected sampling frequency {sfreq} Hz (this pipeline is configured for {expected} Hz). "
            "Filter, epoch and feature settings depend on it; refusing to continue."
        )


def validate_channels(names: Sequence[str], expected: Sequence[str], anchors: dict[int, str] | None = None) -> None:
    """Check channel count, uniqueness and (optionally) the named anchor positions."""
    names = list(names)
    if len(names) != len(expected):
        raise DataValidationError(f"Expected {len(expected)} EEG channels, found {len(names)}: {names}")
    if len(set(names)) != len(names):
        raise DataValidationError("Duplicate channel names")
    for pos, anchor in (anchors or {}).items():
        if names[pos] != anchor:
            raise DataValidationError(
                f"Channel at position {pos} is '{names[pos]}', expected '{anchor}'. "
                "The positional channel mapping cannot be trusted for this file."
            )


def validate_events(events: pd.DataFrame, n_samples: int, known_codes: Iterable[int]) -> None:
    """Check the event table: columns, index range, unknown codes, ordering."""
    required = {"sample", "code", "duration"}
    if not required.issubset(events.columns):
        raise DataValidationError(f"Event table needs columns {sorted(required)}; has {list(events.columns)}")
    if len(events) == 0:
        raise DataValidationError("Recording has no event annotations")
    bad_idx = (events["sample"] < 0) | (events["sample"] >= n_samples)
    if bad_idx.any():
        raise DataValidationError(f"{int(bad_idx.sum())} events lie outside the recording (0..{n_samples - 1})")
    unknown = sorted(set(events["code"]) - set(known_codes))
    if unknown:
        raise DataValidationError(f"Unknown event codes in file: {unknown}")
    if not events["sample"].is_monotonic_increasing:
        raise DataValidationError("Event samples are not in chronological order")


def validate_labels(labels: np.ndarray, allowed: Iterable[int], name: str = "labels") -> None:
    labels = np.asarray(labels)
    if labels.size == 0:
        raise DataValidationError(f"{name} is empty")
    if not np.isfinite(labels.astype(float)).all():
        raise DataValidationError(f"{name} contain non-finite values")
    unknown = sorted(set(np.unique(labels).tolist()) - set(allowed))
    if unknown:
        raise DataValidationError(f"{name} contain unknown values {unknown}; allowed {sorted(allowed)}")
