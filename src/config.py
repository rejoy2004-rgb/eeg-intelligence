"""Configuration loading.

The whole project is driven by one YAML file (``configs/default.yaml``). We keep the
configuration as a plain nested dictionary wrapped in a tiny helper so that
(a) beginners can read it, and (b) the *exact* dictionary can be saved next to
every experiment result for reproducibility.
"""
from __future__ import annotations

import copy
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing or malformed."""


class Config(dict):
    """A dict with attribute-style access to nested sections (cfg.preprocessing.epoch_tmin_s)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path from the config relative to the project root."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


_REQUIRED_SECTIONS = ("dataset", "preprocessing", "features", "spatial", "models", "evaluation")


def load_config(path: str | Path | None = None) -> Config:
    """Load and sanity-check a YAML configuration."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")
    missing = [s for s in _REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ConfigError(f"Config is missing sections: {missing}")
    cfg = Config(raw)
    pre = cfg["preprocessing"]
    if pre["epoch_tmax_s"] <= pre["epoch_tmin_s"]:
        raise ConfigError("preprocessing.epoch_tmax_s must be greater than epoch_tmin_s")
    if len(cfg["dataset"]["eeg_channels"]) != cfg["dataset"]["n_eeg_channels"]:
        raise ConfigError("dataset.eeg_channels length must equal dataset.n_eeg_channels")
    return cfg


def software_versions() -> dict:
    """Versions of the libraries that influence numerical results (stored with every experiment)."""
    import importlib

    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "pandas", "mne", "sklearn", "torch"):
        try:
            versions[pkg] = importlib.import_module(pkg).__version__
        except Exception:  # package missing -> record that fact instead of crashing
            versions[pkg] = "not installed"
    return versions
