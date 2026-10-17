"""
src/utils/config_loader.py
---------------------------
Load a YAML experiment configuration file and return it as a dict.
"""

from __future__ import annotations

import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """
    Load a YAML config file and return a plain dict.

    Parameters
    ----------
    path : str — path to the YAML config file

    Returns
    -------
    dict
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _validate_config(cfg, config_path)
    return cfg


def _validate_config(cfg: dict, path: Path) -> None:
    """Raise a clear error if required top-level keys are missing."""
    required = ["task", "model_name_or_path", "dataset", "training"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config '{path}' is missing required keys: {missing}")

    valid_tasks = {"qa", "summarization", "classification", "extraction"}
    if cfg["task"] not in valid_tasks:
        raise ValueError(
            f"Config '{path}': task must be one of {valid_tasks}, got '{cfg['task']}'"
        )
