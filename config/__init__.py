from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_PATH = _CONFIG_DIR / "default.yaml"

_cache: Dict[str, Any] = {}


def load(path: str | Path | None = None) -> Dict[str, Any]:
    """Load and cache a YAML config.  Falls back to ``config/default.yaml``."""
    key = str(path) if path else "__default__"
    if key in _cache:
        return _cache[key]

    target = Path(path) if path else _DEFAULT_PATH
    with open(target) as fh:
        cfg = yaml.safe_load(fh)

    _cache[key] = cfg
    return cfg


def get(section: str, key: str | None = None, *, path: str | Path | None = None) -> Any:
    """Shorthand:  ``get("apf", "k_att")`` → ``1.0``."""
    cfg = load(path)
    val = cfg[section]
    if key is not None:
        val = val[key]
    return val
