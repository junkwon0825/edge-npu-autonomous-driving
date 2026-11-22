"""JSON configuration loading. Configuration lives in configs/, not in code."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from npu_ad import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.json"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load configs/default.json, optionally deep-merged with another file and overrides."""
    cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        cfg = _merge(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
