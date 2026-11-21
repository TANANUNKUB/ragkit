"""YAML config loading with deep-merge overrides (`extends: base.yaml`)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_config(path: str | Path, *, _seen: frozenset[Path] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    seen = _seen or frozenset()
    if path in seen:
        raise ValueError(f"Circular config extends involving {path}")
    raw = _load_yaml(path)
    extends = raw.pop("extends", None)
    if not extends:
        return raw
    parent = Path(extends)
    if not parent.is_absolute():
        parent = path.parent / parent
    base = load_config(parent, _seen=seen | {path})
    return deep_merge(base, raw)


def dump_config(cfg: dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
