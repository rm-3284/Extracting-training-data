"""Load and merge YAML experiment configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_merged_config(
    defaults_path: Path,
    experiment_path: Path | None = None,
) -> Dict[str, Any]:
    cfg = load_yaml(defaults_path)
    if experiment_path and experiment_path.is_file():
        cfg = deep_merge(cfg, load_yaml(experiment_path))
    return cfg


def active_model_bundle(cfg: Dict[str, Any]) -> Dict[str, Any]:
    key = cfg.get("active_model")
    models = cfg.get("models") or {}
    if key not in models:
        raise KeyError(f"active_model {key!r} not found in config models keys={list(models)}")
    return models[key]


def iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    """Cartesian product of lists in grid."""
    import itertools

    keys = list(grid.keys())
    if not keys:
        yield {}
        return
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def apply_dot_overrides(cfg: Dict[str, Any], pairs: List[str]) -> Dict[str, Any]:
    """pairs like ['generation.seq_len=128', 'active_model=pythia_2p8']."""
    out = copy.deepcopy(cfg)
    for p in pairs:
        if "=" not in p:
            continue
        key, _, val = p.partition("=")
        parts = key.strip().split(".")
        cur: Any = out
        for part in parts[:-1]:
            if part not in cur:
                cur[part] = {}
            cur = cur[part]
        # naive literal parse
        vstrip = val.strip()
        if vstrip.lower() in ("true", "false"):
            cur[parts[-1]] = vstrip.lower() == "true"
        elif vstrip.isdigit():
            cur[parts[-1]] = int(vstrip)
        elif _is_float(vstrip):
            cur[parts[-1]] = float(vstrip)
        elif vstrip == "null":
            cur[parts[-1]] = None
        else:
            cur[parts[-1]] = vstrip
    return out


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
