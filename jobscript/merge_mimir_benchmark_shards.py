#!/usr/bin/env python3
"""Merge JSON outputs from run_mimir_decoding_benchmark.py array shards into one file.

Supports (1) one shard per model and (2) multiple shards per model that split --example-offset
ranges; in case (2), per_example rows are concatenated and summary is recomputed.

Shard configs must match except for: models, example_offset, n_examples_requested, and
wbc_reference_model (Slurm slow_6h uses a different HF reference per target LM). Older shards
missing newer JSON keys are normalized with defaults before comparison. If versions still
differ, use --allow-config-mismatch (see merge_meta.config_mismatch_warnings).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Keys merged runs may legitimately differ on (per-shard Slurm tasks, or per-target-model WBC).
_CONFIG_SKIP_MERGE = frozenset(
    {
        "models",
        "example_offset",
        "n_examples_requested",
        # Array jobs often pick a different reference HF id per target LM (see slow_6h.slurm).
        "wbc_reference_model",
    }
)

# Older shard JSON may omit keys added later; treat missing as these defaults before comparing.
_CONFIG_DEFAULTS_IF_MISSING: Dict[str, Any] = {
    "risk_score_mode": "delta",
    "risk_explore_eps": 0.07,
    "fast_aux_logprob_lambda": 0.05,
    "slow_aux_logprob_lambda": 0.0,
    "wbc_share_target_tokenizer": False,
}


def _load(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _shard_task_index(path: Path) -> int:
    m = re.search(r"_(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def _normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    for k, v in _CONFIG_DEFAULTS_IF_MISSING.items():
        if k not in out:
            out[k] = v
    return out


def _config_signature(cfg: Dict[str, Any]) -> Tuple[Any, ...]:
    """Hyperparameters that must match across shards (exclude dispatch / per-model keys)."""
    cfg = _normalize_config(cfg)
    keys = sorted(k for k in cfg if k not in _CONFIG_SKIP_MERGE)
    return tuple((k, cfg[k]) for k in keys)


def _config_diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Human-readable list of value differences (after normalization), ignoring skip keys."""
    a = _normalize_config(a)
    b = _normalize_config(b)
    keys = sorted(set(a) | set(b))
    lines: List[str] = []
    for k in keys:
        if k in _CONFIG_SKIP_MERGE:
            continue
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if va != vb:
            lines.append(f"  {k}: {va!r} vs {vb!r}")
    return lines


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _modes_from_rows(per_example: List[Dict[str, Any]]) -> List[str]:
    modes: List[str] = []
    if not per_example:
        return modes
    for k in per_example[0]:
        if k.endswith("_overlap"):
            modes.append(k[: -len("_overlap")])
    return sorted(set(modes))


def _recompute_summary(per_example: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    modes = _modes_from_rows(per_example)
    for m in modes:
        key_ov = f"{m}_overlap"
        key_lcp = f"{m}_lcp"
        ovs = [float(row[key_ov]) for row in per_example if key_ov in row]
        lcps = [float(row[key_lcp]) for row in per_example if key_lcp in row]
        summary[m] = {
            "mean_overlap": _mean(ovs),
            "mean_lcp": _mean(lcps),
            "n_scored": len(ovs),
        }
    return summary


def _merge_model_payloads(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if a.get("hf_model_name") != b.get("hf_model_name"):
        raise SystemExit(
            f"hf_model_name mismatch merging {a.get('model_key')!r}: "
            f"{a.get('hf_model_name')!r} vs {b.get('hf_model_name')!r}"
        )
    pe_a = a.get("per_example") or []
    pe_b = b.get("per_example") or []
    seen = {row["example_index"] for row in pe_a}
    for row in pe_b:
        ei = row["example_index"]
        if ei in seen:
            raise SystemExit(f"Duplicate example_index={ei} when merging model {a.get('model_key')!r}")
        seen.add(ei)
    merged_pe = pe_a + pe_b
    merged_pe.sort(key=lambda r: r["example_index"])
    return {
        "model_key": a.get("model_key"),
        "hf_model_name": a.get("hf_model_name"),
        "summary": _recompute_summary(merged_pe),
        "per_example": merged_pe,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "shards",
        nargs="+",
        type=Path,
        help="Shard JSON files (e.g. logs/mimir_slow_shard_<jobid>_*.json)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Merged JSON path",
    )
    p.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Warn and continue if configs differ (first shard wins for merged config block)",
    )
    args = p.parse_args()

    shard_paths = sorted(args.shards, key=_shard_task_index)

    shards: List[Dict[str, Any]] = []
    for path in shard_paths:
        if not path.is_file():
            raise SystemExit(f"Missing shard: {path}")
        shards.append(_load(path))

    base_cfg = dict(shards[0]["config"])
    sig0 = _config_signature(base_cfg)
    wall_times: List[float] = []
    merged_results: Dict[str, Any] = {}
    config_warnings: List[str] = []

    for i, doc in enumerate(shards):
        cfg = doc["config"]
        if _config_signature(cfg) != sig0:
            msg = (
                f"Config mismatch in {shard_paths[i]} "
                "(non-dispatch keys must match first shard after normalization; allowed to differ: "
                f"{sorted(_CONFIG_SKIP_MERGE)})"
            )
            if args.allow_config_mismatch:
                diff = _config_diff(base_cfg, cfg)
                config_warnings.append(f"{shard_paths[i]}:\n" + "\n".join(diff[:50]))
                if len(diff) > 50:
                    config_warnings.append(f"  ... and {len(diff) - 50} more differing keys")
                print(f"WARNING: {msg}", flush=True)
            else:
                print(
                    "\nHint: per-model WBC jobs set different wbc_reference_model per shard; "
                    "the merge script ignores that key. If you still see this, old vs new shard "
                    "JSON may differ; retry with --allow-config-mismatch.\n",
                    flush=True,
                )
                raise SystemExit(msg)
        rbm = doc.get("results_by_model") or {}
        for mk, payload in rbm.items():
            if mk not in merged_results:
                merged_results[mk] = payload
            else:
                merged_results[mk] = _merge_model_payloads(merged_results[mk], payload)
        wt = doc.get("wall_time_s")
        if isinstance(wt, (int, float)):
            wall_times.append(float(wt))

    model_keys = sorted(merged_results.keys())
    merged_span = 0
    for payload in merged_results.values():
        pe = payload.get("per_example") or []
        if pe:
            merged_span = max(merged_span, max(r["example_index"] for r in pe) + 1)

    wall_sum = round(sum(wall_times), 2)

    wbc_ref_by_target: Dict[str, Any] = {}
    for doc in shards:
        cfg = doc.get("config") or {}
        for mk in doc.get("results_by_model") or {}:
            if mk not in wbc_ref_by_target:
                wbc_ref_by_target[mk] = cfg.get("wbc_reference_model")

    unified_cfg = {
        **_normalize_config(base_cfg),
        "models": model_keys,
        "example_offset": 0,
        "n_examples_requested": merged_span,
    }

    out: Dict[str, Any] = {
        "config": unified_cfg,
        "results_by_model": merged_results,
        "wall_time_s": wall_sum,
        "merge_meta": {
            "shard_paths": [str(p.resolve()) for p in shard_paths],
            "shard_wall_times_s": wall_times,
            "merged_wall_time_s": wall_sum,
            "merged_example_span": merged_span,
            "config_keys_ignored_for_merge_check": sorted(_CONFIG_SKIP_MERGE),
        },
    }
    if wbc_ref_by_target:
        out["merge_meta"]["wbc_reference_model_by_target"] = wbc_ref_by_target
    if config_warnings:
        out["merge_meta"]["config_mismatch_warnings"] = config_warnings

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output.resolve()} ({len(model_keys)} models)")


if __name__ == "__main__":
    main()
