#!/usr/bin/env python3
"""
Batch Carlini-style extraction (open prompt → top-k + temperature-decay samples).

Reuses ``mia_eval.generation.generate_diverse_samples`` (same as ``run_pipeline --steps generate``)
but iterates over many Hugging Face checkpoints from ``carlini_open_models.yaml``.

Outputs, per run_key:
  ``{experiment.output_dir}/carlini_extract/{run_key}/samples.jsonl``
  ``{experiment.output_dir}/carlini_extract/{run_key}/run_meta.json``

From repository root:
  python -m mia_eval.run_carlini_extraction_batch --config mia_eval/config/carlini_open_models.yaml
  python -m mia_eval.run_carlini_extraction_batch --config mia_eval/config/carlini_open_models.yaml --only olmo2_7b_base,redpajama_7b_base
  python -m mia_eval.run_carlini_extraction_batch --num-samples-per-strategy 500   # override YAML
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_eval.config_loader import load_yaml
from mia_eval.generation import generate_diverse_samples


def _merge_manifest_runs(
    existing_runs: List[Dict[str, Any]], new_runs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge runs by ``run_key`` (new entries replace prior status for same key).
    Preserves original order for existing keys and appends unseen keys.
    """
    by_key: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for r in existing_runs:
        k = str(r.get("run_key", ""))
        if not k:
            continue
        if k not in by_key:
            order.append(k)
        by_key[k] = r
    for r in new_runs:
        k = str(r.get("run_key", ""))
        if not k:
            continue
        if k not in by_key:
            order.append(k)
        by_key[k] = r
    return [by_key[k] for k in order]


def _merge_manifest_locked(
    manifest_path: Path, partial: Dict[str, Any], config_path: Path, out_root: Path
) -> Dict[str, Any]:
    """
    Merge a task's partial results into a shared manifest with an advisory lock.
    Safe for Slurm array jobs running concurrently on a shared filesystem.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        if manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as f:
                merged: Dict[str, Any] = json.load(f)
        else:
            merged = {
                "config_path": str(config_path.resolve()),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "output_root": str(out_root.resolve()),
                "runs": [],
            }
        merged["config_path"] = str(config_path.resolve())
        merged["output_root"] = str(out_root.resolve())
        merged["runs"] = _merge_manifest_runs(
            list(merged.get("runs") or []), list(partial.get("runs") or [])
        )
        merged["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
    return merged


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_pipeline_cfg(yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Subset of full mia_eval config: only experiment + generation."""
    return {
        "experiment": copy.deepcopy(yaml_cfg.get("experiment") or {}),
        "generation": copy.deepcopy(yaml_cfg.get("generation") or {}),
    }


def _parse_only(s: str | None) -> Set[str] | None:
    if not s or not str(s).strip():
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def run() -> None:
    p = argparse.ArgumentParser(description="Carlini-style extraction for many HF models")
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "mia_eval" / "config" / "carlini_open_models.yaml",
        help="YAML with experiment, generation, and carlini_runs",
    )
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated run_key values to run (default: all)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if samples.jsonl already exists",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs and exit",
    )
    p.add_argument(
        "--num-samples-per-strategy",
        type=int,
        default=None,
        help="Override generation.num_samples_per_strategy in YAML (each enabled method gets N samples)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override generation.batch_size (GPU memory)",
    )
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Shared manifest path (default: <output_root>/manifest.json).",
    )
    p.add_argument(
        "--manifest-mode",
        type=str,
        choices=("merge", "replace"),
        default="merge",
        help="merge: lock+merge with existing manifest (array-safe); replace: overwrite.",
    )
    args = p.parse_args()

    raw = load_yaml(args.config)
    runs: List[Dict[str, Any]] = list(raw.get("carlini_runs") or [])
    if not runs:
        raise SystemExit("config has no carlini_runs")

    only = _parse_only(args.only)
    base_cfg = _build_pipeline_cfg(raw)
    gen = base_cfg.setdefault("generation", {})
    if args.num_samples_per_strategy is not None:
        gen["num_samples_per_strategy"] = int(args.num_samples_per_strategy)
    if args.batch_size is not None:
        gen["batch_size"] = int(args.batch_size)
    exp = base_cfg.get("experiment") or {}
    out_root = ROOT / str(exp.get("output_dir", "mia_eval_outputs")) / "carlini_extract"
    seed0 = int(exp.get("seed", 42))

    planned: List[Dict[str, Any]] = []
    for spec in runs:
        rk = spec.get("run_key")
        if not rk:
            continue
        if only is not None and rk not in only:
            continue
        planned.append(spec)

    if not planned:
        raise SystemExit("No runs selected (check --only / carlini_runs)")

    if args.dry_run:
        gcfg = base_cfg.get("generation") or {}
        nps = int(gcfg.get("num_samples_per_strategy", 0))
        nuc = (gcfg.get("nucleus") or {}) if isinstance(gcfg.get("nucleus"), dict) else {}
        n_methods = 1  # top_k
        temp_on = (gcfg.get("temperature_decay") or {}).get("enabled", True)
        if temp_on:
            n_methods += 1
        nuc_on = bool(nuc.get("enabled"))
        if nuc_on:
            n_methods += 1
        ip = gcfg.get("internet_prefix") or {}
        inet_on = bool(ip.get("enabled"))
        n_inet = int(ip["num_samples_per_strategy"]) if ip.get("num_samples_per_strategy") is not None else nps
        n_nuc = int(nuc.get("num_samples", nps)) if nuc_on else 0
        n_nuc_i = int(nuc.get("num_samples_internet", n_inet)) if nuc_on else 0
        lines_plain = nps + (nps if temp_on else 0) + (n_nuc if nuc_on else 0)
        raw_apply = ip.get("apply_to")
        if raw_apply is None:
            inet_strats = {"top_k"}
        elif isinstance(raw_apply, str):
            inet_strats = {raw_apply}
        else:
            inet_strats = set(raw_apply)
        lines_inet = 0
        if inet_on:
            if "top_k" in inet_strats:
                lines_inet += n_inet
            if temp_on and "temperature_decay" in inet_strats:
                lines_inet += n_inet
            if nuc_on and "nucleus" in inet_strats:
                lines_inet += n_nuc_i
        print(f"Output root: {out_root}")
        print(
            f"Generation: ~{lines_plain + lines_inet} lines/run "
            f"(plain {lines_plain} + internet {lines_inet}; batch_size={gcfg.get('batch_size')})"
        )
        for spec in planned:
            print(
                f"  {spec['run_key']}: {spec.get('hf_model_id')} "
                f"({spec.get('variant')})"
            )
        return

    manifest_partial: Dict[str, Any] = {
        "config_path": str(args.config.resolve()),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(out_root.resolve()),
        "runs": [],
    }

    for spec in planned:
        run_key = spec["run_key"]
        hf_id = spec["hf_model_id"]
        out_dir = out_root / run_key
        out_jsonl = out_dir / "samples.jsonl"
        if args.skip_existing and out_jsonl.is_file():
            print(f"[skip] {run_key}: exists {out_jsonl}")
            manifest_partial["runs"].append(
                {
                    "run_key": run_key,
                    "status": "skipped_existing",
                    "samples_jsonl": str(out_jsonl.resolve()),
                }
            )
            continue

        tok_id = spec.get("tokenizer") or hf_id
        dtype_str = spec.get("torch_dtype") or "float16"
        model_bundle: Dict[str, Any] = {
            "target_model": hf_id,
            "tokenizer": tok_id,
            "torch_dtype": dtype_str,
        }

        _set_seeds(seed0)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "run_meta.json"
        meta = {
            "run_key": run_key,
            "project": spec.get("project"),
            "hf_model_id": hf_id,
            "tokenizer": tok_id,
            "torch_dtype": dtype_str,
            "variant": spec.get("variant"),
            "post_training": spec.get("post_training"),
            "generation": base_cfg.get("generation"),
            "seed": seed0,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[run] {run_key} ← {hf_id} ({dtype_str})", flush=True)
        try:
            generate_diverse_samples(base_cfg, model_bundle, out_jsonl)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[fail] {run_key}: {err}", flush=True)
            traceback.print_exc(file=sys.stdout)
            manifest_partial["runs"].append(
                {
                    "run_key": run_key,
                    "status": "error",
                    "error": err,
                    "samples_jsonl": str(out_jsonl.resolve()),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        n_lines = sum(1 for _ in open(out_jsonl, encoding="utf-8"))
        print(
            f"[ok] {run_key}: wrote {n_lines} lines → {out_jsonl}",
            flush=True,
        )
        manifest_partial["runs"].append(
            {
                "run_key": run_key,
                "status": "ok",
                "samples_jsonl": str(out_jsonl.resolve()),
                "run_meta": str(meta_path.resolve()),
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_partial["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = args.manifest_path or (out_root / "manifest.json")
    if args.manifest_mode == "replace":
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_partial, f, indent=2, ensure_ascii=False)
        print(f"Wrote {manifest_path} (replace)")
    else:
        _merge_manifest_locked(
            manifest_path=manifest_path,
            partial=manifest_partial,
            config_path=args.config,
            out_root=out_root,
        )
        print(f"Wrote {manifest_path} (merge)")


if __name__ == "__main__":
    run()
