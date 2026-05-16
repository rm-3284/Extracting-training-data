#!/usr/bin/env python3
"""
Generate a fixed number of samples for **one** Carlini or memorization_detection ``source``.

Each Slurm array task should call this with a distinct ``--strategy`` so every method gets
the same ``--num-samples`` (e.g. 500) in its own output directory::

  …/{carlini_extract_subdir}/{run_key}/by_method/{strategy}/samples.jsonl

Example::

  python -m mia_eval.run_carlini_extraction_per_strategy \\
    --config mia_eval/config/carlini_open_models.yaml \\
    --merge-config mia_eval/config/carlini_overlay_decode_defenses.yaml \\
    --run-key gpt_neo_2p7 \\
    --strategy top_k \\
    --num-samples 500
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_eval.carlini_sample_sources import GENERATION_SOURCES_ORDER
from mia_eval.config_loader import deep_merge, load_yaml
from mia_eval.generation import generate_samples_for_source


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_run_spec(yaml_cfg: Dict[str, Any], run_key: str) -> Dict[str, Any]:
    for spec in yaml_cfg.get("carlini_runs") or []:
        if spec.get("run_key") == run_key:
            return dict(spec)
    raise KeyError(f"run_key {run_key!r} not in carlini_runs")


def run() -> None:
    p = argparse.ArgumentParser(description="Carlini extraction for one generation source")
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "mia_eval" / "config" / "carlini_open_models.yaml",
    )
    p.add_argument("--merge-config", type=Path, default=None)
    p.add_argument("--run-key", type=str, required=True)
    p.add_argument(
        "--strategy",
        type=str,
        required=True,
        help=f"One of: {', '.join(GENERATION_SOURCES_ORDER)}",
    )
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: …/by_method/{strategy} under the run directory",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    strategy = args.strategy.strip()
    if strategy not in GENERATION_SOURCES_ORDER:
        raise SystemExit(
            f"Unknown --strategy {strategy!r}; allowed: {list(GENERATION_SOURCES_ORDER)}"
        )

    raw = load_yaml(args.config)
    if args.merge_config is not None:
        raw = deep_merge(raw, load_yaml(args.merge_config))

    spec = _find_run_spec(raw, args.run_key)
    exp = raw.get("experiment") or {}
    gen = raw.setdefault("generation", {})
    gen["num_samples_per_strategy"] = int(args.num_samples)
    if args.batch_size is not None:
        gen["batch_size"] = int(args.batch_size)

    out_sub = str(exp.get("carlini_extract_subdir", "carlini_extract") or "carlini_extract")
    out_root = ROOT / str(exp.get("output_dir", "mia_eval_outputs")) / out_sub
    run_dir = out_root / args.run_key
    out_method_dir = args.output_dir or (run_dir / "by_method" / strategy)
    out_jsonl = out_method_dir / "samples.jsonl"

    if args.dry_run:
        print(f"run_key={args.run_key}  strategy={strategy}  num_samples={args.num_samples}")
        print(f"output={out_jsonl}")
        print(f"model={spec.get('hf_model_id')}")
        return

    seed0 = int(exp.get("seed", 42))
    _set_seeds(seed0 + hash(strategy) % 10000)

    model_bundle: Dict[str, Any] = {
        "target_model": spec["hf_model_id"],
        "tokenizer": spec.get("tokenizer") or spec["hf_model_id"],
        "torch_dtype": spec.get("torch_dtype", "float16"),
    }
    if spec.get("reference_model"):
        model_bundle["reference_model"] = spec["reference_model"]

    cfg = {
        "experiment": copy.deepcopy(exp),
        "generation": copy.deepcopy(gen),
    }

    out_method_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_key": args.run_key,
        "strategy": strategy,
        "num_samples": int(args.num_samples),
        "hf_model_id": spec.get("hf_model_id"),
        "generation": gen,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_method_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[run] {args.run_key} / {strategy} → {out_jsonl}", flush=True)
    try:
        generate_samples_for_source(
            cfg,
            model_bundle,
            strategy,
            out_jsonl,
            num_samples=int(args.num_samples),
        )
    except Exception as e:
        print(f"[fail] {strategy}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise

    n_lines = sum(1 for _ in open(out_jsonl, encoding="utf-8"))
    print(f"[ok] wrote {n_lines} lines", flush=True)


if __name__ == "__main__":
    run()
