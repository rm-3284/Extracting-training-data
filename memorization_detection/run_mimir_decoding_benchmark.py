"""
MIMIR member-prefix decoding benchmark: baseline vs fast vs slow risk-aware decoding.

Runs three open LMs by default (GPT-Neo 2.7B, Pythia 2.8B, Pythia 1.4B), one at a time.
Slow mode is many infilling calls per token; use --slow-max-new-tokens (default 24) to keep
runtime manageable, or --skip-slow.

Usage (from repo root):
  export HF_TOKEN=...
  python memorization_detection/run_mimir_decoding_benchmark.py --n-examples 50

  python memorization_detection/run_mimir_decoding_benchmark.py \\
    --models gpt_neo_2p7 pythia_1p4 --n-examples 20 --output mia_decoding_bench.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

import torch
from datasets import load_dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Same directory contains `memorization_detection.py` (module name clashes with folder name;
# load explicitly so `python memorization_detection/run_mimir_decoding_benchmark.py` works).
_spec = importlib.util.spec_from_file_location(
    "mia_memdec_core",
    os.path.join(os.path.dirname(__file__), "memorization_detection.py"),
)
_mia = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mia)

MODEL_CONFIGS = _mia.MODEL_CONFIGS
load_lm = _mia.load_lm
generate_baseline = _mia.generate_baseline
generate_risk_aware = _mia.generate_risk_aware
split_by_tokens = _mia.split_by_tokens
token_overlap_with_suffix = _mia.token_overlap_with_suffix

DEFAULT_MODEL_KEYS = ("gpt_neo_2p7", "pythia_2p8", "pythia_1p4")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_benchmark_for_model(
    model_key: str,
    rows,
    *,
    n_examples: int,
    seed: int,
    prefix_tokens: int,
    suffix_tokens: int,
    max_new_tokens_baseline: int,
    max_new_tokens_fast: int,
    max_new_tokens_slow: int,
    skip_slow: bool,
    lambda_fast: float,
    lambda_slow: float,
    top_k: int,
    temperature: float,
    verbose: bool,
) -> Dict[str, Any]:
    hf_name = MODEL_CONFIGS[model_key]
    print(f"\n{'=' * 60}\nLoading {model_key} ({hf_name})\n{'=' * 60}")
    t0 = time.time()
    model, tokenizer = load_lm(hf_name)
    print(f"Load time: {time.time() - t0:.1f}s")

    modes = ["baseline", "fast"] + ([] if skip_slow else ["slow"])
    per_example: List[Dict[str, Any]] = []
    overlap_lists = {m: [] for m in modes}
    lcp_lists = {m: [] for m in modes}

    for i in range(n_examples):
        text = rows[i]["member"]
        prefix, true_suffix = split_by_tokens(
            text,
            tokenizer,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
        )
        if prefix is None:
            if verbose:
                print(f"skip example {i}: too short for prefix+suffix")
            continue

        row: Dict[str, Any] = {"example_index": i}
        for mode in modes:
            # Reproducible per (example, mode); paths differ so not paired across modes.
            set_seed(seed + 10_000 * i + 100 * modes.index(mode))

            if mode == "baseline":
                gen = generate_baseline(
                    prefix,
                    model,
                    tokenizer,
                    max_new_tokens=max_new_tokens_baseline,
                    temperature=temperature,
                )
            elif mode == "fast":
                gen = generate_risk_aware(
                    prefix,
                    model,
                    tokenizer,
                    max_new_tokens=max_new_tokens_fast,
                    top_k=top_k,
                    lambda_penalty=lambda_fast,
                    temperature=temperature,
                    mode="fast",
                )
            else:
                gen = generate_risk_aware(
                    prefix,
                    model,
                    tokenizer,
                    max_new_tokens=max_new_tokens_slow,
                    top_k=top_k,
                    lambda_penalty=lambda_slow,
                    temperature=temperature,
                    mode="slow",
                )

            ov, lcp = token_overlap_with_suffix(
                gen,
                prefix,
                true_suffix,
                tokenizer,
                max_tokens=suffix_tokens,
            )
            row[f"{mode}_overlap"] = ov
            row[f"{mode}_lcp"] = lcp
            overlap_lists[mode].append(ov)
            lcp_lists[mode].append(lcp)

        per_example.append(row)

        if verbose:
            print(
                f"example {i}: "
                + ", ".join(
                    f"{m} ov={row[f'{m}_overlap']:.4f} lcp={row[f'{m}_lcp']}"
                    for m in modes
                )
            )
        elif (i + 1) % 10 == 0 or i == 0:
            print(f"  progress: {i + 1}/{n_examples} examples with valid prefix/suffix")

    summary = {}
    for m in modes:
        summary[m] = {
            "mean_overlap": mean(overlap_lists[m]),
            "mean_lcp": mean(lcp_lists[m]),
            "n_scored": len(overlap_lists[m]),
        }

    print(f"\n--- {model_key} summary (n={summary['baseline']['n_scored']}) ---")
    for m in modes:
        s = summary[m]
        print(
            f"  {m:8s}  mean_overlap={s['mean_overlap']:.6f}  mean_lcp={s['mean_lcp']:.4f}"
        )

    unload_model(model)

    return {
        "model_key": model_key,
        "hf_model_name": hf_name,
        "summary": summary,
        "per_example": per_example,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_KEYS),
        help=f"Hugging Face model keys from MODEL_CONFIGS (default: {list(DEFAULT_MODEL_KEYS)})",
    )
    p.add_argument("--n-examples", type=int, default=50, help="First N rows from MIMIR split")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    p.add_argument(
        "--mimir-config",
        default="arxiv",
        help="MIMIR dataset config name",
    )
    p.add_argument(
        "--mimir-split",
        default="ngram_7_0.2",
        help="MIMIR split key",
    )
    p.add_argument("--prefix-tokens", type=int, default=64)
    p.add_argument("--suffix-tokens", type=int, default=64)
    p.add_argument(
        "--max-new-tokens-baseline",
        type=int,
        default=64,
        help="max_new_tokens for model.generate baseline",
    )
    p.add_argument(
        "--max-new-tokens-fast",
        type=int,
        default=64,
        help="max_new_tokens for fast risk-aware loop",
    )
    p.add_argument(
        "--slow-max-new-tokens",
        type=int,
        default=24,
        help="max_new_tokens for slow mode (each step is expensive)",
    )
    p.add_argument("--skip-slow", action="store_true", help="Only baseline + fast")
    p.add_argument("--lambda-fast", type=float, default=0.3)
    p.add_argument("--lambda-slow", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output", type=str, default="", help="Write full JSON results here")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for k in args.models:
        if k not in MODEL_CONFIGS:
            raise SystemExit(f"Unknown model key {k!r}. Choose from: {list(MODEL_CONFIGS)}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (required for iamgroot42/mimir).")

    print(f"Loading MIMIR config={args.mimir_config!r} split={args.mimir_split!r} ...")
    ds = load_dataset(
        "iamgroot42/mimir",
        args.mimir_config,
        token=token,
        trust_remote_code=True,
    )
    rows = ds[args.mimir_split]

    out: Dict[str, Any] = {
        "config": {
            "models": args.models,
            "n_examples_requested": args.n_examples,
            "seed": args.seed,
            "mimir_config": args.mimir_config,
            "mimir_split": args.mimir_split,
            "prefix_tokens": args.prefix_tokens,
            "suffix_tokens": args.suffix_tokens,
            "max_new_tokens_baseline": args.max_new_tokens_baseline,
            "max_new_tokens_fast": args.max_new_tokens_fast,
            "max_new_tokens_slow": args.slow_max_new_tokens,
            "skip_slow": args.skip_slow,
            "lambda_fast": args.lambda_fast,
            "lambda_slow": args.lambda_slow,
            "top_k": args.top_k,
            "temperature": args.temperature,
        },
        "results_by_model": {},
    }

    t_all = time.time()
    for model_key in args.models:
        out["results_by_model"][model_key] = run_benchmark_for_model(
            model_key,
            rows,
            n_examples=args.n_examples,
            seed=args.seed,
            prefix_tokens=args.prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            max_new_tokens_baseline=args.max_new_tokens_baseline,
            max_new_tokens_fast=args.max_new_tokens_fast,
            max_new_tokens_slow=args.slow_max_new_tokens,
            skip_slow=args.skip_slow,
            lambda_fast=args.lambda_fast,
            lambda_slow=args.lambda_slow,
            top_k=args.top_k,
            temperature=args.temperature,
            verbose=args.verbose,
        )

    out["wall_time_s"] = round(time.time() - t_all, 2)
    print(f"\nTotal wall time: {out['wall_time_s']}s")

    if args.output:
        path = os.path.abspath(args.output)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
