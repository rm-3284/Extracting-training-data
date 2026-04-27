"""Run baseline scoring methods on already-generated and labeled samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mia_eval.config_loader import active_model_bundle, load_merged_config, apply_dot_overrides
from mia_eval.evaluation_common import auc_direction as _auc_direction, jsonable as _jsonable
from mia_eval.labeling import load_labeled
from mia_eval.model_utils import load_causal_lm, pick_device, torch_dtype_from_str
from mia_eval.evaluation_common import split_masks as _split_masks
from mia_eval.scoring_baselines import (
    score_perplexity,
    score_zlib,
    score_lowercase,
    score_window,
)

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "mia_eval/config/defaults.yaml"))
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_merged_config(Path(args.config), None)
    cfg = apply_dot_overrides(cfg, args.set)

    model_key = cfg.get("active_model", "gpt_neo_2p7")
    out_root = Path(cfg.get("experiment", {}).get("output_dir", "mia_eval_outputs"))
    run_dir = out_root / model_key
    labeled_path = run_dir / "samples_labeled.jsonl"

    print(f"Loading labeled samples from {labeled_path}...")
    texts, labels, sources = load_labeled(labeled_path)
    y = np.array(labels, dtype=np.int64)

    ev = cfg.get("evaluation") or {}
    tr_m, va_m, te_m = _split_masks(
        y,
        float(ev.get("test_fraction", 0.2)),
        float(ev.get("val_fraction", 0.2)),
        int(ev.get("random_state", 42)),
    )

    bundle = active_model_bundle(cfg)
    exp = cfg.get("experiment") or {}
    device = pick_device(exp.get("device"))
    device = pick_device(exp.get("device"))
    print(f"Using device: {device}")
    dtype = torch_dtype_from_str(bundle.get("torch_dtype"))
    max_len = int(exp.get("max_length_tokens", 512))

    target = bundle["target_model"]
    tok_id = bundle.get("tokenizer") or target
    print(f"Loading model {target}...")
    model, tok = load_causal_lm(target, tok_id, device, dtype)

    results = {"model": model_key, "n_samples": len(y), "methods": {}}

    baseline_methods = {
        "perplexity": score_perplexity,
        "zlib": score_zlib,
        "lowercase": score_lowercase,
        "window": score_window,
    }

    for name, fn in baseline_methods.items():
        print(f"Scoring with {name}...")
        scores = np.array(fn(model, tok, texts, device, max_len), dtype=np.float64)
        auc_va, _ = _auc_direction(y[va_m], scores[va_m])
        auc_te, _ = _auc_direction(y[te_m], scores[te_m])
        results["methods"][name] = {
            "val_auc": float(auc_va),
            "test_auc": float(auc_te),
        }
        print(f"  {name}: val={auc_va:.4f} test={auc_te:.4f}")

    out_path = run_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    print(f"\nSaved results to {out_path}")
    print(json.dumps(_jsonable(results), indent=2))


if __name__ == "__main__":
    main()