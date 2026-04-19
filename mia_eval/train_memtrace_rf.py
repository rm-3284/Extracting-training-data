#!/usr/bin/env python3
"""
Train a memTrace RandomForest + StandardScaler on **proxy-labeled** JSONL (from
``prepare_memtrace_proxy_jsonl``), **one artifact per Qwen preset**.

Saves ``joblib`` dict ``{{ "scaler": StandardScaler, "rf": RandomForestClassifier }}``
compatible with ``score_sequence.py --memtrace-rf``.

Example (all presets in one process):
  python -m mia_eval.train_memtrace_rf \\
    --config mia_eval/config/defaults.yaml \\
    --experiment mia_eval/config/qwen2p5.yaml \\
    --proxy-jsonl data/qwen_memtrace_proxy_train.jsonl \\
    --output-dir data/memtrace_rfs_qwen \\
    --use-yaml-presets

One preset (e.g. Slurm array — one GPU job per model):
  python -m mia_eval.train_memtrace_rf ... --output-dir data/memtrace_rfs_qwen --preset qwen25_7b_base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_eval.config_loader import active_model_bundle, load_merged_config, apply_dot_overrides
from mia_eval.model_utils import load_causal_lm, pick_device, torch_dtype_from_str
from mia_eval.scoring_memtrace import (
    _sanitize_feature_matrix,
    extract_memtrace_features_with_model,
)


def _load_proxy_jsonl(path: Path) -> Tuple[List[str], np.ndarray]:
    texts: List[str] = []
    labels: List[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row.get("text") or ""
            if not isinstance(t, str) or len(t.strip()) < 10:
                continue
            texts.append(t.strip())
            labels.append(int(row.get("label", 0)))
    y = np.array(labels, dtype=np.int64)
    if len(texts) < 4 or len(np.unique(y)) < 2:
        raise RuntimeError(
            f"Need at least 4 rows and two classes in {path}; got n={len(texts)}, classes={np.unique(y)!r}"
        )
    return texts, y


def _extract_features_batched(
    cfg: Dict[str, Any],
    bundle: Dict[str, Any],
    texts: List[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    exp = cfg.get("experiment") or {}
    device = pick_device(exp.get("device"))
    dtype = torch_dtype_from_str(bundle.get("torch_dtype"))
    target = bundle["target_model"]
    tok_id = bundle.get("tokenizer") or target
    print(f"Loading {target} for memTrace features...", file=sys.stderr)
    model, tokenizer = load_causal_lm(
        target, tok_id, device, dtype, attn_implementation="eager"
    )
    chunks: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        Xb = extract_memtrace_features_with_model(
            model, tokenizer, batch, max_length, device, show_progress=False
        )
        chunks.append(Xb)
        print(f"  features {min(i + batch_size, len(texts))}/{len(texts)}", file=sys.stderr)
    del model, tokenizer
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()
    return np.vstack(chunks)


def _rf_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    block = (cfg.get("memtrace_rf_train") or {}) or {}
    return {
        "n_estimators": int(block.get("n_estimators", 250)),
        "max_depth": int(block.get("max_depth", 5)),
        "min_samples_leaf": int(block.get("min_samples_leaf", 2)),
        "random_state": int(block.get("random_state", 42)),
        "feature_batch_size": int(block.get("feature_batch_size", 2)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train memTrace RF on proxy JSONL per Qwen preset.")
    ap.add_argument("--config", type=str, default=str(ROOT / "mia_eval/config/defaults.yaml"))
    ap.add_argument("--experiment", type=str, default="", help="YAML merged over config (e.g. qwen2p5.yaml).")
    ap.add_argument("--set", action="append", default=[], help="Config overrides.")
    ap.add_argument("--proxy-jsonl", type=str, required=True, help="Output of prepare_memtrace_proxy_jsonl.")
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for <preset>_memtrace_rf.joblib + manifest.json",
    )
    ap.add_argument(
        "--preset",
        type=str,
        default="",
        help="Train exactly one active_model key (use with Slurm --array; overrides --presets / --use-yaml-presets).",
    )
    ap.add_argument(
        "--presets",
        type=str,
        default="",
        help="Comma-separated active_model keys (overrides --use-yaml-presets).",
    )
    ap.add_argument(
        "--use-yaml-presets",
        action="store_true",
        help="Use memtrace_train_presets from merged YAML (see qwen2p5.yaml).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, shuffle and keep only this many rows (debug).",
    )
    args = ap.parse_args()

    exp_path = Path(args.experiment) if args.experiment else None
    cfg = load_merged_config(Path(args.config), exp_path)
    cfg = apply_dot_overrides(cfg, args.set)

    proxy_path = Path(args.proxy_jsonl)
    texts, y = _load_proxy_jsonl(proxy_path)
    if args.max_samples > 0 and args.max_samples < len(texts):
        rng = np.random.default_rng(_rf_params(cfg)["random_state"])
        idx = rng.choice(len(texts), size=args.max_samples, replace=False)
        texts = [texts[i] for i in idx]
        y = y[idx]

    if args.preset.strip():
        presets = [args.preset.strip()]
    elif args.presets.strip():
        presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    elif args.use_yaml_presets:
        presets = list(cfg.get("memtrace_train_presets") or [])
    else:
        ap.error(
            "Provide --preset NAME, or --presets a,b,c, or --use-yaml-presets "
            "(with memtrace_train_presets in YAML)."
        )

    models = cfg.get("models") or {}
    for p in presets:
        if p not in models:
            raise KeyError(f"preset {p!r} not in config models. Keys: {list(models.keys())}")

    rf_cfg = _rf_params(cfg)
    batch_size = rf_cfg.pop("feature_batch_size")
    rs = rf_cfg["random_state"]
    mt_len = int((cfg.get("score_sequence") or {}).get("memtrace_max_length", 512))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "proxy_jsonl": str(proxy_path.resolve()),
        "n_rows": len(texts),
        "class_counts": {int(c): int((y == c).sum()) for c in np.unique(y)},
        "presets": presets,
        "rf_params": {k: rf_cfg[k] for k in ("n_estimators", "max_depth", "min_samples_leaf", "random_state")},
        "memtrace_max_length": mt_len,
        "feature_batch_size": batch_size,
        "artifacts": {},
    }

    for preset in presets:
        cfg_run = {**cfg, "active_model": preset}
        bundle = active_model_bundle(cfg_run)
        print(f"\n=== Preset {preset} ===", file=sys.stderr)
        X = _extract_features_batched(
            cfg_run, bundle, texts, mt_len, batch_size=batch_size
        )
        X = _sanitize_feature_matrix(X)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import joblib

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        rf = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            class_weight="balanced",
            random_state=rs,
            n_jobs=-1,
        )
        rf.fit(Xs, y)

        out_path = out_dir / f"{preset}_memtrace_rf.joblib"
        joblib.dump({"scaler": scaler, "rf": rf}, out_path)
        manifest["artifacts"][preset] = str(out_path.resolve())
        print(f"Wrote {out_path}", file=sys.stderr)

    if len(presets) == 1:
        man_path = out_dir / f"{presets[0]}_memtrace_rf_manifest.json"
    else:
        man_path = out_dir / "memtrace_rf_manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
