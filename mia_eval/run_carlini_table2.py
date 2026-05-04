#!/usr/bin/env python3
"""
Carlini et al. (2021) **Table 2-style** cheap filters on arbitrary text JSONL.

Scores each line's ``text`` with:
  - target perplexity,
  - reference (smaller) model perplexity,
  - zlib compressed byte length,
  - lowercase target perplexity,
  - sliding-window minimum perplexity,

then forms the same ratios as ``Extracting-Training-Data-from-Large-Langauge-Models/run_carlini.py``:
``Small``, ``zlib``, ``Lowercase``, ``Window``, plus raw ``Perplexity``.

If every input row has integer ``label`` (0/1), also reports **precision@k** for each metric
(as in the original script). Qwen presets use ``mia_eval/config/qwen2p5.yaml`` + ``active_model``.

Example::

  python -m mia_eval.run_carlini_table2 \\
    --config mia_eval/config/defaults.yaml \\
    --experiment mia_eval/config/qwen2p5.yaml \\
    --set active_model=qwen25_7b_base \\
    --input mia_eval_outputs/qwen25_7b_base/samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_eval.config_loader import active_model_bundle, load_merged_config, apply_dot_overrides
from mia_eval.evaluation_common import jsonable as _jsonable
from mia_eval.model_utils import load_causal_lm, pick_device, torch_dtype_from_str


def _perplexity(
    text: str,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
) -> float:
    """Sequence perplexity exp(mean CE), matching ``run_carlini.py``."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return float(torch.exp(loss).item())


def _perplexity_window(
    text: str,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    window: int,
) -> float:
    ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.squeeze(0).to(device)
    if ids.shape[0] < window:
        return _perplexity(text, model, tokenizer, device, max_length)
    min_ppl = float("inf")
    with torch.no_grad():
        for start in range(int(ids.shape[0]) - window):
            window_ids = ids[start : start + window].unsqueeze(0)
            loss = model(window_ids, labels=window_ids).loss
            min_ppl = min(min_ppl, float(torch.exp(loss).item()))
    return min_ppl


def _aggregate_over_samples(arr: np.ndarray) -> Dict[str, float]:
    """When labels are absent, summarize each metric across all texts (not in [0,1] in general)."""
    if arr.size == 0:
        return {"mean": float("nan"), "sum": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "sum": float(np.sum(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def precision_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    lower_better: bool,
) -> float:
    if k <= 0 or k > len(scores):
        return float("nan")
    if lower_better:
        top_k_idx = np.argsort(scores)[:k]
    else:
        top_k_idx = np.argsort(scores)[::-1][:k]
    return float(labels[top_k_idx].mean())


def load_texts_and_labels(path: Path) -> Tuple[List[str], List[str], Optional[np.ndarray]]:
    texts: List[str] = []
    sources: List[str] = []
    labels_list: List[int] = []
    has_all_labels = True
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row.get("text")
            if not isinstance(t, str) or not t.strip():
                continue
            texts.append(t.strip())
            sources.append(str(row.get("source", "")))
            if "label" in row and row["label"] is not None:
                try:
                    labels_list.append(int(row["label"]))
                except (TypeError, ValueError):
                    has_all_labels = False
                    labels_list.append(0)
            else:
                has_all_labels = False
                labels_list.append(0)
    y = np.array(labels_list, dtype=np.int64) if has_all_labels and labels_list else None
    if not has_all_labels:
        y = None
    return texts, sources, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Carlini Table 2-style scores for any causal LM preset.")
    ap.add_argument("--config", type=str, default=str(ROOT / "mia_eval/config/defaults.yaml"))
    ap.add_argument("--experiment", type=str, default="", help="YAML merged over config (e.g. qwen2p5.yaml).")
    ap.add_argument("--set", action="append", default=[], help="Override e.g. active_model=qwen25_7b_base")
    ap.add_argument(
        "--input",
        type=str,
        default="",
        help="JSONL with ``text`` per line (and optional ``label``). Default: output_dir/active_model/samples.jsonl",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="JSON summary path. Default: same dir as input → carlini_table2.json",
    )
    ap.add_argument(
        "--scores-jsonl",
        type=str,
        default="",
        help="If set, write one JSON object per line with raw scores and ratios.",
    )
    ap.add_argument("--window", type=int, default=50, help="Sliding window size for Window metric.")
    ap.add_argument(
        "--precision-k",
        type=str,
        default="10,50,100",
        help="Comma-separated k values for precision@k (only if labels present).",
    )
    args = ap.parse_args()

    cfg = load_merged_config(Path(args.config), Path(args.experiment) if args.experiment else None)
    cfg = apply_dot_overrides(cfg, args.set)
    model_key = cfg.get("active_model", "gpt_neo_2p7")
    out_root = Path(cfg.get("experiment", {}).get("output_dir", "mia_eval_outputs"))
    run_dir = out_root / model_key

    in_path = Path(args.input) if args.input else run_dir / "samples.jsonl"
    if not in_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")

    out_path = Path(args.output) if args.output else in_path.parent / "carlini_table2.json"
    scores_jsonl_path = Path(args.scores_jsonl) if args.scores_jsonl else None

    exp = cfg.get("experiment") or {}
    max_len = int(exp.get("max_length_tokens", 256))
    bundle = active_model_bundle(cfg)
    target = bundle["target_model"]
    ref_name = bundle["reference_model"]
    tok_id = bundle.get("tokenizer") or target

    device = pick_device(exp.get("device"))
    dtype = torch_dtype_from_str(bundle.get("torch_dtype"))

    texts, sources, y = load_texts_and_labels(in_path)
    if not texts:
        raise RuntimeError(f"No texts loaded from {in_path}")

    ks = [int(x.strip()) for x in args.precision_k.split(",") if x.strip().isdigit()]
    ks = [k for k in ks if k <= len(texts)]

    print(f"Loaded {len(texts)} texts from {in_path}", file=sys.stderr)
    if y is None:
        print(
            "No per-row ``label`` (or incomplete) — skipping precision@k; "
            "reporting mean/sum/std over all samples per metric (Carlini scores are not all in [0,1]).",
            file=sys.stderr,
        )
    else:
        print(f"Labels present: {int(y.sum())} positive, {int((1 - y).sum())} negative", file=sys.stderr)

    print(f"Loading target {target} …", file=sys.stderr)
    model_t, tok = load_causal_lm(target, tok_id, device, dtype)
    print(f"Loading reference {ref_name} …", file=sys.stderr)
    model_r, _ = load_causal_lm(ref_name, tok_id, device, dtype)

    ppl_t: List[float] = []
    ppl_r: List[float] = []
    ppl_lower: List[float] = []
    ppl_window: List[float] = []
    zlib_bytes: List[int] = []

    for i, text in enumerate(texts):
        if i % 20 == 0:
            print(f"  {i}/{len(texts)} …", file=sys.stderr)
        ppl_t.append(_perplexity(text, model_t, tok, device, max_len))
        ppl_r.append(_perplexity(text, model_r, tok, device, max_len))
        ppl_lower.append(_perplexity(text.lower(), model_t, tok, device, max_len))
        ppl_window.append(
            _perplexity_window(text, model_t, tok, device, max_len, int(args.window))
        )
        zlib_bytes.append(len(zlib.compress(text.encode("utf-8"))))

    del model_t, model_r
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ppl_t_a = np.array(ppl_t, dtype=np.float64)
    ppl_r_a = np.array(ppl_r, dtype=np.float64)
    ppl_lower_a = np.array(ppl_lower, dtype=np.float64)
    ppl_window_a = np.array(ppl_window, dtype=np.float64)
    zlib_a = np.array(zlib_bytes, dtype=np.float64)

    eps = 1e-8
    small = np.log(ppl_t_a + eps) / np.log(ppl_r_a + eps)
    zlib_ratio = np.log(ppl_t_a + eps) / np.log(zlib_a + eps)
    lower_ratio = np.log(ppl_t_a + eps) / np.log(ppl_lower_a + eps)

    metrics_def: List[Tuple[str, np.ndarray, bool]] = [
        ("Perplexity", ppl_t_a, True),
        ("Small", small, True),
        ("zlib", zlib_ratio, True),
        ("Lowercase", lower_ratio, True),
        ("Window", ppl_window_a, True),
    ]

    results: Dict[str, Any] = {
        "model_key": model_key,
        "target_model": target,
        "reference_model": ref_name,
        "input": str(in_path.resolve()),
        "n_samples": len(texts),
        "max_length_tokens": max_len,
        "window": int(args.window),
        "has_labels": y is not None,
        "metrics": {},
    }
    if y is None:
        results["no_label_summary_note"] = (
            "Carlini Table-2 metrics use heterogeneous scales (e.g. perplexity is unbounded; "
            "Small/zlib/Lowercase are log ratios). They are not probabilities in [0,1] "
            "(unlike WBC in mia_eval). With no labels, ``aggregate_over_samples`` substitutes "
            "for precision@k."
        )

    print("\nMetric (lower score = more memorized-y under Carlini convention)\n", file=sys.stderr)
    for name, arr, lower_better in metrics_def:
        row: Dict[str, Any] = {"lower_is_more_memorized_like": lower_better}
        row["aggregate_over_samples"] = _aggregate_over_samples(arr)
        if y is not None and ks:
            row["precision_at_k"] = {}
            for k in ks:
                row["precision_at_k"][f"@{k}"] = precision_at_k(arr, y, k, lower_better=lower_better)
        results["metrics"][name] = row
        agg = row["aggregate_over_samples"]
        if y is not None and ks:
            parts = [f"P@{k}={row['precision_at_k'][f'@{k}']:.3f}" for k in ks]
            print(f"  {name:<12} " + "  ".join(parts), file=sys.stderr)
        else:
            print(
                f"  {name:<12}  mean={agg['mean']:.6g}  sum={agg['sum']:.6g}  std={agg['std']:.6g}  "
                f"(n={len(texts)}; no P@k)",
                file=sys.stderr,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=2)
    print(f"Wrote {out_path}", file=sys.stderr)

    if scores_jsonl_path is not None:
        scores_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_jsonl_path, "w", encoding="utf-8") as f:
            for i in range(len(texts)):
                doc = {
                    "i": i,
                    "source": sources[i],
                    "perplexity_target": float(ppl_t_a[i]),
                    "perplexity_reference": float(ppl_r_a[i]),
                    "perplexity_lower": float(ppl_lower_a[i]),
                    "perplexity_window_min": float(ppl_window_a[i]),
                    "zlib_compressed_bytes": int(zlib_bytes[i]),
                    "small_log_ratio": float(small[i]),
                    "zlib_log_ratio": float(zlib_ratio[i]),
                    "lowercase_log_ratio": float(lower_ratio[i]),
                }
                if y is not None:
                    doc["label"] = int(y[i])
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"Wrote {scores_jsonl_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
