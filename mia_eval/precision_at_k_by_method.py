#!/usr/bin/env python3
"""
Compute precision@k **separately for each generation method** without merging samples.

Supports both:
1. **Carlini extraction** (top_k, top_k_internet, temperature_decay, …) and **memorization_detection**
   sources (``memorization_baseline``, ``memorization_risk_fast``, …) when enabled in YAML.
2. **Decoding defense methods** (fast, slow, wbc) — defense-based generations to compare against Carlini

For each model and each Carlini metric (Perplexity, Small, zlib, etc.), compute P@k
broken down by generation source. This allows comparison of extraction/defense efficacy
across different sampling strategies.

**Prerequisites:** 

For Carlini methods: Run ``generate_carlini_scores_batch.py`` first to generate per-sample 
Carlini scores (carlini_table2_scores.jsonl) in each model directory. This requires GPU 
and takes time but only needs to be done once.

For Decoding methods: Run ``generate_decoding_samples.py`` to generate samples from each
decoding method, then ``generate_carlini_scores_batch.py`` to score them.

Usage:

**Carlini extraction:**
    # 1. Generate Carlini scores for all models (GPU required, one-time)
    python -m mia_eval.generate_carlini_scores_batch \\
        --scan-glob 'mia_eval_outputs/carlini_extract/*/samples_labeled.jsonl' \\
        --config mia_eval/config/defaults.yaml

    # 2. Compute precision@k by method (CPU, fast)
    python -m mia_eval.precision_at_k_by_method \\
        --scan-glob 'mia_eval_outputs/carlini_extract/*/samples_labeled.jsonl' \\
        --output precision_by_method_all.json \\
        --to-markdown precision_by_method_all.md

**Decoding methods:**
    # 1. Generate samples from defense methods (GPU required, fast+slow+wbc only)
    python -m mia_eval.generate_decoding_samples \\
        --models gpt_neo_2p7 pythia_2p8 \\
        --n-examples 100 \\
        --base-output mia_eval_outputs/decoding_samples

    # 2. Generate Carlini scores for those samples (GPU required)
    python -m mia_eval.generate_carlini_scores_batch \\
        --scan-glob 'mia_eval_outputs/decoding_samples/*/samples.jsonl' \\
        --config mia_eval/config/defaults.yaml

    # 3. Compute precision@k by defense method (CPU, fast)
    python -m mia_eval.precision_at_k_by_method \\
        --scan-glob 'mia_eval_outputs/decoding_samples/*/samples.jsonl' \\
        --output decoding_precision_by_method.json \\
        --to-markdown decoding_precision_by_method.md

**Single model:**
    python -m mia_eval.precision_at_k_by_method \\
        --model dclm_7b_base \\
        --output /tmp/test.json \\
        --to-markdown /tmp/test.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_eval.carlini_sample_sources import GENERATION_SOURCES_ORDER
from mia_eval.evaluation_common import precision_at_k as _precision_at_k

# Known ``source`` tags from ``mia_eval.generation`` (Carlini + memorization_detection block).
CARLINI_SOURCES = list(GENERATION_SOURCES_ORDER)
DECODING_SOURCES = ["fast", "slow", "wbc"]

# Carlini metrics: (display_name, json_field, lower_is_better)
CARLINI_METRICS: List[tuple[str, str, bool]] = [
    ("Perplexity", "perplexity_target", True),
    ("Small", "small_log_ratio", True),
    ("zlib", "zlib_log_ratio", True),
    ("Lowercase", "lowercase_log_ratio", True),
    ("Window", "perplexity_window_min", True),
]


def _detect_sources(samples: List[Dict[str, Any]]) -> tuple[List[str], str]:
    """Detect which sources are present and return (sources, source_type)."""
    all_sources = set()
    for sample in samples:
        source = sample.get("source")
        if source:
            all_sources.add(source)
    
    # Classify as Carlini or Decoding
    carlini_count = len(all_sources & set(CARLINI_SOURCES))
    decoding_count = len(all_sources & set(DECODING_SOURCES))
    
    if decoding_count >= carlini_count:
        # Decoding methods
        sources = [s for s in DECODING_SOURCES if s in all_sources]
        source_type = "decoding"
    else:
        # Carlini extraction methods
        sources = [s for s in CARLINI_SOURCES if s in all_sources]
        source_type = "carlini"
    
    return sources, source_type


def _load_samples_and_labels_and_scores(
    samples_path: Path, scores_path: Optional[Path] = None
) -> tuple[List[Dict[str, Any]], np.ndarray, Optional[List[Dict[str, Any]]]]:
    """Load samples, labels from samples_labeled.jsonl, and Carlini scores if available.
    
    Args:
        samples_path: Path to samples_labeled.jsonl (has text, source, label)
        scores_path: Path to carlini_table2_scores.jsonl (optional, has metric scores)
    
    Returns:
        (samples, labels, scores) where scores is None if not provided
    """
    samples = []
    labels = []
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(row)
            label = row.get("label")
            if label is None:
                labels.append(np.nan)
            else:
                try:
                    labels.append(float(label))
                except (TypeError, ValueError):
                    labels.append(np.nan)
    
    scores = None
    if scores_path and scores_path.is_file():
        scores = []
        with open(scores_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                scores.append(json.loads(line))
    
    return samples, np.array(labels, dtype=np.float64), scores


def compute_precision_by_method(
    samples: List[Dict[str, Any]],
    labels: np.ndarray,
    scores: Optional[List[Dict[str, Any]]] = None,
    ks: Sequence[int] = (10, 50, 100),
) -> Dict[str, Dict[str, Any]]:
    """
    Compute precision@k for each metric, split by generation method.
    
    Args:
        samples: List of sample dicts (with 'source' field)
        labels: Array of labels (0/1 or NaN)
        scores: Optional list of score dicts from carlini_table2_scores.jsonl
        ks: Precision@k values to compute
    
    Returns:
        {
            "metric_name": {
                "metric_info": {
                    "lower_is_more_memorized_like": bool,
                    ...
                },
                "top_k": {"@10": float, "@50": float, ...},
                "top_k_internet": {...},
                "temperature_decay": {...},
            },
            ...
        }
    """
    expected_sources, source_type = _detect_sources(samples)
    
    print(f"[info] detected source type: {source_type}, sources: {expected_sources}", file=sys.stderr)
    
    if scores is None:
        return _compute_precision_from_samples(samples, labels, expected_sources, ks)
    else:
        return _compute_precision_from_scores(samples, labels, scores, expected_sources, ks)


def _compute_precision_from_scores(
    samples: List[Dict[str, Any]],
    labels: np.ndarray,
    scores: List[Dict[str, Any]],
    expected_sources: List[str],
    ks: Sequence[int] = (10, 50, 100),
) -> Dict[str, Dict[str, Any]]:
    """Compute precision@k using per-sample Carlini scores."""
    results: Dict[str, Dict[str, Any]] = {}
    
    if len(scores) != len(samples):
        raise ValueError(f"Mismatch: {len(samples)} samples vs {len(scores)} score rows")
    
    # Group sample indices by source
    samples_by_source: Dict[str, List[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        source = sample.get("source", "unknown")
        samples_by_source[source].append(i)
    
    print(f"[info] samples by source: {dict(samples_by_source)}", file=sys.stderr)
    
    for metric_name, metric_field, lower_is_better in CARLINI_METRICS:
        results[metric_name] = {
            "metric_info": {
                "field": metric_field,
                "lower_is_more_memorized_like": lower_is_better,
            }
        }
        
        # Extract scores for this metric from scores list
        metric_scores = np.full(len(scores), np.nan, dtype=np.float64)
        for i, score_row in enumerate(scores):
            v = score_row.get(metric_field)
            if v is not None:
                try:
                    metric_scores[i] = float(v)
                except (TypeError, ValueError):
                    pass
        
        # For each source, compute P@k
        for source in expected_sources:
            indices = np.array(samples_by_source.get(source, []))
            if len(indices) == 0:
                results[metric_name][source] = {"error": "no samples"}
                continue
            
            source_scores = metric_scores[indices]
            source_labels = labels[indices]
            
            # Skip if no valid scores or labels
            if not np.any(np.isfinite(source_scores)) or not np.any(np.isfinite(source_labels)):
                results[metric_name][source] = {"error": "no valid scores or labels"}
                continue
            
            pk_result: Dict[str, float] = {}
            for k in ks:
                try:
                    pk = _precision_at_k(source_scores, source_labels, k, lower_better=lower_is_better)
                    pk_result[f"@{k}"] = float(pk) if not np.isnan(pk) else None
                except Exception as e:
                    pk_result[f"@{k}"] = None
            
            results[metric_name][source] = {
                "n_samples": int(len(indices)),
                "n_labeled": int(np.sum(np.isfinite(source_labels))),
                "precision_at_k": pk_result,
            }
    
    return results


def _compute_precision_from_samples(
    samples: List[Dict[str, Any]],
    labels: np.ndarray,
    expected_sources: List[str],
    ks: Sequence[int] = (10, 50, 100),
) -> Dict[str, Dict[str, Any]]:
    """Compute precision@k using scores from samples (fallback if no separate scores file)."""
    results: Dict[str, Dict[str, Any]] = {}
    
    # Group samples by source
    samples_by_source: Dict[str, List[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        source = sample.get("source", "unknown")
        samples_by_source[source].append(i)
    
    print(f"[info] samples by source: {dict(samples_by_source)}", file=sys.stderr)
    
    for metric_name, metric_field, lower_is_better in CARLINI_METRICS:
        results[metric_name] = {
            "metric_info": {
                "field": metric_field,
                "lower_is_more_memorized_like": lower_is_better,
            }
        }
        
        # Extract scores for this metric
        scores = np.full(len(samples), np.nan, dtype=np.float64)
        for i, sample in enumerate(samples):
            v = sample.get(metric_field)
            if v is not None:
                try:
                    scores[i] = float(v)
                except (TypeError, ValueError):
                    pass
        
        # For each source, compute P@k
        for source in expected_sources:
            indices = np.array(samples_by_source.get(source, []))
            if len(indices) == 0:
                results[metric_name][source] = {"error": "no samples"}
                continue
            
            source_scores = scores[indices]
            source_labels = labels[indices]
            
            # Skip if no valid scores or labels
            if not np.any(np.isfinite(source_scores)) or not np.any(np.isfinite(source_labels)):
                results[metric_name][source] = {"error": "no valid scores or labels"}
                continue
            
            pk_result: Dict[str, float] = {}
            for k in ks:
                try:
                    pk = _precision_at_k(source_scores, source_labels, k, lower_better=lower_is_better)
                    pk_result[f"@{k}"] = float(pk) if not np.isnan(pk) else None
                except Exception as e:
                    pk_result[f"@{k}"] = None
            
            results[metric_name][source] = {
                "n_samples": int(len(indices)),
                "n_labeled": int(np.sum(np.isfinite(source_labels))),
                "precision_at_k": pk_result,
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        help="Single model name (e.g. dclm_7b_base). Looks for samples in mia_eval_outputs/carlini_extract/{model}/samples_labeled.jsonl",
    )
    parser.add_argument(
        "--samples-jsonl",
        type=Path,
        help="Direct path to samples_labeled.jsonl file.",
    )
    parser.add_argument(
        "--scan-glob",
        type=str,
        help="Glob pattern to find all samples_labeled.jsonl (e.g. 'mia_eval_outputs/carlini_extract/*/samples_labeled.jsonl')",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="10,50,100",
        help="Comma-separated k values for precision@k (default: 10,50,100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file.",
    )
    parser.add_argument(
        "--to-markdown",
        type=Path,
        help="(Optional) also write a Markdown table to this file.",
    )
    parser.add_argument(
        "--to-csv",
        type=Path,
        help="(Optional) also write a CSV to this file.",
    )
    
    args = parser.parse_args()
    
    ks = [int(k.strip()) for k in args.ks.split(",")]
    
    # Determine which files to process
    files_to_process: List[tuple[str, Path]] = []
    
    if args.model:
        model_name = args.model
        sample_path = Path(f"mia_eval_outputs/carlini_extract/{model_name}/samples_labeled.jsonl")
        if not sample_path.exists():
            print(f"ERROR: {sample_path} not found", file=sys.stderr)
            sys.exit(1)
        files_to_process.append((model_name, sample_path))
    elif args.samples_jsonl:
        model_name = args.samples_jsonl.parent.name
        files_to_process.append((model_name, args.samples_jsonl))
    elif args.scan_glob:
        for fpath in sorted(glob.glob(args.scan_glob)):
            p = Path(fpath)
            model_name = p.parent.name
            files_to_process.append((model_name, p))
    else:
        parser.error("Specify one of: --model, --samples-jsonl, or --scan-glob")
    
    if not files_to_process:
        print("ERROR: no files to process", file=sys.stderr)
        sys.exit(1)
    
    # Process each model
    all_results: Dict[str, Dict[str, Any]] = {}
    
    for model_name, sample_path in files_to_process:
        print(f"[processing] {model_name} from {sample_path}", file=sys.stderr)
        try:
            # Try to find carlini_table2_scores.jsonl in the same directory
            scores_path = sample_path.parent / "carlini_table2_scores.jsonl"
            if not scores_path.exists():
                print(f"[info]   scores file not found at {scores_path}", file=sys.stderr)
                print(f"[info]   will compute precision@k without per-sample scores", file=sys.stderr)
                scores_path = None
            else:
                print(f"[info]   using scores from {scores_path}", file=sys.stderr)
            
            samples, labels, scores = _load_samples_and_labels_and_scores(sample_path, scores_path)
            pk_by_method = compute_precision_by_method(samples, labels, scores=scores, ks=ks)
            all_results[model_name] = pk_by_method
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            all_results[model_name] = {"error": str(e)}
    
    # Write JSON output
    output_dict = {
        "schema": "precision_at_k_by_method.v1",
        "ks": ks,
        "models": all_results,
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)
    print(f"[done] wrote {args.output}", file=sys.stderr)
    
    # Optionally write Markdown table
    if args.to_markdown:
        _write_markdown_table(output_dict, args.to_markdown, ks)
    
    # Optionally write CSV
    if args.to_csv:
        _write_csv_table(output_dict, args.to_csv, ks)


def _write_markdown_table(output_dict: Dict[str, Any], output_path: Path, ks: Sequence[int]):
    """Write a Markdown table with results grouped by model and method."""
    models = output_dict.get("models", {})
    if not models:
        return
    
    # Detect source type from first model
    source_type = "unknown"
    expected_sources = []
    for model_data in models.values():
        if not isinstance(model_data, dict) or "error" in model_data:
            continue
        # Pick first metric to find sources
        for metric_data in model_data.values():
            if isinstance(metric_data, dict) and "metric_info" not in metric_data:
                expected_sources = [k for k in metric_data.keys() if k not in ["metric_info", "error"]]
                if expected_sources:
                    # Classify
                    decoding_count = len(set(expected_sources) & set(DECODING_SOURCES))
                    carlini_count = len(set(expected_sources) & set(CARLINI_SOURCES))
                    source_type = "decoding" if decoding_count >= carlini_count else "carlini"
                    break
            if expected_sources:
                break
        if expected_sources:
            break
    
    # Header rows
    lines = [f"# Precision@k by Generation Method ({source_type})\n"]
    
    for model_name in sorted(models.keys()):
        model_result = models[model_name]
        if "error" in model_result:
            lines.append(f"\n## {model_name}\nError: {model_result['error']}\n")
            continue
        
        lines.append(f"\n## {model_name}\n")
        
        # For each metric
        for metric_name in ["Perplexity", "Small", "zlib", "Lowercase", "Window"]:
            if metric_name not in model_result:
                continue
            metric_data = model_result[metric_name]
            
            lines.append(f"\n### {metric_name}\n")
            lines.append("| Method | N | P@" + " | P@".join(str(k) for k in ks) + " |\n")
            lines.append("|" + " --- |" * (3 + len(ks)) + "\n")
            
            for source in expected_sources:
                if source not in metric_data:
                    continue
                source_data = metric_data[source]
                if "error" in source_data:
                    lines.append(f"| {source} | {source_data['error']} |\n")
                    continue
                
                n = source_data.get("n_samples", "?")
                pk = source_data.get("precision_at_k", {})
                pk_vals = [f"{pk.get(f'@{k}', 'N/A'):.3f}" if pk.get(f'@{k}') is not None else "N/A" for k in ks]
                lines.append(f"| {source} | {n} | " + " | ".join(pk_vals) + " |\n")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[done] wrote markdown {output_path}", file=sys.stderr)


def _write_csv_table(output_dict: Dict[str, Any], output_path: Path, ks: Sequence[int]):
    """Write a CSV with results: model, metric, method, k, precision."""
    models = output_dict.get("models", {})
    
    rows = []
    for model_name in sorted(models.keys()):
        model_result = models[model_name]
        if "error" in model_result:
            continue
        
        for metric_name in ["Perplexity", "Small", "zlib", "Lowercase", "Window"]:
            if metric_name not in model_result:
                continue
            metric_data = model_result[metric_name]
            
            # Find all sources in this metric
            sources = [k for k in metric_data.keys() if k not in ["metric_info", "error"]]
            
            for source in sources:
                if source not in metric_data:
                    continue
                source_data = metric_data[source]
                if "error" in source_data:
                    continue
                
                pk = source_data.get("precision_at_k", {})
                for k in ks:
                    precision = pk.get(f"@{k}")
                    rows.append({
                        "model": model_name,
                        "metric": metric_name,
                        "method": source,
                        "k": k,
                        "precision": precision,
                        "n_samples": source_data.get("n_samples"),
                    })
    
    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "metric", "method", "k", "precision", "n_samples"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[done] wrote CSV {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
