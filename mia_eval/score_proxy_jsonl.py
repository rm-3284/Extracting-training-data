#!/usr/bin/env python3
"""
Score memTrace proxy JSONL (``text`` / ``label`` / …) with the same MIA scalars as
``mia_eval.mia_gt_pipeline.annotate_samples_mia_gt``: infilling + WBC at
``mia_gt_pipeline.open_model_*`` hyperparameters, and memTrace ``p_member`` from the
preset's ``*_memtrace_rf.joblib``.

Processes texts in **GPU batches** so a 10k+ line proxy file fits in VRAM.

**Sharding** (many short jobs): ``--num-shards N --shard-id K`` (0 ≤ K < N) scores only
that contiguous slice of the input and sets ``proxy_row_index`` on each line so you
can merge with ``python -m mia_eval.merge_proxy_scored_shards``.

Example::

  python -m mia_eval.score_proxy_jsonl \\
    --input data/qwen_memtrace_proxy_train.jsonl \\
    --output data/qwen_memtrace_proxy_train_scored.jsonl \\
    --config mia_eval/config/defaults.yaml \\
    --experiment mia_eval/config/qwen2p5.yaml \\
    --preset qwen25_7b_base \\
    --batch-size 4

Slurm-style (32 parallel GPUs)::

  python -m mia_eval.score_proxy_jsonl ... --num-shards 32 --shard-id ${SLURM_ARRAY_TASK_ID}

Optional ``--calibration-json`` writes per-class medians and a midpoint threshold on
the **proxy** labels (0/1), for infilling / WBC / ``memtrace_p_member`` (orientation:
infilling lower ⇒ more member-like on the proxy positive class).
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

from mia_eval.config_loader import active_model_bundle, apply_dot_overrides, load_merged_config
from mia_eval.model_utils import load_causal_lm, pick_device, torch_dtype_from_str
from mia_eval.mia_gt_pipeline import (
    _memtrace_p_batch,
    _wbc_params_merge,
    resolve_memtrace_rf_joblib,
)
from mia_eval.scoring_infilling import score_texts as infilling_scores
from mia_eval.scoring_memtrace import extract_memtrace_features_with_model
from mia_eval.scoring_wbc import score_texts as wbc_scores


def shard_row_range(n_total: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    """Contiguous half-open interval [start, end) covering all indices when unioned over shards."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    start = (n_total * shard_id) // num_shards
    end = (n_total * (shard_id + 1)) // num_shards
    return start, end


def _load_proxy_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _calibration_block(
    y: np.ndarray,
    inf: np.ndarray,
    wbc: np.ndarray,
    mt: np.ndarray,
) -> Dict[str, Any]:
    """Summaries on proxy labels (0 = non-member proxy, 1 = member proxy)."""
    yb = y.astype(bool)
    n0, n1 = int((~yb).sum()), int(yb.sum())
    out: Dict[str, Any] = {
        "n_proxy_label_0": n0,
        "n_proxy_label_1": n1,
        "metrics": {},
    }
    specs = [
        ("infilling", inf, "lower_is_member_proxy_like"),
        ("wbc", wbc, "higher_is_member_proxy_like"),
        ("memtrace_p_member", mt, "higher_is_member_proxy_like"),
    ]
    for name, s, orient in specs:
        s = np.asarray(s, dtype=np.float64)
        med0 = float(np.median(s[~yb])) if n0 else float("nan")
        med1 = float(np.median(s[yb])) if n1 else float("nan")
        mid = (med0 + med1) / 2.0 if n0 and n1 and np.isfinite(med0) and np.isfinite(med1) else float("nan")
        out["metrics"][name] = {
            "median_on_proxy_label_0": med0,
            "median_on_proxy_label_1": med1,
            "midpoint_between_class_medians": mid,
            "orientation_note": orient,
        }
    return out


def score_proxy_rows(
    cfg: Dict[str, Any],
    bundle: Dict[str, Any],
    model_key: str,
    rows: List[Dict[str, Any]],
    *,
    global_row_indices: List[int],
    batch_size: int,
    primary_only: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rf_path = resolve_memtrace_rf_joblib(cfg, model_key)
    if not rf_path.is_file():
        raise FileNotFoundError(
            f"memTrace RF not found: {rf_path}. Train with train_memtrace_rf or set "
            "mia_gt_pipeline.memtrace_rf_joblib / memtrace_rf_dir."
        )

    block = (cfg.get("mia_gt_pipeline") or {}) or {}
    exp = cfg.get("experiment") or {}
    device = pick_device(exp.get("device"))
    dtype = torch_dtype_from_str(bundle.get("torch_dtype"))
    target = bundle["target_model"]
    ref_name = bundle["reference_model"]
    tok_id = bundle.get("tokenizer") or target
    max_len = int(exp.get("max_length_tokens", 512))

    inf_p = dict(block.get("open_model_infilling") or {"m": 1, "k": 0.2})
    inf_s = dict(block.get("open_model_infilling_sensitivity") or {"m": 5, "k": 0.2})
    wbc_p = dict(block.get("open_model_wbc") or {"min_window": 2, "max_window": 40, "num_windows": 12})
    wbc_s = dict(
        block.get("open_model_wbc_sensitivity") or {"min_window": 2, "max_window": 56, "num_windows": 10}
    )
    mt_len = int(
        block.get("memtrace_max_length")
        or (cfg.get("methods") or {}).get("memtrace", {}).get("max_length")
        or max_len
    )

    if len(global_row_indices) != len(rows):
        raise ValueError("global_row_indices length must match rows length.")

    texts = [str(r.get("text", "")).strip() for r in rows]
    if any(not t for t in texts):
        bad = sum(1 for t in texts if not t)
        raise ValueError(f"{bad} rows have empty text after strip; fix input JSONL.")

    n = len(rows)
    s_inf_p = np.empty(n, dtype=np.float64)
    s_inf_s = np.empty(n, dtype=np.float64) if not primary_only else None
    s_wbc_p = np.empty(n, dtype=np.float64)
    s_wbc_s = np.empty(n, dtype=np.float64) if not primary_only else None
    s_mt = np.empty(n, dtype=np.float64)

    print(f"[score_proxy] Loading target {target} (eager attn)…", file=sys.stderr, flush=True)
    model_t, tok = load_causal_lm(target, tok_id, device, dtype, attn_implementation="eager")

    print(f"[score_proxy] Loading reference {ref_name}…", file=sys.stderr, flush=True)
    model_r, _ = load_causal_lm(ref_name, tok_id, device, dtype)

    bs = max(1, int(batch_size))
    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch_texts = texts[start:end]
        print(f"[score_proxy] batch {start}:{end} / {n}", file=sys.stderr, flush=True)

        s_inf_p[start:end] = infilling_scores(model_t, tok, batch_texts, inf_p, max_length=max_len)
        if not primary_only:
            assert s_inf_s is not None
            s_inf_s[start:end] = infilling_scores(model_t, tok, batch_texts, inf_s, max_length=max_len)

        s_wbc_p[start:end] = wbc_scores(
            model_t,
            model_r,
            tok,
            batch_texts,
            device,
            _wbc_params_merge(cfg, wbc_p),
            max_length=max_len,
        )
        if not primary_only:
            assert s_wbc_s is not None
            s_wbc_s[start:end] = wbc_scores(
                model_t,
                model_r,
                tok,
                batch_texts,
                device,
                _wbc_params_merge(cfg, wbc_s),
                max_length=max_len,
            )

        X_mt = extract_memtrace_features_with_model(
            model_t, tok, batch_texts, mt_len, device, show_progress=False
        )
        s_mt[start:end] = _memtrace_p_batch(X_mt, rf_path)

    del model_t, model_r
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()

    s_sel: np.ndarray | None = None
    sel_block = block.get("select") or {}
    if sel_block.get("enabled"):
        from mia_eval.scoring_morris2025_select import compute_select_last_layer_scores

        base_id = str(sel_block.get("base_model") or "").strip()
        if not base_id:
            raise ValueError(
                "mia_gt_pipeline.select.enabled requires mia_gt_pipeline.select.base_model."
            )
        sel_max_len = int(sel_block.get("max_length", max_len))
        s_sel = compute_select_last_layer_scores(
            cfg,
            bundle,
            texts,
            base_model_id=base_id,
            tokenizer_id=tok_id,
            max_length=min(max_len, sel_max_len),
            n_monte_carlo=int(sel_block.get("n_monte_carlo", 4096)),
            random_state=int(sel_block.get("random_state", 42)),
            batch_size=int(sel_block.get("batch_size", 1)),
        )

    out_rows: List[Dict[str, Any]] = []
    labels: List[int] = []
    for i, r in enumerate(rows):
        lab = r.get("label")
        if lab is not None:
            try:
                labels.append(int(lab))
            except (TypeError, ValueError):
                labels.append(-1)
        else:
            labels.append(-1)

        primary: Dict[str, Any] = {
            "infilling": float(s_inf_p[i]),
            "wbc": float(s_wbc_p[i]),
            "memtrace_p_member": float(s_mt[i]),
        }
        if primary_only:
            # Same schema as ``annotate_samples_mia_gt``; infilling/WBC not recomputed at sens HPs.
            sensitivity = {
                "infilling": float(s_inf_p[i]),
                "wbc": float(s_wbc_p[i]),
                "memtrace_p_member": float(s_mt[i]),
            }
        else:
            assert s_inf_s is not None and s_wbc_s is not None
            sensitivity = {
                "infilling": float(s_inf_s[i]),
                "wbc": float(s_wbc_s[i]),
                "memtrace_p_member": float(s_mt[i]),
            }
        if s_sel is not None:
            v = float(s_sel[i])
            primary["select_alignment_mc"] = v
            sensitivity["select_alignment_mc"] = v

        gix = int(global_row_indices[i])
        doc = {
            **r,
            "proxy_row_index": gix,
            "mia_gt_primary": primary,
            "mia_gt_sensitivity": sensitivity,
        }
        out_rows.append(doc)

    cal_meta: Dict[str, Any] = {
        "proxy_row_index_min": int(min(global_row_indices)) if global_row_indices else -1,
        "proxy_row_index_max": int(max(global_row_indices)) if global_row_indices else -1,
        "memtrace_rf": str(rf_path),
        "primary_only": primary_only,
        "mia_gt_sensitivity_note": (
            "infilling and wbc duplicate primary when --primary-only."
            if primary_only
            else "full sensitivity HPs from mia_gt_pipeline."
        ),
        "memtrace_max_length": mt_len,
        "max_length_infilling_wbc": max_len,
    }
    y_arr = np.asarray(labels, dtype=np.int64)
    if y_arr.size and np.all((y_arr == 0) | (y_arr == 1)) and len(np.unique(y_arr)) == 2:
        cal_meta["calibration_on_proxy_labels_primary"] = _calibration_block(
            y_arr, s_inf_p, s_wbc_p, s_mt
        )
    else:
        cal_meta["calibration_on_proxy_labels_primary"] = None
        cal_meta["calibration_note"] = "Skipped: every row needs integer label 0 or 1."

    return out_rows, cal_meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score proxy JSONL with infilling, WBC, memTrace (mia_gt_primary-style)."
    )
    ap.add_argument("--config", type=str, default=str(ROOT / "mia_eval/config/defaults.yaml"))
    ap.add_argument(
        "--experiment",
        type=str,
        default=str(ROOT / "mia_eval/config/qwen2p5.yaml"),
        help="YAML merged over config (Qwen presets + mia_gt_pipeline).",
    )
    ap.add_argument("--set", action="append", default=[], help="Overrides, e.g. active_model=qwen25_7b_base")
    ap.add_argument("--preset", type=str, default="", help="Sets active_model (must exist under models:).")
    ap.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "data/qwen_memtrace_proxy_train.jsonl"),
        help="Proxy JSONL with at least a text field.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSONL (default: <input> with _scored before .jsonl).",
    )
    ap.add_argument("--batch-size", type=int, default=4, help="Texts per forward batch for scoring.")
    ap.add_argument(
        "--primary-only",
        action="store_true",
        help="Only primary infilling/WBC HPs (skip sensitivity infilling/WBC passes).",
    )
    ap.add_argument("--max-rows", type=int, default=0, help="If >0, cap this shard to at most N rows (after sharding).")
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split input into N contiguous slices; combine with --shard-id for parallel jobs.",
    )
    ap.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="0-based shard index; must satisfy 0 <= shard-id < num-shards when num-shards > 1.",
    )
    ap.add_argument(
        "--offset",
        type=int,
        default=-1,
        help="If >=0, ignore sharding: score rows [offset, offset+limit) (0-based; use with --limit).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="With --offset, number of rows to score (required when offset >= 0).",
    )
    ap.add_argument(
        "--calibration-json",
        type=str,
        default="",
        help="Optional path to write calibration summary JSON (proxy label medians).",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    cfg = load_merged_config(Path(args.config), Path(args.experiment))
    cfg = apply_dot_overrides(cfg, args.set)
    if args.preset:
        cfg["active_model"] = args.preset

    model_key = str(cfg.get("active_model", ""))
    if not model_key:
        raise SystemExit("active_model is empty; set --preset or config.")

    bundle = active_model_bundle(cfg)
    all_rows = _load_proxy_rows(inp)
    n_full = len(all_rows)
    if n_full == 0:
        raise SystemExit(f"No rows in {inp}")

    use_offset = int(args.offset) >= 0
    if use_offset:
        off = int(args.offset)
        lim = int(args.limit)
        if lim <= 0:
            raise SystemExit("--limit must be positive when --offset is set.")
        start, end = off, min(off + lim, n_full)
        if start >= n_full:
            raise SystemExit(f"--offset {off} is past end of file ({n_full} rows).")
        shard_meta = {"mode": "offset_limit", "offset": off, "limit_requested": lim, "row_range": [start, end]}
    else:
        ns = int(args.num_shards)
        sid = int(args.shard_id)
        if ns < 1:
            raise SystemExit("--num-shards must be >= 1.")
        if sid < 0 or sid >= ns:
            raise SystemExit(f"--shard-id must satisfy 0 <= shard-id < num-shards ({ns}).")
        start, end = shard_row_range(n_full, sid, ns)
        shard_meta = {"mode": "shard", "num_shards": ns, "shard_id": sid, "row_range": [start, end]}

    rows = all_rows[start:end]
    global_indices = list(range(start, end))

    if args.max_rows and args.max_rows > 0:
        cap = int(args.max_rows)
        rows = rows[:cap]
        global_indices = global_indices[:cap]

    if not rows:
        raise SystemExit("No rows in this shard/range (empty slice).")

    if args.output:
        out = Path(args.output)
    elif use_offset:
        out = inp.with_name(f"{inp.stem}_scored_off{start}_len{len(rows)}.jsonl")
    elif int(args.num_shards) > 1:
        ns = int(args.num_shards)
        sid = int(args.shard_id)
        out = inp.with_name(f"{inp.stem}_scored_shard{sid:04d}_of_{ns:04d}.jsonl")
    else:
        out = inp.with_name(inp.stem + "_scored.jsonl")

    out_rows, cal_meta = score_proxy_rows(
        cfg,
        bundle,
        model_key,
        rows,
        global_row_indices=global_indices,
        batch_size=args.batch_size,
        primary_only=args.primary_only,
    )

    gi_sorted = sorted(global_indices)
    covers_full = len(gi_sorted) == n_full and gi_sorted == list(range(n_full))
    cal_meta["calibration_scope"] = (
        "full_input_file" if covers_full else "slice_only_merge_shards_for_global_calibration"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for doc in out_rows:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    summary = {
        "input": str(inp.resolve()),
        "output": str(out.resolve()),
        "active_model": model_key,
        "n_rows_input_file": n_full,
        "n_rows_scored_this_job": len(out_rows),
        "shard": shard_meta,
        **cal_meta,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.calibration_json:
        cpath = Path(args.calibration_json)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote {cpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
