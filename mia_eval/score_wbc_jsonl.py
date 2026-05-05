#!/usr/bin/env python3
"""
Score **WBC only** for each ``text`` line in a JSONL (no infilling, no memTrace).

Useful for exploring WBC score distributions on **large** proxy or sample files with a single
preset (defaults to **Distil** Qwen instruct). Processes input in **line chunks** so memory stays
bounded; GPU work is still batched by ``--batch-size``.

Example (full ``qwen_memtrace_proxy_train.jsonl`` on one GPU)::

  python -m mia_eval.score_wbc_jsonl \\
    --input data/qwen_memtrace_proxy_train.jsonl \\
    --output data/wbc_only/distil_proxy_train_wbc.jsonl \\
    --chunk-lines 256 \\
    --batch-size 4 \\
    --summary-json data/wbc_only/distil_proxy_train_wbc_summary.json

Each shard summary includes a ``quantification`` block (exact zeros vs ``wbc_short`` /
``n < 2`` NLL path). Use ``--per-row-quant`` to store ``wbc_input_tokens``, ``wbc_nll_len``,
and ``wbc_short`` on every row (needed for short-nll breakdown when merging shards).
``--quant-log-json`` writes only that block to a small JSON file.

Parallel shards (same pattern as ``score_proxy_jsonl``)::

  python -m mia_eval.score_wbc_jsonl ... --num-shards 8 --shard-id 0 --output .../shard0000.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_eval.config_loader import active_model_bundle, apply_dot_overrides, load_merged_config
from mia_eval.mia_gt_pipeline import _wbc_params_merge
from mia_eval.model_utils import load_causal_lm, pick_device, torch_dtype_from_str
from mia_eval.scoring_wbc import score_texts as wbc_scores
from mia_eval.wbc_quantification import wbc_quantification_summary


def _iter_nonempty_jsonl_lines(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield (line_index_0_based, obj) for non-empty JSON lines."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def _count_nonempty_lines(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _shard_range(n_total: int, shard_id: int, num_shards: int) -> Tuple[int, int]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards})")
    start = (n_total * shard_id) // num_shards
    end = (n_total * (shard_id + 1)) // num_shards
    return start, end


def _summarize(wbc: np.ndarray, labels: Optional[np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n": int(wbc.size),
        "wbc_mean": float(np.mean(wbc)),
        "wbc_std": float(np.std(wbc)),
        "wbc_min": float(np.min(wbc)),
        "wbc_max": float(np.max(wbc)),
        "wbc_quantiles": {
            "p5": float(np.percentile(wbc, 5)),
            "p25": float(np.percentile(wbc, 25)),
            "p50": float(np.percentile(wbc, 50)),
            "p75": float(np.percentile(wbc, 75)),
            "p95": float(np.percentile(wbc, 95)),
        },
    }
    if labels is not None and labels.size == wbc.size:
        m0 = labels == 0
        m1 = labels == 1
        if int(m0.sum()) and int(m1.sum()):
            out["by_label"] = {
                "0": {
                    "n": int(m0.sum()),
                    "mean": float(np.mean(wbc[m0])),
                    "std": float(np.std(wbc[m0])),
                    "p50": float(np.percentile(wbc[m0], 50)),
                },
                "1": {
                    "n": int(m1.sum()),
                    "mean": float(np.mean(wbc[m1])),
                    "std": float(np.std(wbc[m1])),
                    "p50": float(np.percentile(wbc[m1], 50)),
                },
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="WBC-only JSONL scoring (streaming chunks).")
    ap.add_argument("--config", type=str, default=str(ROOT / "mia_eval/config/defaults.yaml"))
    ap.add_argument("--experiment", type=str, default=str(ROOT / "mia_eval/config/qwen2p5.yaml"))
    ap.add_argument("--set", action="append", default=[], help="Config overrides.")
    ap.add_argument(
        "--preset",
        type=str,
        default="distil_qwen25_7b_instruct",
        help="active_model key under models: (default Distil Qwen).",
    )
    ap.add_argument("--input", type=str, required=True, help="JSONL with a text field per line.")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    ap.add_argument("--batch-size", type=int, default=4, help="Texts per WBC batch (GPU).")
    ap.add_argument(
        "--chunk-lines",
        type=int,
        default=256,
        help="Read this many non-empty JSON lines from disk before scoring (memory bound).",
    )
    ap.add_argument("--max-rows", type=int, default=0, help="If >0, stop after this many scored rows.")
    ap.add_argument("--num-shards", type=int, default=1, help="Split logical row index space.")
    ap.add_argument("--shard-id", type=int, default=0, help="0 <= shard-id < num-shards.")
    ap.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Write distribution summary (needs labels 0/1 on all rows for by_label).",
    )
    ap.add_argument(
        "--per-row-quant",
        action="store_true",
        help="Add wbc_input_tokens, wbc_nll_len, wbc_short to each output row (larger JSONL).",
    )
    ap.add_argument(
        "--quant-log-json",
        type=str,
        default="",
        help="If set, write only the quantification block (zeros vs short-nll) to this JSON file.",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"Not found: {inp}")

    cfg = load_merged_config(Path(args.config), Path(args.experiment))
    cfg = apply_dot_overrides(cfg, args.set)
    cfg["active_model"] = args.preset
    model_key = str(cfg.get("active_model", ""))
    bundle = active_model_bundle(cfg)
    block = (cfg.get("mia_gt_pipeline") or {}) or {}
    wbc_p = dict(block.get("open_model_wbc") or {"min_window": 2, "max_window": 40, "num_windows": 12})
    wbc_params = _wbc_params_merge(cfg, wbc_p)

    exp = cfg.get("experiment") or {}
    device = pick_device(exp.get("device"))
    dtype = torch_dtype_from_str(bundle.get("torch_dtype"))
    target = bundle["target_model"]
    ref_name = bundle["reference_model"]
    tok_id = bundle.get("tokenizer") or target
    max_len = int(exp.get("max_length_tokens", 512))

    print(f"[wbc_only] Counting non-empty lines in {inp} …", file=sys.stderr, flush=True)
    n_total = _count_nonempty_lines(inp)
    if n_total == 0:
        raise SystemExit("No non-empty JSON lines in input.")

    ns = int(args.num_shards)
    sid = int(args.shard_id)
    if ns < 1 or sid < 0 or sid >= ns:
        raise SystemExit("Need 0 <= shard-id < num-shards and num-shards >= 1.")
    row_start, row_end = _shard_range(n_total, sid, ns)
    max_rows = int(args.max_rows) if args.max_rows and args.max_rows > 0 else None
    if max_rows is not None:
        row_end = min(row_end, row_start + max_rows)

    n_shard = row_end - row_start
    if n_shard <= 0:
        raise SystemExit(f"Empty shard range [{row_start}, {row_end}).")
    print(
        f"[wbc_only] preset={model_key} rows_in_file={n_total} this_shard=[{row_start},{row_end}) n={n_shard}",
        file=sys.stderr,
        flush=True,
    )

    print(f"[wbc_only] Loading target {target} …", file=sys.stderr, flush=True)
    model_t, tok = load_causal_lm(target, tok_id, device, dtype, attn_implementation="eager")
    print(f"[wbc_only] Loading reference {ref_name} …", file=sys.stderr, flush=True)
    model_r, _ = load_causal_lm(ref_name, tok_id, device, dtype)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bs = max(1, int(args.batch_size))
    chunk = max(1, int(args.chunk_lines))

    wbc_all: List[float] = []
    lab_all: List[int] = []
    diag_all: List[Dict[str, Any]] = []

    logical = 0
    written = 0
    buf_rows: List[Dict[str, Any]] = []
    buf_logical: List[int] = []

    with open(out_path, "w", encoding="utf-8") as out_f:

        def flush_buffer() -> None:
            nonlocal written
            if not buf_rows:
                return
            texts = [str(r.get("text", "")).strip() for r in buf_rows]
            if any(not t for t in texts):
                raise ValueError("Empty text in chunk after strip.")
            nloc = len(texts)
            scores = np.empty(nloc, dtype=np.float64)
            chunk_diag: List[Dict[str, Any]] = []
            for i in range(0, nloc, bs):
                j = min(i + bs, nloc)
                scores[i:j] = wbc_scores(
                    model_t,
                    model_r,
                    tok,
                    texts[i:j],
                    device,
                    wbc_params,
                    max_length=max_len,
                    diag_out=chunk_diag,
                )
            diag_all.extend(chunk_diag)
            for k, r in enumerate(buf_rows):
                gix = buf_logical[k]
                w = float(scores[k])
                wbc_all.append(w)
                lab = r.get("label")
                if lab is not None:
                    try:
                        lab_all.append(int(lab))
                    except (TypeError, ValueError):
                        lab_all.append(-1)
                else:
                    lab_all.append(-1)
                doc: Dict[str, Any] = {
                    "proxy_row_index": gix,
                    "wbc": w,
                }
                if lab is not None:
                    try:
                        doc["label"] = int(lab)
                    except (TypeError, ValueError):
                        pass
                if r.get("source") is not None:
                    doc["source"] = r.get("source")
                if args.per_row_quant and k < len(chunk_diag):
                    q = chunk_diag[k]
                    doc["wbc_input_tokens"] = int(q["wbc_input_tokens"])
                    doc["wbc_nll_len"] = int(q["wbc_nll_len"])
                    doc["wbc_short"] = bool(q["wbc_short"])
                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += 1
            buf_rows.clear()
            buf_logical.clear()

        for _line_no, obj in _iter_nonempty_jsonl_lines(inp):
            if logical >= row_end:
                break
            if logical < row_start:
                logical += 1
                continue
            buf_rows.append(obj)
            buf_logical.append(logical)
            logical += 1
            if len(buf_rows) >= chunk:
                flush_buffer()

        flush_buffer()

    del model_t, model_r
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()

    wbc_arr = np.asarray(wbc_all, dtype=np.float64)
    labels_arr: Optional[np.ndarray] = None
    if lab_all and all(x in (0, 1) for x in lab_all):
        labels_arr = np.asarray(lab_all, dtype=np.int64)

    short_arr: Optional[np.ndarray] = None
    if len(diag_all) == int(wbc_arr.size):
        short_arr = np.asarray([bool(d["wbc_short"]) for d in diag_all], dtype=bool)
    elif diag_all:
        print(
            "[quant] warning: diag length mismatch; omitting short-nll breakdown",
            file=sys.stderr,
            flush=True,
        )
    quant_block = wbc_quantification_summary(wbc_arr, labels_arr, short_arr)

    summary = {
        "input": str(inp.resolve()),
        "output": str(out_path.resolve()),
        "active_model": model_key,
        "target_model": target,
        "reference_model": ref_name,
        "wbc_params_open_model_wbc": wbc_p,
        "n_rows_file_nonempty": n_total,
        "shard": {"num_shards": ns, "shard_id": sid, "row_range": [row_start, row_end]},
        "n_rows_written": written,
        "distribution": _summarize(wbc_arr, labels_arr),
        "quantification": quant_block,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    qparts = [
        f"n_wbc_zero={quant_block['n_wbc_exactly_zero']}",
        f"frac_zero={quant_block['frac_wbc_exactly_zero']:.6g}",
    ]
    if quant_block.get("wbc_short_available"):
        qparts = [
            f"n_short_nll={quant_block['n_wbc_short_nll']}",
            f"frac_short={quant_block['frac_wbc_short_nll']:.6g}",
            f"n_zero_not_short={quant_block['n_wbc_exactly_zero_not_short']}",
        ] + qparts
    print("[quant] " + " ".join(qparts), file=sys.stderr, flush=True)

    if str(args.summary_json).strip():
        sp = Path(args.summary_json)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote {sp}", file=sys.stderr)

    if str(args.quant_log_json).strip():
        qp = Path(args.quant_log_json)
        qp.parent.mkdir(parents=True, exist_ok=True)
        with open(qp, "w", encoding="utf-8") as f:
            json.dump(quant_block, f, indent=2, ensure_ascii=False)
        print(f"Wrote quantification log {qp}", file=sys.stderr)


if __name__ == "__main__":
    main()
