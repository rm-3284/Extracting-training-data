#!/usr/bin/env python3
"""
Merge shard outputs from ``mia_eval.score_wbc_jsonl`` (``--num-shards`` / ``--shard-id``),
sort by ``proxy_row_index``, optionally write a combined distribution summary.

Example::

  python -m mia_eval.merge_wbc_jsonl_shards \\
    --glob 'data/wbc_only/distil_proxy_shards/*_of_0032.jsonl' \\
    --output data/wbc_only/distil_proxy_train_wbc_merged.jsonl \\
    --expected-rows 12200 \\
    --summary-json data/wbc_only/distil_proxy_train_wbc_merged_summary.json

Merged ``--summary-json`` includes ``quantification``. If every row has ``wbc_short``
(from scoring with ``--per-row-quant``), the merge aggregates short-nll counts;
otherwise only exact-zero stats are available. ``--quant-log-json`` writes just that block.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_eval.wbc_quantification import wbc_quantification_summary


def _paths_from_glob(pattern: str) -> List[Path]:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched: {pattern!r}")
    return paths


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: invalid JSON: {e}") from e
    return rows


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
    ap = argparse.ArgumentParser(description="Merge WBC-only shard JSONLs by proxy_row_index.")
    ap.add_argument("--inputs", nargs="+", help="Shard JSONL paths.")
    ap.add_argument("--glob", type=str, help="Glob of shard files (sorted).")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--expected-rows", type=int, default=0, help="If >0, require exactly this many rows.")
    ap.add_argument("--strip-proxy-row-index", action="store_true")
    ap.add_argument("--summary-json", type=str, default="", help="Write merged distribution JSON.")
    ap.add_argument(
        "--quant-log-json",
        type=str,
        default="",
        help="If set, write only the merged quantification block to this JSON file.",
    )
    args = ap.parse_args()

    if bool(args.glob) == bool(args.inputs):
        raise SystemExit("Provide exactly one of --glob or --inputs.")

    paths = _paths_from_glob(args.glob) if args.glob else [Path(p) for p in args.inputs]

    merged: List[Dict[str, Any]] = []
    for p in paths:
        merged.extend(_read_jsonl(p))

    for i, r in enumerate(merged):
        if "proxy_row_index" not in r:
            raise SystemExit(f"Row {i} missing proxy_row_index (from shard merge).")

    merged.sort(key=lambda r: int(r["proxy_row_index"]))
    keys = [int(r["proxy_row_index"]) for r in merged]
    if len(keys) != len(set(keys)):
        c = Counter(keys)
        dup = [k for k, v in c.items() if v > 1][:10]
        raise SystemExit(f"Duplicate proxy_row_index (examples): {dup}")

    n = len(keys)
    if int(args.expected_rows) > 0 and n != int(args.expected_rows):
        raise SystemExit(f"Expected {args.expected_rows} rows after merge, got {n}.")

    mn, mx = min(keys), max(keys)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in merged:
            doc = dict(r)
            if args.strip_proxy_row_index:
                doc.pop("proxy_row_index", None)
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {"output": str(out_path.resolve()), "n_files": len(paths), "n_rows": n, "index_range": [mn, mx]},
            indent=2,
        )
    )

    want_summ = bool(str(args.summary_json).strip())
    want_quant_only = bool(str(args.quant_log_json).strip())
    if want_summ or want_quant_only:
        wbc = np.array([float(r["wbc"]) for r in merged], dtype=np.float64)
        labels_m: Optional[np.ndarray] = None
        if all("label" in r for r in merged):
            lab = [int(r["label"]) for r in merged]
            if all(x in (0, 1) for x in lab):
                labels_m = np.asarray(lab, dtype=np.int64)
        short = None
        if merged and all("wbc_short" in r for r in merged):
            short = np.asarray([bool(r["wbc_short"]) for r in merged], dtype=bool)
        quant_block = wbc_quantification_summary(wbc, labels_m, short)
        if want_summ:
            summ = {
                "merged_output": str(out_path.resolve()),
                "n_rows": n,
                "distribution": _summarize(wbc, labels_m),
                "quantification": quant_block,
            }
            sp = Path(args.summary_json)
            sp.parent.mkdir(parents=True, exist_ok=True)
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(summ, f, indent=2, ensure_ascii=False)
            print(f"Wrote {sp}", file=sys.stderr)
        if want_quant_only:
            qp = Path(args.quant_log_json)
            qp.parent.mkdir(parents=True, exist_ok=True)
            with open(qp, "w", encoding="utf-8") as f:
                json.dump(quant_block, f, indent=2, ensure_ascii=False)
            print(f"Wrote quantification log {qp}", file=sys.stderr)
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


if __name__ == "__main__":
    main()
