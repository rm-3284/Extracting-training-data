#!/usr/bin/env python3
"""
Merge shard outputs from ``mia_eval.score_proxy_jsonl`` (each row must have
``proxy_row_index``), sort by that index, optionally strip the index field, and write
one JSONL in original order.

Example::

  python -m mia_eval.merge_proxy_scored_shards \\
    --inputs data/proxy_scored_shard0004_of_0032.jsonl data/proxy_scored_shard0005_of_0032.jsonl \\
    --output data/qwen_memtrace_proxy_train_scored.jsonl

  # Or glob (quote for your shell):
  python -m mia_eval.merge_proxy_scored_shards \\
    --glob 'data/qwen_memtrace_proxy_train_scored_shard*_of_0032.jsonl' \\
    --output data/qwen_memtrace_proxy_train_scored.jsonl \\
    --strip-proxy-row-index
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_paths_from_glob(pattern: str) -> List[Path]:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched glob: {pattern!r}")
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge scored proxy JSONL shards by proxy_row_index.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--inputs",
        nargs="+",
        type=str,
        help="Shard JSONL paths (any order).",
    )
    g.add_argument("--glob", type=str, help="Glob pattern of shard files (sorted lexicographically).")
    ap.add_argument("--output", type=str, required=True, help="Merged JSONL path.")
    ap.add_argument(
        "--strip-proxy-row-index",
        action="store_true",
        help="Remove proxy_row_index from each written object.",
    )
    ap.add_argument(
        "--expected-rows",
        type=int,
        default=0,
        help="If >0, require exactly this many rows after merge (full coverage check).",
    )
    args = ap.parse_args()

    if args.glob:
        paths = _load_paths_from_glob(args.glob)
    else:
        paths = [Path(p) for p in args.inputs]

    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Not a file: {p}")

    merged: List[Dict[str, Any]] = []
    for p in paths:
        merged.extend(_read_jsonl(p))

    missing = [i for i, r in enumerate(merged) if "proxy_row_index" not in r]
    if missing:
        raise SystemExit(f"{len(missing)} objects missing proxy_row_index (e.g. row {missing[0]}).")

    merged.sort(key=lambda r: int(r["proxy_row_index"]))

    keys = [int(r["proxy_row_index"]) for r in merged]
    if len(keys) != len(set(keys)):
        from collections import Counter

        c = Counter(keys)
        dup = [k for k, v in c.items() if v > 1][:10]
        raise SystemExit(f"Duplicate proxy_row_index values (examples): {dup}")

    n = len(keys)
    if n == 0:
        raise SystemExit("No rows to merge.")

    mn, mx = min(keys), max(keys)
    expected_span = mx - mn + 1
    if expected_span != n:
        print(
            f"[merge_proxy] warning: indices span [{mn}, {mx}] ({expected_span} slots) but only {n} rows "
            f"(gaps in coverage).",
            file=sys.stderr,
        )

    if int(args.expected_rows) > 0 and n != int(args.expected_rows):
        raise SystemExit(f"Expected {args.expected_rows} rows after merge, got {n}.")

    if mn != 0:
        print(f"[merge_proxy] warning: minimum proxy_row_index is {mn}, not 0.", file=sys.stderr)

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
            {
                "output": str(out_path.resolve()),
                "n_files": len(paths),
                "n_rows": n,
                "proxy_row_index_min": mn,
                "proxy_row_index_max": mx,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
