#!/usr/bin/env python3
"""
Randomly sample N non-empty lines from a proxy JSONL (same row set as
``mia_eval.score_proxy_jsonl``) and write them in **original file order** for a
reproducible calibration subset.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List


def _read_nonempty_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Randomly sample N rows from a proxy JSONL.")
    ap.add_argument("--input", type=str, required=True, help="Input JSONL path.")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    ap.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=512,
        help="Number of rows to sample without replacement (default 512).",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    n = int(args.num_samples)
    if n < 1:
        raise SystemExit("--num-samples must be >= 1.")

    lines = _read_nonempty_lines(inp)
    if not lines:
        raise SystemExit(f"No non-empty lines in {inp}")
    if n > len(lines):
        raise SystemExit(f"Requested {n} samples but only {len(lines)} non-empty lines in {inp}")

    rng = random.Random(int(args.seed))
    pick = sorted(rng.sample(range(len(lines)), n))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i in pick:
            f.write(lines[i] + "\n")

    meta = {
        "input": str(inp.resolve()),
        "output": str(out.resolve()),
        "n_non_empty_input": len(lines),
        "n_sampled": n,
        "seed": int(args.seed),
        "sampled_original_indices": pick,
    }
    meta_path = out.with_name(out.stem + ".sample_meta.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)
    summary = {
        "input": meta["input"],
        "output": meta["output"],
        "sample_meta": str(meta_path.resolve()),
        "n_non_empty_input": len(lines),
        "n_sampled": n,
        "seed": int(args.seed),
        "sampled_index_min": min(pick),
        "sampled_index_max": max(pick),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out} ({n} lines); meta {meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
