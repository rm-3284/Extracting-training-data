#!/usr/bin/env python3
"""
Per-``source`` breakdown for labeled Carlini / decode-defense ``samples.jsonl``.

When shingle labels saturate (~100% positive), positive rate alone is weak; this
still helps compare methods before running table2 and spot count imbalances.

Example::

  python -m mia_eval.summarize_decode_defense_metrics \\
    --jsonl mia_eval_outputs/carlini_extract_decode_defenses/gpt_neo_2p7/samples_labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_labeled(path: Path) -> Dict[str, Dict[str, Any]]:
    by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in _iter_jsonl(path):
        by_src[str(row.get("source", "?"))].append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for src, rows in sorted(by_src.items()):
        labels = [int(r.get("label", 0)) for r in rows]
        n_pos = sum(labels)
        n = len(rows)
        lengths = [len(str(r.get("text", ""))) for r in rows]
        out[src] = {
            "n": n,
            "n_positive": n_pos,
            "positive_rate": (n_pos / n) if n else 0.0,
            "len_chars_mean": statistics.mean(lengths) if lengths else 0.0,
            "len_chars_median": statistics.median(lengths) if lengths else 0.0,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, required=True, help="Labeled samples.jsonl")
    p.add_argument("--json-out", type=Path, default=None, help="Optional JSON summary path")
    args = p.parse_args()

    summary = summarize_labeled(args.jsonl)
    print(f"File: {args.jsonl}")
    print(f"{'source':<40} {'n':>6} {'pos':>6} {'rate':>8} {'len_med':>8}")
    print("-" * 72)
    for src, s in summary.items():
        print(
            f"{src:<40} {s['n']:>6} {s['n_positive']:>6} "
            f"{s['positive_rate']:>8.3f} {s['len_chars_median']:>8.0f}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"jsonl": str(args.jsonl), "by_source": summary}, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
