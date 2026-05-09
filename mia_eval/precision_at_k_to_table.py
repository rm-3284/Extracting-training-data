#!/usr/bin/env python3
"""
Flatten ``precision_at_k_*`` JSON (hybrid / report output) into CSV / Markdown tables.

By default extracts **wbc_proxy** precision@k (rank by Carlini or MIA score; label =
thresholded WBC pseudo-label). Use ``--proxy-key`` for another proxy column.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _pk(block: Optional[Dict[str, Any]], ks: Sequence[int]) -> List[str]:
    if not block:
        return [""] * len(ks)
    out: List[str] = []
    for k in ks:
        key = f"@{k}"
        v = block.get(key)
        out.append("" if v is None else f"{float(v):.4f}".rstrip("0").rstrip("."))
    return out


def rows_from_model(
    model: str,
    doc: Dict[str, Any],
    proxy_key: str,
    ks: List[int],
) -> List[List[str]]:
    rows: List[List[str]] = []
    cr = doc.get("carlini_ranking_proxy_labels") or {}
    for ranker, proxies in sorted(cr.items()):
        if not isinstance(proxies, dict):
            continue
        wp = proxies.get(proxy_key)
        if not isinstance(wp, dict):
            continue
        rows.append(
            ["Carlini-ranked", model, ranker, *_pk(wp, ks)]
        )
    mr = doc.get("mia_ranking_proxy_labels") or {}
    for ranker, proxies in sorted(mr.items()):
        if not isinstance(proxies, dict):
            continue
        wp = proxies.get(proxy_key)
        if not isinstance(wp, dict):
            continue
        rows.append(
            ["MIA-ranked", model, ranker, *_pk(wp, ks)]
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CSV/MD tables from precision_at_k JSON.")
    ap.add_argument("--input", type=str, required=True, help="precision_at_k JSON path.")
    ap.add_argument(
        "--proxy-key",
        type=str,
        default="wbc_proxy",
        help="Nested key under each ranker (default wbc_proxy).",
    )
    ap.add_argument("--csv", type=str, default="", help="Write CSV path.")
    ap.add_argument("--markdown", type=str, default="", help="Write Markdown path.")
    args = ap.parse_args()

    path = Path(args.input)
    with open(path, encoding="utf-8") as f:
        root = json.load(f)

    ks = root.get("k") or [10, 50, 100]
    models = root.get("models") or {}
    if not isinstance(models, dict):
        raise SystemExit("Expected top-level 'models' object.")

    header = ["family", "model", "rank_by", *[f"P@{k}" for k in ks]]
    all_rows: List[List[str]] = []
    for model_key in sorted(models.keys()):
        doc = models[model_key]
        if not isinstance(doc, dict):
            continue
        all_rows.extend(rows_from_model(model_key, doc, args.proxy_key, ks))

    csv_path = Path(args.csv) if args.csv else path.with_name(
        path.stem + f"_{args.proxy_key}_table.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(all_rows)
    print(f"Wrote {csv_path} ({len(all_rows)} rows)", file=sys.stderr)

    md_path = Path(args.markdown) if args.markdown else path.with_name(
        path.stem + f"_{args.proxy_key}_table.md"
    )
    title = f"Precision@k using **{args.proxy_key}** (proxy labels from thresholds)"
    lines = [
        title,
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in all_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
