#!/usr/bin/env python3
"""
Build a **wide** precision@k table: metrics as rows, models as column groups with P@10 / P@50 / P@100.

Matches the usual paper layout (Carlini-style metric names as rows). Uses **wbc_proxy** values from
``carlini_ranking_proxy_labels`` (rank by each Carlini score; precision = fraction with WBC proxy = 1 in top-k).

Outputs HTML (two-row header), CSV, and a simple Markdown variant.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Row order like Carlini Table 2-style listings (reference figure).
CARLINI_ROW_ORDER = ["Perplexity", "zlib", "Lowercase", "Window", "Small"]


def _fmt(v: Optional[float], decimals: int) -> str:
    if v is None:
        return ""
    return f"{float(v):.{decimals}f}"


def _get_pk(
    doc: Dict[str, Any],
    section: str,
    ranker: str,
    proxy_key: str,
    ks: Sequence[int],
) -> List[Optional[float]]:
    root = doc.get(section) or {}
    block = root.get(ranker) if isinstance(root, dict) else None
    if not isinstance(block, dict):
        return [None] * len(ks)
    wp = block.get(proxy_key)
    if not isinstance(wp, dict):
        return [None] * len(ks)
    out: List[Optional[float]] = []
    for k in ks:
        key = f"@{k}"
        raw = wp.get(key)
        out.append(None if raw is None else float(raw))
    return out


def build_wide_rows(
    models: Dict[str, Any],
    ks: List[int],
    proxy_key: str,
) -> tuple[List[str], List[List[str]]]:
    """Returns (model_keys_sorted, table_rows as list of [metric_label, v1, v2, ...])."""
    model_keys = sorted(models.keys())
    metric_rows: List[tuple[str, str, str]] = []
    # (display_label, section, ranker)
    for name in CARLINI_ROW_ORDER:
        metric_rows.append((name, "carlini_ranking_proxy_labels", name))

    out: List[List[str]] = []
    for label, section, ranker in metric_rows:
        row: List[str] = [label]
        for mk in model_keys:
            doc = models.get(mk)
            if not isinstance(doc, dict):
                row.extend([""] * len(ks))
                continue
            vals = _get_pk(doc, section, ranker, proxy_key, ks)
            row.extend(_fmt(v, 2) if v is not None else "" for v in vals)
        out.append(row)
    return model_keys, out


def write_csv(
    path: Path,
    model_keys: List[str],
    ks: List[int],
    rows: List[List[str]],
) -> None:
    header = ["Metric"]
    for mk in model_keys:
        for k in ks:
            header.append(f"{mk}_P@{k}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def write_html(
    path: Path,
    model_keys: List[str],
    ks: List[int],
    rows: List[List[str]],
    title: str,
) -> None:
    """Two-level header: model names colspan, then P@k."""
    nmodels = len(model_keys)
    css = """
    table.pkat-wide { border-collapse: collapse; font-family: system-ui, -apple-system, Segoe UI, sans-serif; font-size: 13px; margin: 1em 0; }
    table.pkat-wide th, table.pkat-wide td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #ccc; }
    table.pkat-wide th.metric, table.pkat-wide td.metric { text-align: left; font-weight: 500; border-right: 1px solid #ccc; }
    table.pkat-wide thead tr:first-child th { border-bottom: none; padding-bottom: 2px; }
    table.pkat-wide thead tr:nth-child(2) th { padding-top: 2px; font-weight: 600; color: #333; }
    table.pkat-wide thead { border-bottom: 2px solid #999; }
    table.pkat-wide caption { caption-side: top; text-align: left; font-weight: 600; margin-bottom: 8px; }
    """
    parts: List[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>"
        + html.escape(title)
        + "</title><style>"
        + css
        + "</style></head><body>",
        f"<table class='pkat-wide'><caption>{html.escape(title)}</caption>",
        "<thead>",
        "<tr><th class='metric' rowspan='2'>Metric</th>",
    ]
    for mk in model_keys:
        parts.append(f"<th colspan='{len(ks)}'>{html.escape(mk)}</th>")
    parts.append("</tr><tr>")
    for _ in model_keys:
        for k in ks:
            parts.append(f"<th>P@{k}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr><td class='metric'>" + html.escape(row[0]) + "</td>")
        for cell in row[1:]:
            parts.append("<td>" + html.escape(cell) + "</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def write_markdown_simple(
    path: Path,
    model_keys: List[str],
    ks: List[int],
    rows: List[List[str]],
    title: str,
) -> None:
    """Single header row (wide) — readable where HTML is not used."""
    hdr = ["Metric"] + [f"{m} P@{k}" for m in model_keys for k in ks]
    lines = [
        "### " + title,
        "",
        "| " + " | ".join(hdr) + " |",
        "| " + " | ".join(["---"] * len(hdr)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Wide precision@k table (models × P@k columns).")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--proxy-key", type=str, default="wbc_proxy")
    ap.add_argument("--title", type=str, default="Precision@k (wbc proxy labels)")
    ap.add_argument("--html", type=str, default="")
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--markdown", type=str, default="")
    args = ap.parse_args()

    path = Path(args.input)
    with open(path, encoding="utf-8") as f:
        root = json.load(f)
    ks = list(root.get("k") or [10, 50, 100])
    models = root.get("models") or {}
    if not isinstance(models, dict) or not models:
        raise SystemExit("Expected non-empty 'models' in JSON.")

    model_keys, rows = build_wide_rows(models, ks, args.proxy_key)

    stem = path.with_suffix("")
    html_path = Path(args.html) if args.html else Path(str(stem) + "_wide_wbc_proxy.html")
    csv_path = Path(args.csv) if args.csv else Path(str(stem) + "_wide_wbc_proxy.csv")
    md_path = Path(args.markdown) if args.markdown else Path(str(stem) + "_wide_wbc_proxy.md")

    write_html(html_path, model_keys, ks, rows, args.title)
    write_csv(csv_path, model_keys, ks, rows)
    write_markdown_simple(md_path, model_keys, ks, rows, args.title)

    print(f"Wrote {html_path}", file=sys.stderr)
    print(f"Wrote {csv_path}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
