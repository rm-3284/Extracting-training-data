#!/usr/bin/env python3
"""
Materialize ``text`` / ``label`` / ``stream_id`` JSONL from ``qwen_memtrace_datasets.yaml``.

Uses Hugging Face ``datasets`` streaming (no ``trust_remote_code``). Run from repo root:

  python -m mia_eval.prepare_memtrace_proxy_jsonl \\
    --manifest mia_eval/config/qwen_memtrace_datasets.yaml \\
    --output data/qwen_memtrace_proxy_train.jsonl

  # Add code + math member streams (see YAML keys ``coder_*`` / ``math_*``):
  python -m mia_eval.prepare_memtrace_proxy_jsonl \\
    --manifest mia_eval/config/qwen_memtrace_datasets.yaml \\
    --output data/qwen_memtrace_with_coder_math.jsonl \\
    --include coder math

Then fit memTrace RF on features extracted from your Qwen checkpoint (separate script / notebook).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _open_stream(
    stream: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Any:
    from datasets import load_dataset

    path = stream["hf_path"]
    name = stream.get("hf_name")
    split = stream.get("split", defaults.get("split", "train"))
    streaming = stream.get("streaming", defaults.get("streaming", True))
    if name in (None, "", "null", "None"):
        return load_dataset(path, split=split, streaming=streaming)
    return load_dataset(path, name, split=split, streaming=streaming)


def _row_text(row: Dict[str, Any], text_field: str) -> str:
    t = row.get(text_field) or row.get("text") or row.get("content") or ""
    if not isinstance(t, str):
        return ""
    return t.strip()


def _passes_date(
    row: Dict[str, Any],
    date_field: Optional[str],
    on_or_after: Optional[str],
) -> bool:
    if not date_field or not on_or_after:
        return True
    raw = row.get(date_field)
    if raw is None or raw == "":
        return True
    try:
        cutoff = date.fromisoformat(str(on_or_after)[:10])
    except ValueError:
        return True
    s = str(raw)[:10]
    try:
        d = date.fromisoformat(s)
    except ValueError:
        return True
    return d >= cutoff


def _collect_stream(
    stream: Dict[str, Any],
    label: int,
    defaults: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    min_c = int(defaults.get("min_chars", 180))
    max_c = int(defaults.get("max_chars", 6000))
    text_field = stream["text_field"]
    cap = int(stream["max_samples"])
    date_field = stream.get("date_field")
    date_on_or_after = stream.get("date_on_or_after")

    try:
        ds = _open_stream(stream, defaults)
    except Exception as e:
        raise RuntimeError(f"Failed to open stream {stream.get('id')!r}: {e}") from e

    got = 0
    for row in ds:
        if got >= cap:
            break
        if not _passes_date(row, date_field, date_on_or_after):
            continue
        text = _row_text(row, text_field)
        if len(text) < min_c:
            continue
        if len(text) > max_c:
            text = text[:max_c]
        yield {
            "text": text,
            "label": int(label),
            "stream_id": stream["id"],
            "hf_path": stream["hf_path"],
            "hf_name": stream.get("hf_name"),
        }
        got += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "mia_eval/config/qwen_memtrace_datasets.yaml"),
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path (created parent dirs).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--include",
        nargs="*",
        default=[],
        choices=("coder", "math"),
        help="Append optional member streams (coder → the-stack python; math → open-web-math).",
    )
    args = ap.parse_args()

    man = _load_yaml(Path(args.manifest))
    defaults = {**(man.get("defaults") or {})}
    members: List[Dict[str, Any]] = list(man.get("member_proxy_streams") or [])
    if "coder" in args.include:
        members.extend(man.get("coder_member_proxy_streams") or [])
    if "math" in args.include:
        members.extend(man.get("math_member_proxy_streams") or [])
    nonmem = man.get("non_member_proxy_streams") or []
    if not members or not nonmem:
        raise SystemExit("Manifest must define member_proxy_streams and non_member_proxy_streams.")

    rows: List[Dict[str, Any]] = []
    per_stream: Dict[str, int] = {}
    for st in members:
        chunk = list(_collect_stream(st, 1, defaults))
        per_stream[st["id"]] = len(chunk)
        rows.extend(chunk)
    for st in nonmem:
        chunk = list(_collect_stream(st, 0, defaults))
        per_stream[st["id"]] = len(chunk)
        rows.extend(chunk)

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n1 = sum(1 for r in rows if r["label"] == 1)
    n0 = sum(1 for r in rows if r["label"] == 0)
    print(
        json.dumps(
            {
                "output": str(out),
                "n_total": len(rows),
                "n_member_proxy": n1,
                "n_non_member_proxy": n0,
                "per_stream_counts": per_stream,
                "include": args.include,
                "manifest": str(Path(args.manifest).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
