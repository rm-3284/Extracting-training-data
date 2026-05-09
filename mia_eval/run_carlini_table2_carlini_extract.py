#!/usr/bin/env python3
"""
Run ``mia_eval.run_carlini_table2`` on every model under ``mia_eval_outputs/carlini_extract``
that has ``samples.jsonl``.

Requires ``mia_eval/config/carlini_open_table2_models.yaml`` (merged over defaults) so each
``run_key`` is a valid ``active_model`` with target + reference HF IDs.

Examples (repo root)::

  # All runs with samples under carlini_extract/
  python -m mia_eval.run_carlini_table2_carlini_extract

  # Custom root + skip runs that already wrote carlini_table2.json
  python -m mia_eval.run_carlini_table2_carlini_extract \\
    --carlini-root mia_eval_outputs/carlini_extract --skip-existing

  # Only selected keys (comma-separated)
  python -m mia_eval.run_carlini_table2_carlini_extract --only olmo2_7b_base,redpajama_7b_base

Env:
  PYTHONPATH must include the repo root (same as other ``python -m mia_eval.*`` modules).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Set


ROOT = Path(__file__).resolve().parents[1]


def _parse_only(s: str | None) -> Set[str] | None:
    if not s or not str(s).strip():
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def _discover_run_dirs(carlini_root: Path) -> List[Path]:
    if not carlini_root.is_dir():
        raise FileNotFoundError(f"carlini_root not found: {carlini_root}")
    out: List[Path] = []
    for p in sorted(carlini_root.iterdir()):
        if p.is_dir() and (p / "samples.jsonl").is_file():
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch Carlini Table-2 / precision@k for carlini_extract sample dirs."
    )
    ap.add_argument(
        "--carlini-root",
        type=Path,
        default=ROOT / "mia_eval_outputs" / "carlini_extract",
        help="Directory containing <run_key>/samples.jsonl (default: mia_eval_outputs/carlini_extract).",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "mia_eval" / "config" / "defaults.yaml",
        help="Base config (default: mia_eval/config/defaults.yaml).",
    )
    ap.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "mia_eval" / "config" / "carlini_open_table2_models.yaml",
        help="YAML merged over base; must define ``models`` for each run_key (default: carlini_open_table2_models.yaml).",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated run_key values (default: all dirs with samples.jsonl).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if <run_dir>/carlini_table2.json already exists.",
    )
    ap.add_argument(
        "--precision-k",
        type=str,
        default="10,50,100",
        help="Forwarded to run_carlini_table2 --precision-k.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned subprocess commands only.",
    )
    args = ap.parse_args()

    only = _parse_only(args.only)
    run_dirs = _discover_run_dirs(args.carlini_root)
    if only is not None:
        run_dirs = [p for p in run_dirs if p.name in only]

    if not run_dirs:
        msg = f"No runs found under {args.carlini_root} (need */samples.jsonl)"
        if only:
            msg += "; --only filter left nothing"
        raise SystemExit(msg)

    cfg = args.config.resolve()
    exp = args.experiment.resolve()

    failures: List[str] = []
    for run_dir in run_dirs:
        run_key = run_dir.name
        out_json = run_dir / "carlini_table2.json"
        if args.skip_existing and out_json.is_file():
            print(f"[skip] {run_key}: exists {out_json}", flush=True)
            continue

        cmd = [
            sys.executable,
            "-m",
            "mia_eval.run_carlini_table2",
            "--config",
            str(cfg),
            "--experiment",
            str(exp),
            "--set",
            f"active_model={run_key}",
            "--input",
            str(run_dir / "samples.jsonl"),
            "--output",
            str(out_json),
            "--precision-k",
            args.precision_k,
        ]
        print(f"[run] {run_key}", flush=True)
        if args.dry_run:
            print(" ", subprocess.list2cmdline(cmd), flush=True)
            continue

        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            failures.append(run_key)

    if failures:
        raise SystemExit(f"Failed runs ({len(failures)}): {', '.join(failures)}")


if __name__ == "__main__":
    main()
