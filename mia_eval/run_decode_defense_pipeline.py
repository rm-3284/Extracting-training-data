#!/usr/bin/env python3
"""
Run Carlini-style generation for decode-time defenses **and** standard Carlini extraction on open
HF checkpoints (the default overlay keeps ``carlini_sampling.enabled: true`` so ``top_k``,
``temperature_decay``, internet-tagged rows, etc. appear in the same JSONL as defense strategies for
comparison).

This is a thin wrapper around ``mia_eval.run_carlini_extraction_batch`` with documentation for
measuring extraction **after** you have ``samples.jsonl``:

1. **Shingle overlap vs released text** (optional): build an index from
   ``mia_eval/config/carlini_training_corpora.yaml`` and run
   ``python -m mia_eval.label_carlini_extract_samples …`` (see that module’s docstring).

2. **Carlini Table 2-style cheap filters** (perplexity / zlib / window, optional P@k if you have
   labels): ``python -m mia_eval.run_carlini_table2 --input …/samples.jsonl`` with a merged
   experiment YAML that sets ``active_model`` to the same HF id as the run.

Examples::

  python -m mia_eval.run_decode_defense_pipeline \\
    --config mia_eval/config/carlini_open_models.yaml \\
    --merge-config mia_eval/config/carlini_overlay_decode_defenses.yaml \\
    --only pythia_12b_base --num-samples-per-strategy 64

  python -m mia_eval.run_decode_defense_pipeline --dry-run \\
    --config mia_eval/config/carlini_open_models.yaml \\
    --merge-config mia_eval/config/carlini_overlay_decode_defenses.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "mia_eval" / "config" / "carlini_open_models.yaml"
DEFAULT_MERGE = ROOT / "mia_eval" / "config" / "carlini_overlay_decode_defenses.yaml"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Decode-defense extraction benchmark (delegates to run_carlini_extraction_batch)"
    )
    p.add_argument("--config", type=Path, default=DEFAULT_BASE, help="Base YAML (e.g. carlini_open_models)")
    p.add_argument(
        "--merge-config",
        type=Path,
        default=DEFAULT_MERGE,
        help="Overlay YAML (e.g. carlini_overlay_decode_defenses)",
    )
    p.add_argument("--only", type=str, default="", help="Comma-separated run_key filter")
    p.add_argument("--num-samples-per-strategy", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "mia_eval.run_carlini_extraction_batch",
        "--config",
        str(args.config.resolve()),
        "--merge-config",
        str(args.merge_config.resolve()),
    ]
    if args.only.strip():
        cmd.extend(["--only", args.only.strip()])
    if args.num_samples_per_strategy is not None:
        cmd.extend(["--num-samples-per-strategy", str(int(args.num_samples_per_strategy))])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(int(args.batch_size))])
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.dry_run:
        cmd.append("--dry-run")

    print("Running:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT), env=env))


if __name__ == "__main__":
    main()
