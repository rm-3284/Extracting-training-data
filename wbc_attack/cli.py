"""CLI for running WBC on precomputed token-loss arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .core import WBCConfig, wbc_score_from_losses


def _read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute WBC score from token-level losses."
    )
    parser.add_argument(
        "--target-losses",
        type=str,
        required=True,
        help="JSON file containing a list of target losses.",
    )
    parser.add_argument(
        "--reference-losses",
        type=str,
        required=True,
        help="JSON file containing a list of reference losses.",
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit window sizes.",
    )
    parser.add_argument("--min-window", type=int, default=2)
    parser.add_argument("--max-window", type=int, default=40)
    parser.add_argument("--num-windows", type=int, default=10)

    args = parser.parse_args()
    target_losses = _read_json(args.target_losses)
    reference_losses = _read_json(args.reference_losses)

    cfg = WBCConfig(
        window_sizes=args.window_sizes,
        min_window=args.min_window,
        max_window=args.max_window,
        num_windows=args.num_windows,
    )
    score = wbc_score_from_losses(target_losses, reference_losses, config=cfg)
    print(json.dumps({"wbc_score": score}, indent=2))


if __name__ == "__main__":
    main()
