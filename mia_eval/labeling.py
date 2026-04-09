"""Assign member / non-member labels using training-data index + sample source heuristics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .ground_truth import TrainingShingleIndex


def label_jsonl(
    samples_path: Path,
    index: TrainingShingleIndex,
    min_match_chars: int,
    out_path: Path,
) -> Path:
    rows: List[Dict[str, Any]] = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = r.get("text", "")
            src = r.get("source", "")
            if src == "training_excerpt":
                label = 1
            elif src == "wikipedia_ood":
                label = 0
            else:
                label = 1 if index.matches(text, min_match_chars) else 0
            r["label"] = int(label)
            rows.append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def load_labeled(path: Path) -> tuple[list[str], list[int], list[str]]:
    texts, labels, sources = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(int(r["label"]))
            sources.append(r.get("source", ""))
    return texts, labels, sources
