"""Ground-truth membership via substring (character shingle) overlap with training text."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional, Set

WHITESPACE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    return WHITESPACE.sub(" ", s)


def shingle_hashes(text: str, shingle_chars: int) -> Iterator[str]:
    t = normalize_text(text)
    if len(t) < shingle_chars:
        if t:
            yield hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]
        return
    for i in range(0, len(t) - shingle_chars + 1):
        chunk = t[i : i + shingle_chars]
        yield hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]


class TrainingShingleIndex:
    """Stores hashed character n-grams seen in training data."""

    def __init__(self, shingle_chars: int) -> None:
        self.shingle_chars = shingle_chars
        self._hashes: Set[str] = set()

    def __len__(self) -> int:
        return len(self._hashes)

    def add_document(self, text: str, max_new: Optional[int] = None) -> int:
        added = 0
        for h in shingle_hashes(text, self.shingle_chars):
            if h not in self._hashes:
                self._hashes.add(h)
                added += 1
                if max_new is not None and added >= max_new:
                    break
        return added

    def matches(self, text: str, min_match_chars: int) -> bool:
        """True if any contiguous normalized substring of length >= min_match_chars is fully covered by indexed shingles."""
        t = normalize_text(text)
        if len(t) < min_match_chars:
            return any(h in self._hashes for h in shingle_hashes(t, min(len(t), self.shingle_chars)))

        sc = self.shingle_chars
        # Sliding window of length min_match_chars: check if every shingle in window exists
        for start in range(0, len(t) - min_match_chars + 1):
            window = t[start : start + min_match_chars]
            ok = True
            for i in range(0, len(window) - sc + 1):
                chunk = window[i : i + sc]
                hh = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
                if hh not in self._hashes:
                    ok = False
                    break
            if ok:
                return True
        return False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"shingle_chars": self.shingle_chars, "hashes": list(self._hashes)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "TrainingShingleIndex":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        idx = cls(int(payload["shingle_chars"]))
        idx._hashes = set(payload["hashes"])
        return idx


def stream_texts(
    dataset_name: str,
    dataset_config: Optional[str],
    text_field: str,
    max_documents: int,
) -> Iterator[str]:
    from datasets import load_dataset

    # togethercomputer/RedPajama-Data-1T requires a builder config (default, c4, …).
    if dataset_name == "togethercomputer/RedPajama-Data-1T" and not dataset_config:
        dataset_config = "default"

    try:
        if dataset_config:
            ds = load_dataset(
                dataset_name,
                dataset_config,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        else:
            ds = load_dataset(
                dataset_name,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
    except Exception:
        if dataset_config:
            ds = load_dataset(
                dataset_name,
                dataset_config,
                split="train",
                streaming=True,
            )
        else:
            ds = load_dataset(dataset_name, split="train", streaming=True)

    for i, row in enumerate(ds):
        if i >= max_documents:
            break
        text = row.get(text_field) or row.get("content") or ""
        if isinstance(text, str) and len(text) > 50:
            yield text


def build_index_from_hf(
    dataset_name: str,
    dataset_config: Optional[str],
    text_field: str,
    max_documents: int,
    shingle_chars: int,
    max_shingles: int,
) -> TrainingShingleIndex:
    idx = TrainingShingleIndex(shingle_chars)
    for doc in stream_texts(dataset_name, dataset_config, text_field, max_documents):
        remaining = max(0, max_shingles - len(idx))
        if remaining <= 0:
            break
        idx.add_document(doc, max_new=min(50000, remaining))
    return idx
