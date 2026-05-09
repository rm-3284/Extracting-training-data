"""Ground-truth membership via substring (character shingle) overlap with training text."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

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
    *,
    hub_extra: Optional[Dict[str, Any]] = None,
) -> Iterator[str]:
    from datasets import load_dataset

    hub_extra = dict(hub_extra or {})

    # togethercomputer/RedPajama-Data-1T requires a builder config (default, c4, …).
    if dataset_name == "togethercomputer/RedPajama-Data-1T" and not dataset_config:
        dataset_config = "default"

    kw = dict(
        split="train",
        streaming=True,
        trust_remote_code=True,
        **hub_extra,
    )

    try:
        if dataset_config:
            ds = load_dataset(dataset_name, dataset_config, **kw)
        else:
            ds = load_dataset(dataset_name, **kw)
    except Exception:
        kw.pop("trust_remote_code", None)
        if dataset_config:
            ds = load_dataset(dataset_name, dataset_config, **kw)
        else:
            ds = load_dataset(dataset_name, **kw)

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


def _hub_extra_from_source(source: Dict[str, Any]) -> Dict[str, Any]:
    """Optional HF ``load_dataset`` kwargs from YAML (verification_mode, revision, …)."""
    out: Dict[str, Any] = {}
    if source.get("verification_mode") is not None:
        out["verification_mode"] = source["verification_mode"]
    if source.get("revision") is not None:
        out["revision"] = source["revision"]
    extra = source.get("load_dataset_kwargs")
    if isinstance(extra, dict):
        out.update(extra)
    return out


def _load_dataset_stream(
    dataset_name: str,
    dataset_config: Optional[str],
    *,
    data_files: Any = None,
    data_dir: Any = None,
    hub_extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """``split=train`` streaming, with optional Parquet/JSONL ``data_files`` or ``data_dir`` (e.g. The Stack)."""
    from datasets import load_dataset

    if dataset_name == "togethercomputer/RedPajama-Data-1T" and not dataset_config:
        dataset_config = "default"

    base_kw: Dict[str, Any] = {
        "split": "train",
        "streaming": True,
        "trust_remote_code": True,
    }
    base_kw.update(dict(hub_extra or {}))
    if data_files is not None:
        base_kw["data_files"] = data_files
    if data_dir is not None:
        base_kw["data_dir"] = data_dir

    def _try(kw: Dict[str, Any]) -> Any:
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, **kw)
        return load_dataset(dataset_name, **kw)

    try:
        return _try(dict(base_kw))
    except Exception:
        kw2 = dict(base_kw)
        kw2.pop("trust_remote_code", None)
        return _try(kw2)


def _unwrap_stream(ds: Any) -> Any:
    """If ``load_dataset`` returns a ``DatasetDict``, use ``train``."""
    try:
        from datasets import DatasetDict

        if isinstance(ds, DatasetDict) and "train" in ds:
            return ds["train"]
    except Exception:
        pass
    if isinstance(ds, dict) and "train" in ds:
        return ds["train"]
    return ds


def _extract_doc_text(row: Dict[str, Any], source: Dict[str, Any]) -> str:
    """Pull document text from a dataset row (plain ``text`` / ``content`` or chat ``messages``)."""
    if source.get("flatten_messages"):
        msgs = row.get("messages")
        if isinstance(msgs, list):
            parts: List[str] = []
            for m in msgs:
                if isinstance(m, dict):
                    parts.append(str(m.get("content", "")))
            return "\n".join(parts)
    tf = str(source.get("text_field", "text"))
    text = row.get(tf) or row.get("content") or ""
    return text if isinstance(text, str) else ""


def iter_training_documents(source: Dict[str, Any]) -> Iterator[str]:
    """
    Yield text from a single **open** training-corpus spec (from YAML ``sources``).

    Two modes:
      * **Text rows:** standard HF ``text`` (or ``content``) field.
      * **Token rows:** ``tokenizer_for_decode`` + ``token_field`` (e.g. LLM360/AmberDatasets
        ``token_ids``) — decode with that model's tokenizer to text, then shingle.
    """
    max_documents = int(source.get("max_documents", 2000))
    tok_ref = source.get("tokenizer_for_decode") or source.get("tokenizer_model_id")
    hub = _hub_extra_from_source(source)

    if tok_ref:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(tok_ref), trust_remote_code=True)
        token_field = str(source.get("token_field", "token_ids"))
        ds_name = str(source["dataset_name"])
        ds_cfg = source.get("dataset_config")
        if ds_cfg in ("", None):
            ds_cfg = None
        data_files = source.get("data_files")

        raw = None
        last_err: Optional[BaseException] = None
        attempts: List[Dict[str, Any]] = [
            dict(split="train", streaming=True, trust_remote_code=True),
            dict(streaming=True, trust_remote_code=True),
        ]
        if data_files is not None:
            for a in attempts:
                a["data_files"] = data_files
        for a in attempts:
            a.update(hub)
        for extra in attempts:
            try:
                if ds_cfg is not None:
                    raw = load_dataset(ds_name, ds_cfg, **extra)
                else:
                    raw = load_dataset(ds_name, **extra)
                break
            except Exception as e:
                last_err = e
        if raw is None:
            raise RuntimeError(
                f"Could not stream {ds_name!r} with tokenizer decode (data_files={data_files!r})"
            ) from last_err
        ds = _unwrap_stream(raw)
        for i, row in enumerate(ds):
            if i >= max_documents:
                break
            ids = row.get(token_field)
            if ids is None:
                continue
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if isinstance(ids, list) and len(ids) > 0:
                text = tok.decode(ids)
                if len(text) > 50:
                    yield text
        return

    dataset_name = str(source["dataset_name"])
    ds_cfg = source.get("dataset_config")
    if ds_cfg in ("", None):
        ds_cfg = None
    text_field = str(source.get("text_field", "text"))
    data_files = source.get("data_files")
    data_dir = source.get("data_dir")

    if data_files is not None or data_dir is not None:
        raw = _load_dataset_stream(
            dataset_name,
            ds_cfg,
            data_files=data_files,
            data_dir=data_dir,
            hub_extra=hub,
        )
        ds = _unwrap_stream(raw)
        for i, row in enumerate(ds):
            if i >= max_documents:
                break
            text = _extract_doc_text(row, source)
            if isinstance(text, str) and len(text) > 50:
                yield text
        return

    if source.get("flatten_messages"):
        raw = _load_dataset_stream(dataset_name, ds_cfg, hub_extra=hub)
        ds = _unwrap_stream(raw)
        for i, row in enumerate(ds):
            if i >= max_documents:
                break
            r = dict(row) if not isinstance(row, dict) else row
            text = _extract_doc_text(r, source)
            if isinstance(text, str) and len(text) > 50:
                yield text
        return

    for doc in stream_texts(
        dataset_name, ds_cfg, text_field, max_documents, hub_extra=hub or None
    ):
        yield doc


def build_index_from_training_sources(
    sources: List[Dict[str, Any]],
    shingle_chars: int,
    max_shingles: int,
) -> TrainingShingleIndex:
    """
    Build a shingle index by streaming **multiple** public training sources in order
    (model cards / Hub releases) until ``max_shingles`` unique hashes are stored.
    """
    if not sources:
        raise ValueError("training sources list is empty")
    idx = TrainingShingleIndex(shingle_chars)
    for spec in sources:
        s = spec if isinstance(spec, dict) else dict(spec)
        for doc in iter_training_documents(s):
            remaining = max(0, max_shingles - len(idx))
            if remaining <= 0:
                return idx
            idx.add_document(doc, max_new=min(50_000, remaining))
    return idx
