"""Canonical ``source`` string tags written to Carlini-style ``samples.jsonl``.

Used by ``mia_eval.generation`` (writers), ``mia_eval.run_carlini_table2`` (per-source P@k),
and ``mia_eval.precision_at_k_by_method`` (column order / detection).
"""

from __future__ import annotations

# Carlini §5.1.1 / §5.1.2-style extraction (``mia_eval.generation``).
CARLINI_EXTRACTION_SOURCES: tuple[str, ...] = (
    "top_k",
    "top_k_internet",
    "temperature_decay",
    "temperature_decay_internet",
    "nucleus",
    "nucleus_internet",
)

# ``memorization_detection/memorization_detection.py`` decoders (decode-time MI-style steering).
MEMORIZATION_DETECTION_SOURCES: tuple[str, ...] = (
    "memorization_baseline",
    "memorization_risk_fast",
    "memorization_risk_slow",
    "memorization_wbc",
    "memorization_wbc_no_infilling",
    "memorization_cheap_logits",
    "memorization_contrastive",
    "memorization_sparse_infilling",
)

# Stable column / iteration order for reporting (subset may appear in any given run).
GENERATION_SOURCES_ORDER: tuple[str, ...] = CARLINI_EXTRACTION_SOURCES + MEMORIZATION_DETECTION_SOURCES
