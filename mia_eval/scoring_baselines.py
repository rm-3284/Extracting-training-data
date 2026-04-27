"""Carlini et al. baseline scoring methods: perplexity, zlib, lowercase, window."""

from __future__ import annotations

import zlib
from typing import List

import torch

from .losses import per_token_nll


def _perplexity(nlls: List[float]) -> float:
    """Average NLL = log perplexity."""
    if not nlls:
        return 0.0
    return sum(nlls) / len(nlls)


def _zlib_entropy(text: str) -> float:
    """Length in bits of zlib-compressed text."""
    return len(zlib.compress(text.encode("utf-8"))) * 8


def score_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
) -> List[float]:
    """Lower perplexity = more likely memorized (scores are negated so higher = more likely member)."""
    scores = []
    for text in texts:
        nlls = per_token_nll(model, tokenizer, text, device, max_length)
        scores.append(-_perplexity(nlls))
    return scores


def score_zlib(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
) -> List[float]:
    """Ratio of model perplexity to zlib entropy. Higher = more likely memorized."""
    scores = []
    for text in texts:
        nlls = per_token_nll(model, tokenizer, text, device, max_length)
        perp = _perplexity(nlls)
        zlib_e = _zlib_entropy(text)
        scores.append(perp / zlib_e if zlib_e > 0 else 0.0)
    return scores


def score_lowercase(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
) -> List[float]:
    """Ratio of perplexity on original vs lowercased text. Higher = more likely memorized."""
    scores = []
    for text in texts:
        nlls_orig = per_token_nll(model, tokenizer, text, device, max_length)
        nlls_lower = per_token_nll(model, tokenizer, text.lower(), device, max_length)
        perp_orig = _perplexity(nlls_orig)
        perp_lower = _perplexity(nlls_lower)
        scores.append(perp_orig / perp_lower if perp_lower > 0 else 0.0)
    return scores


def score_window(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
    window_size: int = 50,
) -> List[float]:
    """Minimum average NLL over a sliding window of tokens. Higher = more likely memorized."""
    scores = []
    for text in texts:
        nlls = per_token_nll(model, tokenizer, text, device, max_length)
        if len(nlls) < window_size:
            scores.append(-_perplexity(nlls))
            continue
        min_window_perp = min(
            sum(nlls[i: i + window_size]) / window_size
            for i in range(len(nlls) - window_size + 1)
        )
        scores.append(-min_window_perp)
    return scores