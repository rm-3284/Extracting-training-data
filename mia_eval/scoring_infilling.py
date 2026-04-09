"""Infilling score MIA (OpenReview); adapted from repo ``infilling_score/infilling_score.py``."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def _get_log_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0))
    logits = outputs.logits[0]
    return F.log_softmax(logits.float(), dim=-1)


def _get_mean_std(log_probs_at_position: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.exp(log_probs_at_position)
    mean = (probs * log_probs_at_position).sum()
    variance = (probs * (log_probs_at_position - mean) ** 2).sum()
    std = torch.sqrt(variance + 1e-8)
    return mean, std


def _infilling_score_token(
    log_probs_real: torch.Tensor,
    model,
    input_ids: torch.Tensor,
    i: int,
    m: int,
) -> torch.Tensor:
    seq_len = input_ids.shape[0]
    log_p_xi = log_probs_real[i - 1, input_ids[i]]
    mean_i, std_i = _get_mean_std(log_probs_real[i - 1])
    r = (log_p_xi - mean_i) / std_i
    x_star_i = torch.argmax(log_probs_real[i - 1]).item()
    log_p_xstar = log_probs_real[i - 1, x_star_i]
    r = r - (log_p_xstar - mean_i) / std_i

    input_ids_star = input_ids.clone()
    input_ids_star[i] = int(x_star_i)
    log_probs_star = _get_log_probs(model, input_ids_star)

    for j in range(i + 1, min(i + m + 1, seq_len)):
        log_p_xj_real = log_probs_real[j - 1, input_ids[j]]
        mean_j, std_j = _get_mean_std(log_probs_real[j - 1])
        r = r + (log_p_xj_real - mean_j) / std_j

        log_p_xj_star = log_probs_star[j - 1, input_ids[j]]
        mean_j_star, std_j_star = _get_mean_std(log_probs_star[j - 1])
        r = r - (log_p_xj_star - mean_j_star) / std_j_star

    return r


def infilling_score(
    model,
    tokenizer,
    text: str,
    m: int = 5,
    k: float = 0.1,
    max_length: int = 512,
) -> float:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).squeeze(0).to(device)
    seq_len = input_ids.shape[0]
    if seq_len < 2:
        return 0.0

    log_probs_real = _get_log_probs(model, input_ids)
    token_scores = []
    for i in range(1, seq_len):
        token_scores.append(_infilling_score_token(log_probs_real, model, input_ids, i, m))
    token_scores_t = torch.stack(token_scores)
    k_count = max(1, int(len(token_scores_t) * k))
    bottom_k = torch.topk(token_scores_t, k_count, largest=False).values
    return float(bottom_k.mean().item())


def score_texts(
    model,
    tokenizer,
    texts: list[str],
    params: Dict[str, Any],
    max_length: int = 512,
) -> list[float]:
    m = int(params.get("m", 5))
    k = float(params.get("k", 0.1))
    return [infilling_score(model, tokenizer, t, m=m, k=k, max_length=max_length) for t in texts]
