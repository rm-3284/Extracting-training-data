"""WBC attack scores (target vs reference NLL sequences)."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from wbc_attack import WBCConfig, wbc_score_from_losses

from .losses import per_token_nll


def score_texts(
    target_model,
    ref_model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    params: Dict[str, Any],
    max_length: int = 512,
) -> List[float]:
    ws = params.get("window_sizes")
    cfg = WBCConfig(
        window_sizes=ws if ws else None,
        min_window=int(params.get("min_window", 2)),
        max_window=int(params.get("max_window", 40)),
        num_windows=int(params.get("num_windows", 10)),
    )
    scores: List[float] = []
    for text in texts:
        t_loss = per_token_nll(target_model, tokenizer, text, device, max_length)
        r_loss = per_token_nll(ref_model, tokenizer, text, device, max_length)
        n = min(len(t_loss), len(r_loss))
        if n < 2:
            scores.append(0.0)
            continue
        scores.append(wbc_score_from_losses(t_loss[:n], r_loss[:n], config=cfg))
    return scores
