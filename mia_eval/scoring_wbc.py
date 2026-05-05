"""WBC attack scores (target vs reference NLL sequences)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from wbc_attack import WBCConfig, wbc_score_from_losses

from .losses import per_token_nll


def _wbc_row_diag(
    tokenizer,
    text: str,
    max_length: int,
    t_loss: List[float],
    r_loss: List[float],
) -> Dict[str, Any]:
    """Per-row stats aligned with ``score_texts`` (``n < 2`` forces score 0)."""
    n = min(len(t_loss), len(r_loss))
    if len(t_loss) > 0:
        input_tokens = len(t_loss) + 1
    else:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_tokens = int(enc["input_ids"].size(1))
    return {
        "wbc_input_tokens": input_tokens,
        "wbc_nll_len": n,
        "wbc_short": bool(n < 2),
    }


def score_texts(
    target_model,
    ref_model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    params: Dict[str, Any],
    max_length: int = 512,
    diag_out: Optional[List[Dict[str, Any]]] = None,
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
        if diag_out is not None:
            diag_out.append(_wbc_row_diag(tokenizer, text, max_length, t_loss, r_loss))
        n = min(len(t_loss), len(r_loss))
        if n < 2:
            scores.append(0.0)
            continue
        scores.append(wbc_score_from_losses(t_loss[:n], r_loss[:n], config=cfg))
    return scores
