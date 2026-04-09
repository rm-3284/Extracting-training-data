"""Per-token negative log-likelihoods for causal LMs (WBC, diagnostics)."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


@torch.inference_mode()
def per_token_nll(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 512,
) -> List[float]:
    """
    For tokens x_1..x_n (0-based ids), return NLL[i] = -log p(x_i | x_<i) for i >= 1
    (length n-1, aligned with standard next-token prediction).
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    if input_ids.size(1) < 2:
        return []

    out = model(input_ids=input_ids)
    logits = out.logits[0]  # (seq, vocab)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    ids = input_ids[0]
    nlls: List[float] = []
    for i in range(1, ids.size(0)):
        lp = log_probs[i - 1, ids[i]].item()
        nlls.append(float(-lp))
    return nlls
