"""
Optional score inspired by Morris et al., *Approximating Language Model Training Data
from Weights* (arXiv:2506.15553).

The paper's SELECT objective uses inner products between **per-example gradients** and the
**weight difference** (θ_f − θ_0), with greedy batching, JL projections, and synthetic
checkpoints. This module implements only a **lightweight diagnostic**:

  For each text, at the **last non-padding token**, take the closed-form gradient of
  cross-entropy (w.r.t. logits) as ``o = softmax(logits_0) − one_hot(y)``, with **pseudolabel**
  ``y = argmax( logits_f )`` from the finetuned model θ_f. Under a linear LM head
  ``logits = W @ h``, ``∂L/∂W = o ⊗ h``. We report an **unbiased Monte Carlo estimate** of
  ``⟨∂L/∂W, W_f − W_0⟩_F`` using random coordinate pairs ``(i,j)`` shared across the run.

This is **not** the full SELECT pipeline (no greedy subset, no multi-checkpoint trajectory,
no ``torch.func`` last-layer vmap as in the paper). Enable via ``mia_gt_pipeline.select`` in
YAML; see ``mia_eval/docs/morris_2025_select.md``.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mia_eval.model_utils import load_causal_lm, pick_device, resolve_lm_head, torch_dtype_from_str


def _lm_head_weight_matrix(model: torch.nn.Module) -> torch.Tensor:
    head = resolve_lm_head(model)
    w = head.weight
    if w.ndim != 2:
        raise ValueError(f"Expected 2D lm_head.weight, got shape {tuple(w.shape)}")
    return w.detach().float().cpu()


def prepare_monte_carlo_pairs(
    W_f: torch.Tensor,
    W_0: torch.Tensor,
    *,
    n_pairs: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if W_f.shape != W_0.shape:
        raise ValueError(
            f"LM head shapes differ: θ_f {tuple(W_f.shape)} vs θ_0 {tuple(W_0.shape)} "
            "(use same tokenizer / compatible checkpoints)."
        )
    vocab, hidden = int(W_f.shape[0]), int(W_f.shape[1])
    rng = np.random.default_rng(int(random_state))
    idx_i = rng.integers(0, vocab, size=int(n_pairs), endpoint=False)
    idx_j = rng.integers(0, hidden, size=int(n_pairs), endpoint=False)
    wf = W_f.numpy()
    w0 = W_0.numpy()
    delta = (wf[idx_i, idx_j] - w0[idx_i, idx_j]).astype(np.float64)
    return idx_i.astype(np.int64), idx_j.astype(np.int64), delta, vocab, hidden


@torch.inference_mode()
def compute_select_last_layer_scores(
    cfg: Dict[str, Any],
    bundle: Dict[str, Any],
    texts: List[str],
    *,
    base_model_id: str,
    tokenizer_id: str,
    max_length: int,
    n_monte_carlo: int,
    random_state: int,
    batch_size: int = 1,
) -> np.ndarray:
    """
    One scalar per text (Monte Carlo Frobenius inner product ``⟨o⊗h, W_f−W_0⟩`` at the last
    valid **next-token** LM position; see module docstring).
    Loads θ_f = ``bundle['target_model']`` and θ_0 = ``base_model_id`` twice (extract heads,
    then score). VRAM holds both full models briefly during head extraction unless caller
    manages externally — we load ft, extract W_f, delete ft, load base, extract W_0, delete
    base, then reload ft and base for batched scoring (two models). For large 7B+7B, set
    ``select.score_device`` to ``cpu`` in config to force CPU scoring.
    """
    exp = cfg.get("experiment") or {}
    sel = (cfg.get("mia_gt_pipeline") or {}).get("select") or {}
    device = pick_device(sel.get("score_device") or exp.get("device"))
    dtype = torch_dtype_from_str(bundle.get("torch_dtype"))
    ft_id = bundle["target_model"]
    bs = max(1, int(batch_size))

    # --- Extract LM head weights (minimize two full models at once) ---
    print("[select] Loading θ_f to extract LM head…", file=sys.stderr, flush=True)
    model_ft, tok = load_causal_lm(ft_id, tokenizer_id, device, dtype)
    W_f = _lm_head_weight_matrix(model_ft)
    del model_ft
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[select] Loading θ_0 to extract LM head…", file=sys.stderr, flush=True)
    model_0, _ = load_causal_lm(base_model_id, tokenizer_id, device, dtype)
    W_0 = _lm_head_weight_matrix(model_0)
    del model_0
    if device.type == "cuda":
        torch.cuda.empty_cache()

    idx_i, idx_j, delta_np, vocab, hidden = prepare_monte_carlo_pairs(
        W_f, W_0, n_pairs=n_monte_carlo, random_state=random_state
    )
    K = int(len(idx_i))
    scale = float(vocab * hidden) / float(K)
    idx_i_t = torch.as_tensor(idx_i, device=device, dtype=torch.long)
    idx_j_t = torch.as_tensor(idx_j, device=device, dtype=torch.long)
    delta_t = torch.as_tensor(delta_np, device=device, dtype=torch.float32)

    print("[select] Loading θ_f and θ_0 for batched scoring…", file=sys.stderr, flush=True)
    model_ft, tok = load_causal_lm(ft_id, tokenizer_id, device, dtype)
    model_0, _ = load_causal_lm(base_model_id, tokenizer_id, device, dtype)

    out_scores: List[float] = []
    for start in range(0, len(texts), bs):
        batch = texts[start : start + bs]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        bsz = enc["input_ids"].shape[0]
        mask = enc["attention_mask"]
        last_tok = mask.sum(dim=1).clamp(min=1).long() - 1  # index of last real token
        # Causal LM: logits[..., pos, :] predicts token at pos+1; use pos = last_tok - 1 when possible.
        pos_lm = (last_tok - 1).clamp(min=0)

        row_idx = torch.arange(bsz, device=device, dtype=torch.long)
        out_f = model_ft(**enc)
        out_0 = model_0(**enc, output_hidden_states=True)
        lf = out_f.logits[row_idx, pos_lm].float()
        lb = out_0.logits[row_idx, pos_lm].float()
        h_last = out_0.hidden_states[-1][row_idx, pos_lm].float()

        y = lf.argmax(dim=-1)  # pseudolabel from θ_f for this next-token position
        probs = torch.softmax(lb, dim=-1)
        oh = F.one_hot(y, num_classes=vocab).float()
        o = probs - oh

        # (B, K): o[:, idx_i] * h[:, idx_j] * delta
        contrib = o[:, idx_i_t] * h_last[:, idx_j_t] * delta_t.unsqueeze(0)
        scores_b = scale * contrib.sum(dim=-1)
        out_scores.extend([float(x) for x in scores_b.cpu().numpy().tolist()])

    del model_ft, model_0
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.asarray(out_scores, dtype=np.float64)
