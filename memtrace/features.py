"""
memTrace feature extraction (Makhija et al., EACL 2026).

Implements the feature families described in Sec. 2 (Methodology):
prediction confidence/entropy, attention patterns, layer transitions,
activation patterns, context evolution, and token-position features.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / torch.clamp(den, min=1e-12)


def _masked_stats(
    x: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    """x, mask: (n,). Return mean, std, min, max, median, p25, p75 over valid positions."""
    m = mask.bool()
    if not m.any():
        z = torch.zeros((), device=x.device, dtype=x.dtype)
        return z, z, z, z, z, z, z
    v = x[m]
    return (
        v.mean(),
        v.std(unbiased=False),
        v.min(),
        v.max(),
        v.median(),
        torch.quantile(v.float(), 0.25).to(v.dtype),
        torch.quantile(v.float(), 0.75).to(v.dtype),
    )


def _row_entropy(attn_row: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = attn_row.clamp_min(eps)
    p = p / p.sum()
    return -(p * p.log()).sum()


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (a * b).sum() / (
        a.norm(p=2) * b.norm(p=2) + eps
    )


def extract_memtrace_features_from_tensors(
    hidden_states: Sequence[torch.Tensor],
    attentions: Sequence[torch.Tensor] | None,
    lm_head: torch.nn.Module,
    attention_mask: torch.Tensor | None = None,
    *,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build memTrace feature vector for one sequence (batch size 1).

    Args:
        hidden_states: Tuple from HF ``output_hidden_states=True`` (embedding + each layer).
        attentions: Tuple from HF ``output_attentions=True``, or None to skip attention features.
        lm_head: Maps hidden (..., d) -> (..., vocab). Causal LM ``model.lm_head`` (or tied equivalent).
        attention_mask: (1, seq) with 1 for real tokens, 0 for pad.

    Returns:
        (feature_vector, feature_names)
    """
    if len(hidden_states) < 2:
        raise ValueError("hidden_states must include embedding and at least one layer")

    hs0 = hidden_states[0]
    if hs0.dim() != 3 or hs0.size(0) != 1:
        raise ValueError("Expected hidden states with batch size 1, shape (1, seq, d)")

    dev = device or hs0.device
    mask = (
        attention_mask[0].to(dev).float()
        if attention_mask is not None
        else torch.ones(hs0.size(1), device=dev, dtype=torch.float32)
    )

    features: List[float] = []
    names: List[str] = []

    n = int(mask.sum().item())
    seq_len = hs0.size(1)
    if n < 1:
        raise ValueError("No valid tokens in attention_mask")

    L = len(hidden_states) - 1  # transformer layers

    def add(name: str, t: torch.Tensor) -> None:
        features.append(float(t.detach().cpu()))
        names.append(name)

    # --- Prediction confidence & entropy (per layer via LM head on hidden states) ---
    for l in range(1, len(hidden_states)):
        h = hidden_states[l][0]  # (seq, d)
        logits = lm_head(h)  # (seq, V)
        probs = F.softmax(logits.float(), dim=-1)
        ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)  # (seq,)
        conf, idx = probs.max(dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        gap = top2[:, 0] - top2[:, 1]

        for metric, arr in [("entropy", ent), ("confidence", conf), ("conf_gap", gap)]:
            stats = _masked_stats(arr, mask)
            for s, tag in zip(
                stats,
                ["mean", "std", "min", "max", "median", "p25", "p75"],
            ):
                add(f"pred_l{l}_{metric}_{tag}", s)

        mean_c = _masked_stats(conf, mask)[0]
        std_c = _masked_stats(conf, mask)[1]
        stab = _safe_div(mean_c, std_c + 1e-8)
        add(f"pred_l{l}_confidence_stability", stab)

        for pos_name, idx_t in [
            ("first", 0),
            ("mid", seq_len // 2),
            ("last", seq_len - 1),
        ]:
            if mask[idx_t] > 0:
                add(f"pred_l{l}_entropy_{pos_name}", ent[idx_t])
                add(f"pred_l{l}_confidence_{pos_name}", conf[idx_t])
                add(f"pred_l{l}_conf_gap_{pos_name}", gap[idx_t])
            else:
                add(f"pred_l{l}_entropy_{pos_name}", torch.tensor(0.0, device=dev))
                add(f"pred_l{l}_confidence_{pos_name}", torch.tensor(0.0, device=dev))
                add(f"pred_l{l}_conf_gap_{pos_name}", torch.tensor(0.0, device=dev))

    # --- Attention pattern features ---
    if attentions is not None and len(attentions) == L:
        for l in range(L):
            attn = attentions[l][0]  # (H, n, n) — HF uses keys as columns for probs
            Hh, _, _ = attn.shape
            a_bar = attn.mean(dim=0)  # (n, n)

            row_ents = []
            for t in range(seq_len):
                if mask[t] <= 0:
                    continue
                row_ents.append(_row_entropy(a_bar[t]))
            if row_ents:
                re = torch.stack(row_ents)
                m = torch.ones_like(re)
                for s, tag in zip(
                    _masked_stats(re, m),
                    ["mean", "std", "min", "max", "median", "p25", "p75"],
                ):
                    add(f"attn_l{l}_row_entropy_{tag}", s)
            else:
                for tag in ["mean", "std", "min", "max", "median", "p25", "p75"]:
                    add(f"attn_l{l}_row_entropy_{tag}", torch.tensor(0.0, device=dev))

            conc = []
            dist_num = torch.tensor(0.0, device=dev)
            dist_den = torch.tensor(0.0, device=dev)
            self_tok = []
            prev_tok = []
            for t in range(seq_len):
                if mask[t] <= 0:
                    continue
                row = a_bar[t]
                conc.append(row.max())
                self_tok.append(row[t])
                if t > 0 and mask[t - 1] > 0:
                    prev_tok.append(row[t - 1])
            for t in range(seq_len):
                if mask[t] <= 0:
                    continue
                for s in range(seq_len):
                    if mask[s] <= 0:
                        continue
                    w = a_bar[t, s]
                    dist_num = dist_num + abs(t - s) * w
                    dist_den = dist_den + w

            add(
                f"attn_l{l}_concentration",
                torch.stack(conc).mean() if conc else torch.tensor(0.0, device=dev),
            )
            add(f"attn_l{l}_mean_attention_distance", _safe_div(dist_num, dist_den))
            add(
                f"attn_l{l}_self_attention",
                torch.stack(self_tok).mean() if self_tok else torch.tensor(0.0, device=dev),
            )
            add(
                f"attn_l{l}_prev_token_bias",
                torch.stack(prev_tok).mean() if prev_tok else torch.tensor(0.0, device=dev),
            )

            tau = a_bar[mask > 0].median() if (mask > 0).any() else a_bar.median()
            sparse = (a_bar < tau).float()
            sp_vals = []
            for t in range(seq_len):
                if mask[t] <= 0:
                    continue
                sp_vals.append(sparse[t][mask > 0].mean())
            add(
                f"attn_l{l}_sparsity",
                torch.stack(sp_vals).mean() if sp_vals else torch.tensor(0.0, device=dev),
            )

            for h in range(Hh):
                ah = attn[h]
                hents, focs = [], []
                for t in range(seq_len):
                    if mask[t] <= 0:
                        continue
                    row = ah[t]
                    hents.append(_row_entropy(row))
                    r_valid = row[mask > 0]
                    focs.append(r_valid.max() if r_valid.numel() else row.max())
                if hents:
                    add(
                        f"attn_l{l}_h{h}_entropy_mean",
                        torch.stack(hents).mean(),
                    )
                    add(
                        f"attn_l{l}_h{h}_focus_mean",
                        torch.stack(focs).mean(),
                    )
                else:
                    add(f"attn_l{l}_h{h}_entropy_mean", torch.tensor(0.0, device=dev))
                    add(f"attn_l{l}_h{h}_focus_mean", torch.tensor(0.0, device=dev))

    # --- Layer transition & activation (consecutive layer outputs) ---
    # hidden_states[0]=embed, hidden_states[k]=output after layer k-1.
    for l in range(L - 1):
        h_curr = hidden_states[l + 1][0]
        h_next = hidden_states[l + 2][0]

        diff = (h_next - h_curr).pow(2).sum(dim=-1).sqrt()
        stats = _masked_stats(diff, mask)
        for s, tag in zip(
            stats,
            ["mean", "std", "min", "max", "median", "p25", "p75"],
        ):
            add(f"trans_l{l}_surprise_{tag}", s)

        cos_sims = []
        for t in range(seq_len):
            if mask[t] <= 0:
                continue
            cos_sims.append(_cosine_sim(h_curr[t], h_next[t]))
        if cos_sims:
            cs = torch.stack(cos_sims)
            m = torch.ones_like(cs)
            for s, tag in zip(
                _masked_stats(cs, m),
                ["mean", "std", "min", "max", "median", "p25", "p75"],
            ):
                add(f"trans_l{l}_cos_stability_{tag}", s)
        else:
            for tag in ["mean", "std", "min", "max", "median", "p25", "p75"]:
                add(f"trans_l{l}_cos_stability_{tag}", torch.tensor(0.0, device=dev))

        valid_idx = torch.nonzero(mask > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() > 0:
            diff_v = diff[valid_idx]
            imn = int(diff_v.argmin().item())
            imx = int(diff_v.argmax().item())
            add(
                f"trans_l{l}_argmin_surprise_norm",
                torch.tensor(
                    float(valid_idx[imn].item() / max(seq_len - 1, 1)),
                    device=dev,
                ),
            )
            add(
                f"trans_l{l}_argmax_surprise_norm",
                torch.tensor(
                    float(valid_idx[imx].item() / max(seq_len - 1, 1)),
                    device=dev,
                ),
            )
        else:
            add(f"trans_l{l}_argmin_surprise_norm", torch.tensor(0.0, device=dev))
            add(f"trans_l{l}_argmax_surprise_norm", torch.tensor(0.0, device=dev))

    # --- Activation patterns on each layer output (hidden_states[l+1]) ---
    for l in range(L):
        h_curr = hidden_states[l + 1][0]
        h_abs = h_curr.abs()
        thresh = 1e-6
        sparsity = (h_abs < thresh).float()
        sp_m = []
        for t in range(seq_len):
            if mask[t] <= 0:
                continue
            sp_m.append(sparsity[t].mean())
        add(
            f"act_l{l}_sparsity",
            torch.stack(sp_m).mean() if sp_m else torch.tensor(0.0, device=dev),
        )
        add(
            f"act_l{l}_peak_abs",
            h_curr[mask > 0].abs().max()
            if (mask > 0).any()
            else torch.tensor(0.0, device=dev),
        )

        act_ents = []
        for t in range(seq_len):
            if mask[t] <= 0:
                continue
            z = h_curr[t].float()
            p = F.softmax(z, dim=-1)
            act_ents.append(-(p * p.clamp_min(1e-12).log()).sum())
        if act_ents:
            ae = torch.stack(act_ents)
            m = torch.ones_like(ae)
            for s, tag in zip(
                _masked_stats(ae, m),
                ["mean", "std", "min", "max", "median", "p25", "p75"],
            ):
                add(f"act_l{l}_hidden_entropy_{tag}", s)
        else:
            for tag in ["mean", "std", "min", "max", "median", "p25", "p75"]:
                add(f"act_l{l}_hidden_entropy_{tag}", torch.tensor(0.0, device=dev))

        util = (h_abs > thresh).float()
        ut_m = []
        for t in range(seq_len):
            if mask[t] <= 0:
                continue
            ut_m.append(util[t].mean())
        add(
            f"act_l{l}_neuron_utilization",
            torch.stack(ut_m).mean() if ut_m else torch.tensor(0.0, device=dev),
        )

        hv = h_curr[mask > 0]
        if hv.numel() > 0:
            pos_frac = (hv > 0).float().mean()
            neg_frac = (hv < 0).float().mean()
            add(f"act_l{l}_pos_frac", pos_frac)
            add(f"act_l{l}_neg_frac", neg_frac)
        else:
            add(f"act_l{l}_pos_frac", torch.tensor(0.0, device=dev))
            add(f"act_l{l}_neg_frac", torch.tensor(0.0, device=dev))

    # --- Context evolution (mean hidden state up to i, difference when extending) ---
    for l in range(1, len(hidden_states)):
        h = hidden_states[l][0]  # (seq, d)
        evols = []
        for i in range(1, seq_len):
            if mask[i] <= 0:
                continue
            c_i = h[: i + 1][mask[: i + 1] > 0].mean(dim=0)
            c_prev = h[:i][mask[:i] > 0].mean(dim=0)
            if c_prev.numel() == 0 or c_i.numel() == 0:
                continue
            evols.append((c_i - c_prev).norm(p=2))
        if evols:
            ev = torch.stack(evols)
            m = torch.ones_like(ev)
            for s, tag in zip(
                _masked_stats(ev, m),
                ["mean", "std", "min", "max", "median", "p25", "p75"],
            ):
                add(f"ctx_l{l}_evolution_{tag}", s)
        else:
            for tag in ["mean", "std", "min", "max", "median", "p25", "p75"]:
                add(f"ctx_l{l}_evolution_{tag}", torch.tensor(0.0, device=dev))

    # --- Token-position: first-last similarity per layer; local confidence std on final logits ---
    final_h = hidden_states[-1][0]
    if (mask > 0).sum() >= 2:
        first_i = int(torch.nonzero(mask > 0, as_tuple=False)[0].item())
        last_i = int(torch.nonzero(mask > 0, as_tuple=False)[-1].item())
        for l in range(1, len(hidden_states)):
            hh = hidden_states[l][0]
            add(
                f"pos_l{l}_first_last_cos",
                _cosine_sim(hh[first_i], hh[last_i]),
            )
    else:
        for l in range(1, len(hidden_states)):
            add(f"pos_l{l}_first_last_cos", torch.tensor(0.0, device=dev))

    final_logits = lm_head(final_h)
    probs = F.softmax(final_logits.float(), dim=-1)
    conf_final, _ = probs.max(dim=-1)

    for pos_name, idx_t in [("first", 0), ("mid", seq_len // 2), ("last", seq_len - 1)]:
        c = 2
        lo, hi = max(0, idx_t - c), min(seq_len, idx_t + c + 1)
        window = []
        for t in range(lo, hi):
            if mask[t] > 0:
                window.append(conf_final[t])
        if len(window) >= 2:
            w = torch.stack(window)
            add(f"pos_conf_std_{pos_name}", w.std(unbiased=False))
        else:
            add(f"pos_conf_std_{pos_name}", torch.tensor(0.0, device=dev))

    vec = np.array(features, dtype=np.float64)
    return vec, names
