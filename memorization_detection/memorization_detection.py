import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from infilling_score.infilling_score import infilling_score
from datasets import load_dataset



# -----------------------------
# Model config
# -----------------------------

MODEL_CONFIGS = {
    "pythia_2p8": "EleutherAI/pythia-2.8b",
    "gpt_neo_2p7": "EleutherAI/gpt-neo-2.7B",

    # fallback/debug models
    "pythia_70m": "EleutherAI/pythia-70m",
    "pythia_410m": "EleutherAI/pythia-410m",
    "pythia_1p4": "EleutherAI/pythia-1.4b",
}

ACTIVE_MODEL = "gpt_neo_2p7"  # change to "gpt_neo_2p7" later if needed


def load_lm(model_name):
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on: {next(model.parameters()).device}")
    return model, tokenizer


# -----------------------------
# NLL scoring
# -----------------------------

@torch.no_grad()
def suffix_nll(prefix: str, suffix: str, model, tokenizer):
    device = next(model.parameters()).device

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

    input_ids = prefix_ids + suffix_ids
    labels = [-100] * len(prefix_ids) + suffix_ids

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([labels], dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids, labels=labels)
    avg_nll = outputs.loss.item()

    num_suffix_tokens = len(suffix_ids)
    total_nll = avg_nll * num_suffix_tokens

    return avg_nll, total_nll, num_suffix_tokens


# -----------------------------
# Prefix-local infilling score
# -----------------------------

def prefix_infilling_score(model, tokenizer, prefix_text, window=64, m=5, k=0.1):
    device = next(model.parameters()).device

    ids = tokenizer.encode(prefix_text, return_tensors="pt").squeeze(0).to(device)

    if len(ids) < 4:
        return 0.0

    ids = ids[-window:]
    text = tokenizer.decode(ids.detach().cpu(), skip_special_tokens=False)

    return infilling_score(model, tokenizer, text, m=m, k=k)


# -----------------------------
# Risk-aware decoding
# -----------------------------

DEFAULT_DECODE_TOP_K = 20
DEFAULT_GATE_GAMMA = 5.0


def _minmax01_1d(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map 1D tensor to [0, 1] with min -> 0, max -> 1 (equal values -> zeros)."""
    lo = x.min()
    hi = x.max()
    span = hi - lo
    if span < eps:
        return torch.zeros_like(x)
    return (x - lo) / (span + eps)


def _safe_float(x, default: float = 0.0) -> float:
    """Return finite float value or default."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not torch.isfinite(torch.tensor(v)):
        return default
    return v


def _safe_probs_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """
    Build a numerically safe probability vector from log-probs.
    Guarantees finite non-negative probs that sum to 1.
    """
    safe_log_probs = torch.nan_to_num(log_probs, nan=-1e4, posinf=1e4, neginf=-1e4)
    probs = F.softmax(safe_log_probs, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=0.0)
    total = probs.sum()
    if not torch.isfinite(total) or total <= 0:
        return torch.full_like(probs, 1.0 / probs.numel())
    return probs / total


# Version 1: slow — candidate infilling scores with absolute [0,1] weights (not z-scored),
# scaled by a sigmoid gate from prefix-only infilling (intervention only when prefix looks risky).
# Cost: top_k + 1 infilling calls per generated token.
@torch.no_grad()
def risk_aware_next_token_slow(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    top_k=DEFAULT_DECODE_TOP_K,
    lambda_penalty=0.5,
    temperature=1.0,
    window=64,
    m=5,
    k=0.1,
    gate_gamma=DEFAULT_GATE_GAMMA,
):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1, :]

    base_log_probs = F.log_softmax(logits / temperature, dim=-1)
    topk_log_probs, topk_ids = torch.topk(base_log_probs, k=top_k, dim=-1)

    prefix_text = tokenizer.decode(input_ids[0].detach().cpu(), skip_special_tokens=False)

    prefix_risk = prefix_infilling_score(
        model, tokenizer, prefix_text, window=window, m=m, k=k
    )
    prefix_risk = _safe_float(prefix_risk, default=0.0)
    gate = torch.sigmoid(
        torch.tensor(
            gate_gamma * prefix_risk,
            device=input_ids.device,
            dtype=topk_log_probs.dtype,
        )
    )

    risks = []
    for j in range(top_k):
        tok_id = topk_ids[0, j].item()
        tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)
        candidate_text = prefix_text + tok_text

        risk = prefix_infilling_score(
            model, tokenizer, candidate_text, window=window, m=m, k=k
        )
        risks.append(_safe_float(risk, default=prefix_risk))

    risks = torch.tensor(risks, device=input_ids.device, dtype=topk_log_probs.dtype)
    risks = torch.nan_to_num(risks, nan=prefix_risk, posinf=prefix_risk, neginf=0.0)
    g = _minmax01_1d(risks)

    adjusted_log_probs = topk_log_probs[0] - lambda_penalty * gate * g
    adjusted_probs = _safe_probs_from_log_probs(adjusted_log_probs)

    sampled_index = torch.multinomial(adjusted_probs, num_samples=1).item()
    return topk_ids[0, sampled_index].item()


# Version 2: fast — sigmoid(prefix risk) * min-max log-prob danger among top-k.
# Cost: default one infilling call per token (risk_every=1); optional cache if risk_every>1.
RISK_CACHE = {"step": -1, "risk": 0.0}

@torch.no_grad()
def risk_aware_next_token_fast(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    top_k=DEFAULT_DECODE_TOP_K,
    lambda_penalty=0.3,
    temperature=1.0,
    window=24,
    m=1,
    k=0.1,
    risk_every=1,
    gate_gamma=DEFAULT_GATE_GAMMA,
):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1, :]

    base_log_probs = F.log_softmax(logits / temperature, dim=-1)
    topk_log_probs, topk_ids = torch.topk(base_log_probs, k=top_k, dim=-1)

    step = input_ids.shape[1]
    prefix_text = tokenizer.decode(input_ids[0].detach().cpu(), skip_special_tokens=False)

    if risk_every <= 1 or step % risk_every == 0 or RISK_CACHE["step"] < 0:
        RISK_CACHE["risk"] = prefix_infilling_score(
            model, tokenizer, prefix_text, window=window, m=m, k=k
        )
        RISK_CACHE["risk"] = _safe_float(RISK_CACHE["risk"], default=0.0)
        RISK_CACHE["step"] = step

    prefix_risk = _safe_float(RISK_CACHE["risk"], default=0.0)
    gate = torch.sigmoid(
        torch.tensor(
            gate_gamma * prefix_risk,
            device=input_ids.device,
            dtype=topk_log_probs.dtype,
        )
    )

    # Higher base log-prob -> higher danger in [0, 1] (most likely next token penalized most).
    lp = topk_log_probs[0]
    token_danger = _minmax01_1d(lp)

    adjusted_log_probs = lp - lambda_penalty * gate * token_danger
    adjusted_probs = _safe_probs_from_log_probs(adjusted_log_probs)

    sampled_index = torch.multinomial(adjusted_probs, num_samples=1).item()
    return topk_ids[0, sampled_index].item()


@torch.no_grad()
def generate_risk_aware(
    prompt,
    model,
    tokenizer,
    max_new_tokens=40,
    top_k=DEFAULT_DECODE_TOP_K,
    lambda_penalty=0.3,
    temperature=1.0,
    mode="fast",   # "fast" or "slow"
    gate_gamma=DEFAULT_GATE_GAMMA,
    risk_every=1,
):
    global RISK_CACHE
    RISK_CACHE = {"step": -1, "risk": 0.0}

    device = next(model.parameters()).device

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    for _ in range(max_new_tokens):
        if mode == "slow":
            next_id = risk_aware_next_token_slow(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                top_k=top_k,
                lambda_penalty=lambda_penalty,
                temperature=temperature,
                gate_gamma=gate_gamma,
            )
        elif mode == "fast":
            next_id = risk_aware_next_token_fast(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                top_k=top_k,
                lambda_penalty=lambda_penalty,
                temperature=temperature,
                risk_every=risk_every,
                gate_gamma=gate_gamma,
            )
        else:
            raise ValueError("mode must be 'fast' or 'slow'")

        next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)
        next_attn = torch.ones((1, 1), dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat([attention_mask, next_attn], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=False)


@torch.no_grad()
def generate_baseline(
    prompt,
    model,
    tokenizer,
    max_new_tokens=80,
    temperature=1.0,
):
    device = next(model.parameters()).device

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(out[0], skip_special_tokens=False)


# Evaluation helper functions
def split_by_tokens(text, tokenizer, prefix_tokens=64, suffix_tokens=64):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) < prefix_tokens + suffix_tokens:
        return None, None

    prefix_ids = ids[:prefix_tokens]
    suffix_ids = ids[prefix_tokens:prefix_tokens + suffix_tokens]

    prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    suffix = tokenizer.decode(suffix_ids, skip_special_tokens=False)

    return prefix, suffix


def token_overlap_with_suffix(
    generated_text, prefix_text, true_suffix, tokenizer, max_tokens=64
):
    gen_cont = generated_text[len(prefix_text):]

    gen_ids = tokenizer(gen_cont, add_special_tokens=False)["input_ids"][:max_tokens]
    suffix_ids = tokenizer(true_suffix, add_special_tokens=False)["input_ids"][:max_tokens]

    if len(gen_ids) == 0 or len(suffix_ids) == 0:
        return 0.0, 0

    n = min(len(gen_ids), len(suffix_ids))
    matches = sum(int(gen_ids[i] == suffix_ids[i]) for i in range(n))

    # longest exact prefix match
    lcp = 0
    for i in range(n):
        if gen_ids[i] == suffix_ids[i]:
            lcp += 1
        else:
            break

    return matches / n, lcp

# Evaluation
def evaluate_fast_on_mimir(
    rows,
    model,
    tokenizer,
    n_examples=10,
    prefix_tokens=64,
    suffix_tokens=64,
    max_new_tokens=64,
):
    results = []

    for i in range(n_examples):
        text = rows[i]["member"]

        prefix, true_suffix = split_by_tokens(
            text,
            tokenizer,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
        )

        if prefix is None:
            continue

        print(f"\n=== Example {i} ===")

        baseline = generate_baseline(
            prefix,
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
        )

        fast = generate_risk_aware(
            prefix,
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            lambda_penalty=0.3,
            temperature=1.0,
            mode="fast",
        )

        base_overlap, base_lcp = token_overlap_with_suffix(
            baseline,
            prefix,
            true_suffix,
            tokenizer,
            max_tokens=suffix_tokens,
        )

        fast_overlap, fast_lcp = token_overlap_with_suffix(
            fast,
            prefix,
            true_suffix,
            tokenizer,
            max_tokens=suffix_tokens,
        )

        print("baseline overlap:", base_overlap, "baseline LCP:", base_lcp)
        print("fast overlap:", fast_overlap, "fast LCP:", fast_lcp)

        results.append({
            "i": i,
            "baseline_overlap": base_overlap,
            "fast_overlap": fast_overlap,
            "baseline_lcp": base_lcp,
            "fast_lcp": fast_lcp,
        })

    if not results:
        print("No valid examples.")
        return results

    mean_base_overlap = sum(r["baseline_overlap"] for r in results) / len(results)
    mean_fast_overlap = sum(r["fast_overlap"] for r in results) / len(results)

    mean_base_lcp = sum(r["baseline_lcp"] for r in results) / len(results)
    mean_fast_lcp = sum(r["fast_lcp"] for r in results) / len(results)

    print("\n=== SUMMARY ===")
    print("Mean baseline overlap:", mean_base_overlap)
    print("Mean fast overlap:", mean_fast_overlap)
    print("Mean baseline LCP:", mean_base_lcp)
    print("Mean fast LCP:", mean_fast_lcp)

    return results

# -----------------------------
# Quick sanity check
# -----------------------------

def compare_infilling_scores(rows, model, tokenizer, n=20):
    member_scores = []
    nonmember_scores = []

    for i in range(n):
        m_text = rows[i]["member"]
        nm_text = rows[i]["nonmember"]

        m_score = prefix_infilling_score(model, tokenizer, m_text, window=64, m=1, k=0.1)
        nm_score = prefix_infilling_score(model, tokenizer, nm_text, window=64, m=1, k=0.1)

        member_scores.append(m_score)
        nonmember_scores.append(nm_score)

        print(i, "member:", m_score, "nonmember:", nm_score)

    print("mean member:", sum(member_scores) / len(member_scores))
    print("mean nonmember:", sum(nonmember_scores) / len(nonmember_scores))
    

member_prefix = """---
abstract: 'In this paper we extend the deterministic sublinear FFT algorithm for fast reconstruction of $M$-sparse vectors of length $N= 2^J$ considered in [@PWC18]. Our numerical experiences show"""

member_suffix = """ that our modification has a huge impact on the stability of the algorithm, while the runtime of the algorithm is still ${\\mathcal O}(M^2 \\, \\log N)$.'
bibliography:
- 'bibliography.bib'
---

Gerlind """

nonmember_prefix = """---
abstract: 'Interventions of central, top-down planning are serious limitations to the possibility of modelling the dynamics of cities. An example is the city of Paris (France), which during the 19"""

nonmember_suffix = """th century experienced large modifications supervised by a central authority, the ‘Haussmann period’. In this article, we report an empirical analysis of more than 200 years (1789-2010) of the evoluti"""


if __name__ == "__main__":
    model, tokenizer = load_lm(MODEL_CONFIGS[ACTIVE_MODEL])

    print("\n=== NLL sanity check ===")
    m_avg, m_total, m_n = suffix_nll(member_prefix, member_suffix, model, tokenizer)
    nm_avg, nm_total, nm_n = suffix_nll(nonmember_prefix, nonmember_suffix, model, tokenizer)
    print("Member avg NLL:", m_avg, "total NLL:", m_total, "tokens:", m_n)
    print("Nonmember avg NLL:", nm_avg, "total NLL:", nm_total, "tokens:", nm_n)

    print("\n=== Baseline generation ===")
    print(generate_baseline(member_prefix, model, tokenizer, max_new_tokens=40, temperature=1.0))

    print("\n=== MI-guided risk-aware generation: FAST ===")
    print(generate_risk_aware(
        member_prefix,
        model,
        tokenizer,
        max_new_tokens=40,
        lambda_penalty=0.3,
        temperature=1.0,
        mode="fast",
    ))

    print("\n=== MI-guided risk-aware generation: SLOW ===")
    print(generate_risk_aware(
        member_prefix,
        model,
        tokenizer,
        max_new_tokens=20,
        top_k=10,
        lambda_penalty=0.5,
        temperature=1.0,
        mode="slow",
    ))
    
    ds = load_dataset(
        "iamgroot42/mimir",
        "arxiv",
        token=os.environ["HF_TOKEN"],
        trust_remote_code=True,
    )

    rows = ds["ngram_7_0.2"]
    print(compare_infilling_scores(rows, model, tokenizer))
    evaluate_fast_on_mimir(
        rows,
        model,
        tokenizer,
        n_examples=10,
        prefix_tokens=64,
        suffix_tokens=64,
        max_new_tokens=64,
    )