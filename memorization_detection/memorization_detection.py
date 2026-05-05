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

# Version 1: slow but more direct candidate-level MI rescoring.
# Cost: top_k infilling calls per generated token.
@torch.no_grad()
def risk_aware_next_token_slow(
    model,
    tokenizer,
    input_ids,
    top_k=5,
    lambda_penalty=0.5,
    temperature=1.0,
    window=64,
    m=5,
    k=0.1,
):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, -1, :]

    base_log_probs = F.log_softmax(logits / temperature, dim=-1)
    topk_log_probs, topk_ids = torch.topk(base_log_probs, k=top_k, dim=-1)

    prefix_text = tokenizer.decode(input_ids[0].detach().cpu(), skip_special_tokens=False)

    risks = []
    for j in range(top_k):
        tok_id = topk_ids[0, j].item()
        tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)
        candidate_text = prefix_text + tok_text

        risk = prefix_infilling_score(
            model, tokenizer, candidate_text, window=window, m=m, k=k
        )
        risks.append(risk)

    risks = torch.tensor(risks, device=input_ids.device, dtype=topk_log_probs.dtype)
    risks = (risks - risks.mean()) / (risks.std() + 1e-6)

    adjusted_log_probs = topk_log_probs[0] - lambda_penalty * risks
    adjusted_probs = F.softmax(adjusted_log_probs, dim=-1)

    sampled_index = torch.multinomial(adjusted_probs, num_samples=1).item()
    return topk_ids[0, sampled_index].item()


# Version 2: fast cached prefix-gated MI decoding.
# Cost: one infilling call every risk_every generated tokens.
RISK_CACHE = {"step": -1, "risk": 0.0}

@torch.no_grad()
def risk_aware_next_token_fast(
    model,
    tokenizer,
    input_ids,
    top_k=5,
    lambda_penalty=0.3,
    temperature=1.0,
    window=24,
    m=1,
    k=0.1,
    risk_every=5,
):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, -1, :]

    base_log_probs = F.log_softmax(logits / temperature, dim=-1)
    topk_log_probs, topk_ids = torch.topk(base_log_probs, k=top_k, dim=-1)

    step = input_ids.shape[1]

    if step % risk_every == 0 or RISK_CACHE["step"] < 0:
        prefix_text = tokenizer.decode(input_ids[0].detach().cpu(), skip_special_tokens=False)

        RISK_CACHE["risk"] = prefix_infilling_score(
            model, tokenizer, prefix_text, window=window, m=m, k=k
        )
        RISK_CACHE["step"] = step

    risk = RISK_CACHE["risk"]

    # Cheap token danger proxy: high-probability tokens are more likely
    # to continue memorized text under a risky prefix.
    token_danger = topk_log_probs[0]
    token_danger = (token_danger - token_danger.mean()) / (token_danger.std() + 1e-6)

    adjusted_log_probs = topk_log_probs[0] - lambda_penalty * risk * token_danger
    adjusted_probs = F.softmax(adjusted_log_probs, dim=-1)

    sampled_index = torch.multinomial(adjusted_probs, num_samples=1).item()
    return topk_ids[0, sampled_index].item()


@torch.no_grad()
def generate_risk_aware(
    prompt,
    model,
    tokenizer,
    max_new_tokens=40,
    top_k=5,
    lambda_penalty=0.3,
    temperature=1.0,
    mode="fast",   # "fast" or "slow"
):
    global RISK_CACHE
    RISK_CACHE = {"step": -1, "risk": 0.0}

    device = next(model.parameters()).device

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    for _ in range(max_new_tokens):
        if mode == "slow":
            next_id = risk_aware_next_token_slow(
                model,
                tokenizer,
                input_ids,
                top_k=top_k,
                lambda_penalty=lambda_penalty,
                temperature=temperature,
            )
        elif mode == "fast":
            next_id = risk_aware_next_token_fast(
                model,
                tokenizer,
                input_ids,
                top_k=top_k,
                lambda_penalty=lambda_penalty,
                temperature=temperature,
            )
        else:
            raise ValueError("mode must be 'fast' or 'slow'")

        next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)

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

    out = model.generate(
        input_ids,
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
            top_k=5,
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
        top_k=5,
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
        top_k=3,
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