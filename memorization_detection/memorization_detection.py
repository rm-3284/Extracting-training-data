import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from infilling_score.infilling_score import infilling_score as compute_infilling_score
import torch.nn.functional as F
MODEL_NAME = "EleutherAI/pythia-70m"  # start tiny; swap up later

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()


if torch.cuda.is_available():
    model = model.cuda()

def suffix_nll(prefix: str, suffix: str):
    """
    Returns:
      avg_nll: average negative log-likelihood over suffix tokens
      total_nll: total negative log-likelihood over suffix tokens
      num_suffix_tokens: number of suffix tokens scored
    """
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

    input_ids = prefix_ids + suffix_ids
    labels = [-100] * len(prefix_ids) + suffix_ids

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    labels = torch.tensor([labels], dtype=torch.long)

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        labels = labels.cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        avg_nll = outputs.loss.item()

    num_suffix_tokens = len(suffix_ids)
    total_nll = avg_nll * num_suffix_tokens
    return avg_nll, total_nll, num_suffix_tokens


# quick sanity check
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

m_avg, m_total, m_n = suffix_nll(member_prefix, member_suffix)
nm_avg, nm_total, nm_n = suffix_nll(nonmember_prefix, nonmember_suffix)

print("Member avg NLL:", m_avg, "total NLL:", m_total, "tokens:", m_n)
print("Nonmember avg NLL:", nm_avg, "total NLL:", nm_total, "tokens:", nm_n)

def prefix_infilling_score(model, tokenizer, prefix_text, window=64, m=5, k=0.1):
    ids = tokenizer.encode(prefix_text, return_tensors="pt").squeeze(0)
    if len(ids) < 4:
        return 0.0

    ids = ids[-window:]
    if torch.cuda.is_available():
        ids = ids.cuda()

    text = tokenizer.decode(ids.detach().cpu(), skip_special_tokens=False)
    return compute_infilling_score(model, tokenizer, text, m=m, k=k)

# Naive risk-aware next-token decoding.
# Rescore the base model's top-k candidates by subtracting
# lambda_penalty * R_i from each candidate log-prob, where
# R_i = prefix_infilling_score(prefix + candidate_token).
# This approximates a modified softmax:
#   q(i | prefix) ∝ p(i | prefix) * exp(-lambda * R_i),
# i.e., downweighting tokens that increase estimated memorization risk.
# This is a cheap approximation to full risk-aware softmax,
# but it is still expensive (one detector call per top-k token)
# and currently uses an uncalibrated, prefix-only risk proxy.
# Ultimately, an optimized, token-wise approach
# or a lightweight model bolted to token generation will
# likely be a more performance-cognizant, clean solution.

# Limitations:
# - Computational cost is O(top_k * detector_cost) per step.
# - R_i is uncalibrated; scale mismatch can over- or under-penalize logits.
# - Prefix-only scoring may misclassify low-entropy but non-memorized text
#   (e.g., formulaic or technical language).
# - Decoding artifacts (tokenization, whitespace) can affect R_i.
#
# Future directions:
# - Learn a calibrated risk model approximating membership log-odds.
# - Replace text-level scoring with token-level or embedding-level proxy.
# - Use short-horizon lookahead (multi-token) instead of 1-step proxy.
# - Implement as a Hugging Face LogitsProcessor for cleaner integration.
@torch.no_grad()
def risk_aware_next_token(
    model,
    tokenizer,
    input_ids,
    top_k=20,
    lambda_penalty=2.0,
    temperature=1.0,
):
    device = input_ids.device
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, -1, :]  # [1, V]

    base_log_probs = F.log_softmax(logits / temperature, dim=-1)
    topk_log_probs, topk_ids = torch.topk(base_log_probs, k=top_k, dim=-1)

    adjusted_log_probs = topk_log_probs.clone()

    prefix_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    for j in range(top_k):
        tok_id = topk_ids[0, j].item()
        tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)

        candidate_text = prefix_text + tok_text

        risk = prefix_infilling_score(candidate_text)   # your current detector
        adjusted_log_probs[0, j] -= lambda_penalty * risk

    adjusted_probs = F.softmax(adjusted_log_probs, dim=-1)
    sampled_index = torch.multinomial(adjusted_probs[0], num_samples=1).item()
    next_token_id = topk_ids[0, sampled_index].item()

    return next_token_id