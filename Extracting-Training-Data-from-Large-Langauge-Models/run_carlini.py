"""
Replicates Carlini et al. Table 2: for each scoring metric, rank generated samples
and report what fraction of top-N are actually memorized (precision@N).
Runs on GPT-Neo, Pythia, and RedPajama using already-labeled samples.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "mia_eval_outputs"

# --- model configs (target + reference, matching defaults.yaml) ----------------------
MODEL_CONFIGS = {
    "gpt_neo_2p7": {
        "target":    "EleutherAI/gpt-neo-2.7B",
        "reference": "EleutherAI/gpt-neo-125M",
    },
    "pythia_2p8": {
        "target":    "EleutherAI/pythia-2.8b",
        "reference": "EleutherAI/pythia-160m",
    },
    "redpajama_7b": {
        "target":    "togethercomputer/RedPajama-INCITE-7B-Base",
        "reference": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
    },
}


# ---- helpers ------------------------------------------------------------------
def load_model(name: str, device: torch.device) -> Tuple:
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    return model, tok


def perplexity(text: str, model, tokenizer, device, max_length: int = 256) -> float:
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return float(torch.exp(loss))


def perplexity_window(
    text: str, model, tokenizer, device, max_length: int = 256, window: int = 50
) -> float:
    ids = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).input_ids.squeeze(0).to(device)
    if ids.shape[0] < window:
        return perplexity(text, model, tokenizer, device, max_length)
    min_ppl = float("inf")
    with torch.no_grad():
        for start in range(ids.shape[0] - window):
            window_ids = ids[start: start + window].unsqueeze(0)
            loss = model(window_ids, labels=window_ids).loss
            min_ppl = min(min_ppl, float(torch.exp(loss)))
    return min_ppl


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int, lower_better: bool) -> float:
    """Fraction of top-k scored samples that are actually memorized."""
    if lower_better:
        top_k_idx = np.argsort(scores)[:k]
    else:
        top_k_idx = np.argsort(scores)[::-1][:k]
    return float(labels[top_k_idx].mean())


def load_labeled(path: Path) -> Tuple[List[str], np.ndarray]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, np.array(labels, dtype=np.int64)


# -- main ----------------------------------------------------------------------
def run_model(model_key: str, device: torch.device):
    labeled_path = OUTPUTS / model_key / "samples_labeled.jsonl"
    if not labeled_path.exists():
        print(f"  Skipping {model_key} — no labeled samples found at {labeled_path}")
        return

    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"{'='*60}")

    texts, labels = load_labeled(labeled_path)
    print(f"Loaded {len(texts)} samples ({labels.sum()} memorized, {(1-labels).sum()} not memorized)")

    cfg = MODEL_CONFIGS[model_key]

    print(f"\nLoading target model: {cfg['target']}...")
    model_t, tok = load_model(cfg["target"], device)

    print(f"Loading reference model: {cfg['reference']}...")
    model_r, _ = load_model(cfg["reference"], device)

    # --- score every sample -----------------------------------------------
    print("\nScoring samples...")
    ppl_t, ppl_r, ppl_lower, ppl_window, zlib_scores = [], [], [], [], []

    for i, text in enumerate(texts):
        if i % 20 == 0:
            print(f"  {i}/{len(texts)}...")

        ppl_t.append(perplexity(text, model_t, tok, device))
        ppl_r.append(perplexity(text, model_r, tok, device))
        ppl_lower.append(perplexity(text.lower(), model_t, tok, device))
        ppl_window.append(perplexity_window(text, model_t, tok, device))
        zlib_scores.append(len(zlib.compress(text.encode("utf-8"))))

    ppl_t = np.array(ppl_t)
    ppl_r = np.array(ppl_r)
    ppl_lower = np.array(ppl_lower)
    ppl_window = np.array(ppl_window)
    zlib_scores = np.array(zlib_scores)

    # ──---- compute metrics (matching Carlini et al. Table 2) ----------------------
    metrics = {
        "Perplexity":  (ppl_t,                          True),   # lower = more memorized
        "Small":       (np.log(ppl_t) / np.log(ppl_r),  True),
        "zlib":        (np.log(ppl_t) / np.log(zlib_scores), True),
        "Lowercase":   (np.log(ppl_t) / np.log(ppl_lower),   True),
        "Window":      (ppl_window,                      True),
    }

    # ---- report precision@k for k = 10, 50, 100 ---------------------------------
    ks = [10, 50, 100]
    ks = [k for k in ks if k <= len(texts)]

    print(f"\n{'Metric':<15}", end="")
    for k in ks:
        print(f"  P@{k:<6}", end="")
    print()
    print("-" * (15 + 10 * len(ks)))

    results = {"model": model_key, "n_samples": len(texts), "metrics": {}}

    for name, (scores, lower_better) in metrics.items():
        print(f"{name:<15}", end="")
        results["metrics"][name] = {}
        for k in ks:
            p = precision_at_k(scores, labels, k, lower_better)
            print(f"  {p:.3f}   ", end="")
            results["metrics"][name][f"precision@{k}"] = p
        print()

    # -- save results -----------------------------------------------
    out_path = OUTPUTS / model_key / "carlini_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=["gpt_neo_2p7", "pythia_2p8", "redpajama_7b"],
        choices=list(MODEL_CONFIGS.keys()),
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for model_key in args.models:
        run_model(model_key, device)