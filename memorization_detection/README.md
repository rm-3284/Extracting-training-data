# Prefix-aware MI-guided decoding

Inference-time decoding that downweights “suspicious” next tokens using an **infilling-based risk score** from [`../infilling_score/`](../infilling_score/). No weight updates and no training corpus at decode time. The same style of signal is often used **after** generation (membership-style scoring); here it is applied **during** generation.

---

## How to run

### Prerequisites

- **Python** with `torch`, `transformers`, and `datasets` installed (versions compatible with your GPU / CUDA setup).
- **Hugging Face token** with access to the [MIMIR dataset](https://huggingface.co/datasets/iamgroot42/mimir) (`iamgroot42/mimir`). Request access on the dataset page if needed.

### Environment

From the **repository root** (`Extracting-training-data/`), set the token and use a venv if you have one:

```bash
cd /path/to/Extracting-training-data
export HF_TOKEN="hf_..."   # your Hugging Face token
```

The scripts add the repo root to `sys.path` so `infilling_score` imports resolve when you run from this folder; running as a module from the root also works.

### 1. Preview MIMIR rows (no model load)

Prints prefix/suffix snippets for one example from the `arxiv` config:

```bash
python memorization_detection/preview_data.py
```

Requires `HF_TOKEN`. Uses `huggingface_hub.login` and `datasets.load_dataset`.

### 2. Full memorization / decoding demo (loads an LM)

**Warning:** This downloads a large model (default **GPT-Neo 2.7B** via `ACTIVE_MODEL` in the script) and runs generation plus MIMIR scoring. Use a **GPU** for reasonable speed; on CPU the script still runs but uses `float32` and is slow.

```bash
python memorization_detection/memorization_detection.py
```

What the `__main__` block does, in order:

1. **NLL sanity check** on hardcoded member vs non-member prefix/suffix snippets.
2. **Baseline** `model.generate` on a member-like prefix.
3. **Risk-aware generation** in **fast** and **slow** modes (top-k resampling with infilling-based penalties).
4. **`load_dataset("iamgroot42/mimir", "arxiv", ...)`** on split `ngram_7_0.2`.
5. **`compare_infilling_scores`** — infilling scores on member vs non-member text for the first 20 rows.
6. **`evaluate_fast_on_mimir`** — baseline vs fast decoder overlap / LCP vs true suffix for 10 examples.

### 3. Multi-model benchmark (baseline + fast + slow)

[`run_mimir_decoding_benchmark.py`](run_mimir_decoding_benchmark.py) loads **three open models in sequence** (default: `gpt_neo_2p7`, `pythia_2p8`, `pythia_1p4`), evaluates **baseline**, **fast**, and **slow** on the first **N** MIMIR member rows, and prints mean token overlap and LCP vs the held-out suffix. Writes JSON if you pass `--output`.

```bash
python memorization_detection/run_mimir_decoding_benchmark.py --n-examples 50 --output bench.json
```

**Slow mode** runs several infilling passes **per generated token**; the default `--slow-max-new-tokens 24` keeps runs tractable. Match lengths across modes with e.g. `--max-new-tokens-baseline 32 --max-new-tokens-fast 32 --slow-max-new-tokens 32` when you can afford it. Skip slow entirely: `--skip-slow`.

Other useful flags: `--models gpt_neo_2p7 pythia_1p4`, `--seed 0`, `--verbose`.

### Changing the model

In `memorization_detection.py`, edit:

```python
ACTIVE_MODEL = "gpt_neo_2p7"  # e.g. "pythia_2p8", "pythia_70m", ...
```

Keys are defined in `MODEL_CONFIGS` at the top of the file.

### Using helpers from another script

Call `load_lm(MODEL_CONFIGS[key])` once, then pass `model` and `tokenizer` into `suffix_nll`, `generate_baseline`, `generate_risk_aware`, `split_by_tokens`, `token_overlap_with_suffix`, and `evaluate_fast_on_mimir`. The benchmark script loads `memorization_detection.py` via `importlib` so a normal `import` does not pull in a model.

---

## Files in this directory

| File | Purpose |
|------|---------|
| [`memorization_detection.py`](memorization_detection.py) | LM load, suffix NLL, prefix infilling score, fast/slow risk-aware top-k sampling, baseline generation, MIMIR eval helpers. |
| [`run_mimir_decoding_benchmark.py`](run_mimir_decoding_benchmark.py) | CLI: three default open LMs, more examples, baseline vs fast vs slow, optional JSON. |
| [`preview_data.py`](preview_data.py) | HF login + load MIMIR `arxiv` and print sample prefix/suffix text. |

Scoring calls [`../infilling_score/infilling_score.py`](../infilling_score/infilling_score.py).

---

## What the implementation does

1. **`suffix_nll(prefix, suffix, model, tokenizer)`** — Teacher-forces the suffix after the prefix; returns average and total NLL over suffix tokens (via `labels` masking the prefix).

2. **`prefix_infilling_score(model, tokenizer, ...)`** — Encodes the prefix, keeps the last `window` tokens, decodes to text, runs `infilling_score(model, tokenizer, text, ...)`.

3. **Risk-aware next token**
   - **Slow** (`risk_aware_next_token_slow`): For each of the top-k logits, append that token to the prefix text, compute a prefix infilling score, z-score across the k candidates, subtract `lambda_penalty * risk` from log-probs, softmax, sample. Cost scales with **k infilling evaluations per step**.
   - **Fast** (`risk_aware_next_token_fast`): Every `risk_every` tokens, compute **one** cached infilling score on the current prefix. Penalty uses that scalar times a **normalized log-probability** proxy over the top-k candidates. Cost: **one infilling call every `risk_every` steps**.

4. **`evaluate_fast_on_mimir(rows, model, tokenizer, ...)`** — For MIMIR **member** strings, split into prefix/suffix by token count; compare baseline `generate` vs fast risk-aware continuation using token overlap and longest common prefix (LCP) with the true suffix.

---

## Method (notation without LaTeX delimiters)

The base LM has next-token logits `z_t` and distribution

```
p_theta(v | x_<t) = softmax(z_t)[v]
```

A prefix-level score `s_MI(x_<t)` (here: infilling score) can be turned into a bounded gate

```
r_t = sigmoid( alpha * (s_MI(x_<t) - tau) )
```

with threshold `tau` and sharpness `alpha`. With a per-token term `g_t(v)` (suspiciousness of candidate `v`),

```
z'_t[v] = z_t[v] - lambda * r_t * g_t(v)
p'(v | x_<t) = softmax(z'_t)_v
```

equivalently `p' proportional to p_theta * exp(-lambda * r_t * g_t(v))`. In code, **lambda** is `lambda_penalty`; **g** is infilling-based per candidate (**slow**) or normalized log-prob (**fast**). The idealized sigmoid gate is descriptive; the **fast** path uses a cached scalar risk and a cheap token proxy.

---

## Motivation (short)

- N-gram / Bloom-style filters need training data or huge indexes.
- Activation steering needs layer picks and hidden-state edits.
- This path uses **decode-time** behavioral scores (infilling) only.

---

## Roadmap vs code

| Phase | In code |
|-------|---------|
| Data & scoring (MIMIR, NLL, infilling) | Partially: NLL, infilling, MIMIR in `__main__` |
| Surrogate risk on hidden states | Not implemented |
| Logit reweighting | Yes: fast / slow decoders |
| Eval (overlap, LCP) | Yes: `evaluate_fast_on_mimir` |

---

## Scope

This does **not** claim perfect memorization detection. The hypothesis is that MI-style / infilling signals can act as **prefix-level risk proxies** for selective decoding-time mitigation.

---

## Relation to the rest of the repo

Same broad theme as membership-inference work elsewhere in the project, but applied **while** generating text, not only to score finished outputs.
