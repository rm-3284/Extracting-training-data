# Proxy data for memTrace on Qwen (no official training manifest)

Qwen’s public disclosures describe **broad categories** (web, partners, synthetic, etc.), not a reproducible index like The Pile. Any “member” label is a **proxy**: *consistent with having appeared in some large text mix*, not *verified Qwen pretraining*.

## Bundled dataset manifest (this repo)

| File | Purpose |
|------|---------|
| `mia_eval/config/qwen_memtrace_datasets.yaml` | Curated **Parquet-first** HF streams: **member proxies** (FineWeb, FineWeb-Edu, Pile-10k) + **non-member proxies** (AG News, IMDB, FineWeb-Edu **CC-MAIN-2025-** shards). Optional **coder** / **math** member add-ons. |
| `mia_eval/prepare_memtrace_proxy_jsonl.py` | Streams those configs into a single **shuffled JSONL** (`text`, `label`, `stream_id`) for RF training. |

```bash
python -m mia_eval.prepare_memtrace_proxy_jsonl \
  --manifest mia_eval/config/qwen_memtrace_datasets.yaml \
  --output data/qwen_memtrace_proxy_train.jsonl

# For Qwen2.5-Coder / Math experiments, add code + math member text:
python -m mia_eval.prepare_memtrace_proxy_jsonl \
  --manifest mia_eval/config/qwen_memtrace_datasets.yaml \
  --output data/qwen_memtrace_proxy_coder_math.jsonl \
  --include coder math
```

If a **CC-MAIN-2025-** config is renamed on the Hub, edit the YAML shard names. Set **`HF_TOKEN`** for rate limits on large corpora.

## Positive (member-like) proxies

Use **large, timestamped, Parquet-friendly** corpora that plausibly overlap **general** LM pretraining (web, encyclopedic, books-style text).

| Corpus | Role | Notes |
|--------|------|--------|
| **HuggingFaceFW/fineweb** (or **fineweb-edu** subset) | Web-like bulk text | Common in recent LM stacks; check license and snapshot **date** vs your Qwen2.5 **knowledge cutoff**. |
| **mlfoundations/dclm-baseline-1.0** | Open web baseline | Large; good for shingles / overlap heuristics. |
| **NeelNanda/pile-10k** | Small Pile slice | Easy smoke test; **not** “Qwen data,” but cheap for pipeline debugging. |
| **Wikipedia (HF script)** | Encyclopedic | On `datasets>=3.6`, classic `wikipedia` **script loaders fail**; use a **Parquet mirror** (e.g. `wikimedia/wikipedia-20231101` style dumps packaged as Parquet on the Hub if available) or **export** articles yourself. |

**Wikipedia alone** is a weak stand-in for “member”: many models use it, but Qwen also uses much non-Wikipedia data—expect **false negatives**. Prefer **mixing** 2–3 corpora (web + wiki + books) for a broader positive class.

## Negative (non-member-like) proxies — post–Qwen2.5 release

Goal: text whose **first public availability** is plausibly **after** the model’s pretraining cutoff (define the exact checkpoint + date from the model card / tech report).

| Source | Role | Notes |
|--------|------|--------|
| **arXiv / ACL Anthology / OpenReview** papers (version **after** cutoff) | “Too new to be in PT” proxy | You still get boilerplate overlap; use **full paragraphs**, dedupe, and filter very short lines. |
| **News** datasets with **article timestamps** after cutoff | Temporal negative | e.g. stream articles with `published` field; keep only post-cutoff. |
| **Post-cutoff Reddit / forum dumps** (if license OK) | Colloquial OOD | Noisy; good diversity vs encyclopedia. |

Avoid claiming **zero** n-gram overlap with all historical data—use wording like **“unlikely in PT snapshot dated X”**.

## Practical recipe

1. Fix a **cutoff date** per checkpoint (e.g. Qwen2.5-7B base card).
2. **Members**: sample from 1–2 **large static** corpora (FineWeb slice + Parquet Wikipedia mirror).
3. **Non-members**: sample from **timestamped** sources strictly after cutoff; mix domains (paper abstracts + news).
4. Extract memTrace **features** on Qwen **target**; fit **StandardScaler + RF** on train; validate on held-out **proxy** labels and report **caveats** in writing.

## Training the RF artifact for `--memtrace-rf`

Fit on your labeled `(features, proxy_label)` table, then save either:

- a **`sklearn.pipeline.Pipeline`** `[StandardScaler, RandomForestClassifier]`, or  
- `joblib.dump({"scaler": scaler, "rf": rf}, "qwen_memtrace_rf.joblib")`

Pass the path to `python -m mia_eval.score_sequence --memtrace-rf ...`.
