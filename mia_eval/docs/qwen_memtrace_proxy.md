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

The **coder** add-on uses **`HuggingFaceH4/CodeAlpaca_20K`** (ungated). Older YAML used `bigcode/the-stack` with config `data/python`, which many `datasets` installs no longer expose (only `default`, often gated)—see commented alternatives in `qwen_memtrace_datasets.yaml`.

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

## Train one memTrace RF per Qwen preset (automated)

After `prepare_memtrace_proxy_jsonl`:

```bash
python -m mia_eval.train_memtrace_rf \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/qwen2p5.yaml \
  --proxy-jsonl data/qwen_memtrace_proxy_train.jsonl \
  --output-dir data/memtrace_rfs_qwen \
  --use-yaml-presets
```

Uses **`memtrace_train_presets`** and **`memtrace_rf_train`** in `qwen2p5.yaml`. Writes `<preset>_memtrace_rf.joblib` plus `memtrace_rf_manifest.json`. Each artifact is valid **only** for that preset’s architecture (feature length matches layer/head layout).

Override which checkpoints: `--presets qwen25_7b_base,distil_qwen25_7b_instruct`. Lower VRAM: reduce **`memtrace_rf_train.feature_batch_size`** in YAML.

### Slurm: one GPU job per model

Use a **job array** so each task runs **`--preset <one_model>`** (parallel training; each task writes its own `*_memtrace_rf.joblib` and `*_memtrace_rf_manifest.json`).

Template (edit `#SBATCH` and `cd` for your site):

```bash
mkdir -p logs
export PROXY_JSONL=/path/to/qwen_memtrace_proxy_train.jsonl
export OUT_DIR=/path/to/memtrace_rfs_qwen
export REPO=/path/to/Extracting-training-data
sbatch mia_eval/scripts/slurm/train_memtrace_qwen_array.sbatch
```

The script `mia_eval/scripts/slurm/train_memtrace_qwen_array.sbatch` maps `SLURM_ARRAY_TASK_ID` 0–4 to the five presets (keep in sync with `memtrace_train_presets` in `qwen2p5.yaml`).

**Without arrays:** loop `sbatch` with `--wrap`:

```bash
for p in qwen25_7b_base qwen25_7b_instruct qwen25_7b_coder qwen25_7b_math distil_qwen25_7b_instruct; do
  sbatch --gres=gpu:1 --job-name="mt_$p" --wrap="cd \$REPO && python -m mia_eval.train_memtrace_rf --config mia_eval/config/defaults.yaml --experiment mia_eval/config/qwen2p5.yaml --proxy-jsonl data/qwen_memtrace_proxy_train.jsonl --output-dir data/memtrace_rfs_qwen --preset $p"
done
```

(Escape or export `REPO` so the compute node sees the right path.)

## Training the RF artifact for `--memtrace-rf` (manual)

Alternatively fit on your own `(features, proxy_label)` table, then save either:

- a **`sklearn.pipeline.Pipeline`** `[StandardScaler, RandomForestClassifier]`, or  
- `joblib.dump({"scaler": scaler, "rf": rf}, "qwen_memtrace_rf.joblib")`

Pass the path to `python -m mia_eval.score_sequence --memtrace-rf ...`.
