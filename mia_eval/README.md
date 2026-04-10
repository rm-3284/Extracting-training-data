# MIA evaluation pipeline (infilling score, WBC, memTrace)

This directory runs a **config-driven** pipeline to compare three membership signals against **labels derived from training-data overlap**, using generation and sampling practices aligned with the Carlini et al. extraction setup (see `Extracting-Training-Data-from-Large-Langauge-Models/` in this repo).

## Models (defaults in `config/defaults.yaml`)

| Preset (`active_model`) | Target | Reference (WBC) | Ground-truth index dataset |
|-------------------------|--------|-----------------|----------------------------|
| `gpt_neo_2p7` | `EleutherAI/gpt-neo-2.7B` | `EleutherAI/gpt-neo-125M` | `EleutherAI/the_pile_deduplicated` (streaming) |
| `pythia_2p8` | `EleutherAI/pythia-2.8b` | `EleutherAI/pythia-160m` | same |
| `redpajama_7b` | `togethercomputer/RedPajama-INCITE-7B-Base` | `togethercomputer/RedPajama-INCITE-Base-3B-v1` | `togethercomputer/RedPajama-Data-1T` with **`dataset_config`** (default preset uses `default`; other options include `c4`, `wikipedia`, `github`, …) |

Override any field with a second YAML (`--experiment`) or `--set key=value` (see `run_pipeline.py`).

## Ground truth

1. **Index**: Character shingles (hashed) from a streamed slice of the model’s **public training-style** corpus (configurable).
2. **Labels** (`label` step):
   - `training_excerpt` → **member (1)** (snippet taken from the same corpus stream).
   - `wikipedia_ood` → **non-member (0)** (English Wikipedia text; intended OOD vs Pile).
   - **Generated** samples (`top_k`, `temperature_decay`, `nucleus`) → **member (1)** if a long normalized substring is fully covered by indexed shingles, else **0**.

This is a **practical proxy** for “appears in training”: false positives/negatives depend on `shingle_chars`, `min_match_chars`, and how much of the corpus you index. Increase `max_documents` / `max_shingles` for stricter matching (more RAM/time).

## Hyperparameters

Edit `config/defaults.yaml` under `methods.<name>.hyperparameter_search.grid` (Cartesian product). Defaults aim at a **reasonable benchmarking budget**: 16 infilling configs, 12 WBC configs, 18 memTrace RF configs (trim lists if too slow). Examples:

- **Infilling**: `m`, `k` (see `infilling_score/` and OpenReview paper linked there).
- **WBC**: `min_window`, `max_window`, `num_windows`, or add `window_sizes: [[2,3,4,...]]` as a list of lists in YAML.
- **memTrace**: `n_estimators`, `max_depth`, `min_samples_leaf`, plus `methods.memtrace.max_length`.

Model selection uses **val AUC** (max of AUC with score and with negated score for infilling/WBC); **test AUC** is reported for the chosen hyperparameters.

## Run (from repository root)

```bash
pip install -r mia_eval/requirements.txt

# Full pipeline for one preset
python -m mia_eval.run_pipeline --config mia_eval/config/defaults.yaml --steps all

# Switch model preset without editing files
python -m mia_eval.run_pipeline --config mia_eval/config/defaults.yaml --set active_model=redpajama_7b --steps all

# Merge custom YAML
python -m mia_eval.run_pipeline \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/experiment.example.yaml \
  --steps evaluate
```

Artifacts go to `mia_eval_outputs/<active_model>/`:

- `training_shingle_index.json` — saved index
- `samples.jsonl` — generated + auxiliary texts
- `samples_labeled.jsonl` — adds `label`
- `results.json` — best val/test AUC per method, paths to score files, and `score_orientation` notes
- `evaluation_splits.json` — `train_indices`, `val_indices`, `test_indices` (into the `samples_labeled.jsonl` line order)
- `scores_per_sample.jsonl` — one row per sample: `i`, `label`, `source`, `split`, `infilling_score`, `wbc_score`, `memtrace_p_member`, plus `infilling_score_raw` / `wbc_score_raw` (model output before sign flip). The non-`raw` infilling/WBC fields are **oriented** so **higher ≈ member** on the **validation** split. Join with texts via line `i` in `samples_labeled.jsonl`.

## Steps only

- `build_index` — stream HF data, build shingle set
- `generate` — Carlini-style **top-k** + **temperature decay** (+ optional nucleus), optional **training excerpts** / **Wikipedia** controls
- `label` — apply index + heuristics
- `evaluate` — load labels, grid search, write `results.json`

## Requirements

Large GPUs are recommended for **2.7B–7B** models and especially **memTrace** (`output_attentions=True`). The pipeline loads the target model with **eager attention** for memTrace so SDPA/Flash does not silently break attention features. Use smaller `generation.num_samples_per_strategy`, `methods.memtrace.max_length`, and `float16` in config if you hit OOM.

## References

- Carlini et al., *Extracting Training Data from Large Language Models* (2021); sampling code in this repo’s `Extracting-Training-Data-from-Large-Langauge-Models/`.
- WBC: `wbc_attack/` (arXiv:2601.02751).
- memTrace: `memtrace/` ([EACL 2026 paper](https://aclanthology.org/2026.eacl-long.262.pdf)).
- Infilling score: `infilling_score/` (OpenReview link in that README).
