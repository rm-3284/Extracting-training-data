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

## Score a sequence (no shingle pipeline)

For **Qwen2.5** and other targets, `score_sequence.py` runs **infilling**, **WBC**, and **memTrace features** on demand (optional `joblib` RF for `p_member`). Presets live in `config/qwen2p5.yaml`; proxy-data guidance for memTrace RF training is in `docs/qwen_memtrace_proxy.md`.

```bash
python -m mia_eval.score_sequence \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/qwen2p5.yaml \
  --preset qwen25_7b_base \
  --text "Your candidate string here."
```

**memTrace proxy JSONL** (for RF training on Qwen checkpoints): see `config/qwen_memtrace_datasets.yaml` and run `python -m mia_eval.prepare_memtrace_proxy_jsonl --output data/qwen_memtrace_proxy_train.jsonl` (details in `docs/qwen_memtrace_proxy.md`). Then train one RF per preset: `python -m mia_eval.train_memtrace_rf --config mia_eval/config/defaults.yaml --experiment mia_eval/config/qwen2p5.yaml --proxy-jsonl data/qwen_memtrace_proxy_train.jsonl --output-dir data/memtrace_rfs_qwen --use-yaml-presets`.

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

### MIA reference scores (no shingle ground truth)

For presets **without** `ground_truth` (e.g. Qwen in `config/qwen2p5.yaml`), use **`--steps all_mia_gt`** (or `generate,mia_annotate,mia_evaluate`). Each sample gets two triples of scores — **infilling**, **WBC**, **memTrace `p_member`** — at HPs from `mia_gt_pipeline` in the experiment YAML (open-model transferred infilling/WBC + pre-trained `*_memtrace_rf.joblib`). Outputs: `samples_mia_gt.jsonl`, `results_mia_gt.json` (Spearman matrices + mean abs delta between primary vs sensitivity combo). See `mia_eval/mia_gt_pipeline.py`.

Optional **Morris et al. (2025)** arXiv:2506.15553–inspired diagnostic: set `mia_gt_pipeline.select.enabled: true` and `base_model` (θ₀) vs target θ_f; adds **`select_alignment_mc`** and extends Spearman in `results_mia_gt.json`. This is **not** the full SELECT / greedy / JL pipeline — see `mia_eval/docs/morris_2025_select.md`.

### Carlini et al. (2021) Table 2 scores (Qwen and any preset)

The extraction repo’s ``run_carlini.py`` only lists Neo / Pythia / RedPajama. For **Qwen** (or any ``models`` entry with ``target_model`` + ``reference_model``), run:

```bash
python -m mia_eval.run_carlini_table2 \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/qwen2p5.yaml \
  --set active_model=qwen25_7b_base \
  --input mia_eval_outputs/qwen25_7b_base/samples.jsonl \
  --scores-jsonl mia_eval_outputs/qwen25_7b_base/carlini_table2_scores.jsonl
```

Omit ``--input`` to default to ``mia_eval_outputs/<active_model>/samples.jsonl``. Writes ``carlini_table2.json``. If every line has integer ``label``, reports **precision@k** (true memorization labels). If labels are missing but every line has ``mia_gt_primary`` with ``infilling``, ``wbc``, and ``memtrace_p_member`` (e.g. ``samples_mia_gt.jsonl``), also writes **``proxy_precision_at_k``**: same P@k ranking, but pseudo-labels from **median splits** on those three MIA scores (see ``proxy_precision_at_k_note`` in the JSON). Otherwise only **aggregate** stats. Carlini scores are **not** all in ``[0,1]``. Same metrics as ``Extracting-Training-Data-from-Large-Langauge-Models/run_carlini.py``.

## Requirements

Large GPUs are recommended for **2.7B–7B** models and especially **memTrace** (`output_attentions=True`). The pipeline loads the target model with **eager attention** for memTrace so SDPA/Flash does not silently break attention features. Use smaller `generation.num_samples_per_strategy`, `methods.memtrace.max_length`, and `float16` in config if you hit OOM.

## References

- Carlini et al., *Extracting Training Data from Large Language Models* (2021); sampling code in this repo’s `Extracting-Training-Data-from-Large-Langauge-Models/`.
- WBC: `wbc_attack/` (arXiv:2601.02751).
- memTrace: `memtrace/` ([EACL 2026 paper](https://aclanthology.org/2026.eacl-long.262.pdf)).
- Infilling score: `infilling_score/` (OpenReview link in that README).
