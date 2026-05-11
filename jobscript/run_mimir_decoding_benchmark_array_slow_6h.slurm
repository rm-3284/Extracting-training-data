#!/bin/bash
#SBATCH --job-name=mimir_slw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=all
#   3 models × 4 example shards = 12 parallel tasks (must match MIMIR_MODEL_KEYS × MIMIR_EXAMPLE_SHARDS).
#   WBC loads a second HF model per task (default small refs below); raise --mem if OOM.
#SBATCH --array=0-11
#SBATCH --output=logs/mimir_slow_%A_%a.out
#SBATCH --error=logs/mimir_slow_%A_%a.err
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#
# Runs baseline + fast + **slow** + **wbc** by splitting each model across several tasks (--example-offset),
# each with a 6h wall limit. Merge when every task finishes:
#   python jobscript/merge_mimir_benchmark_shards.py -o logs/mimir_slow_merged.json \\
#     logs/mimir_slow_shard_${SLURM_ARRAY_JOB_ID}_*.json
#
# Defaults: 4 shards × 6 examples/shard = 24 MIMIR rows per model after merge (cheap slow budget per task).
# Tune without editing #SBATCH --array (keep array size = N_MODELS * N_SHARDS):
#   export MIMIR_EXAMPLE_SHARDS=4      # must match (last_task_id + 1) / N_MODELS
#   export MIMIR_EXAMPLES_PER_SHARD=6   # N_EXAMPLES per task; total ≈ SHARDS * PER_SHARD
#
# WBC defaults (per target model): Pythia → EleutherAI/pythia-70m + MIMIR_WBC_SHARE_TARGET_TOKENIZER=1;
# GPT-Neo 2.7B → EleutherAI/gpt-neo-125M + shared tokenizer. Override with MIMIR_WBC_REFERENCE_MODEL or
# disable defaults entirely with MIMIR_DISABLE_WBC=1 (baseline+fast+slow only).
# Other optional flags (see run_mimir_decoding_benchmark_array.slurm): MIMIR_FAST_LEGACY_LOGPROB,
# MIMIR_INFILLING_PENALTY_SIGN, MIMIR_FAST_INFILLING_*, MIMIR_WBC_LAMBDA, MIMIR_WBC_GATE_EVERY, ...
#
# Submit:
#   mkdir -p logs && export HF_TOKEN="hf_..."
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" jobscript/run_mimir_decoding_benchmark_array_slow_6h.slurm

set -euo pipefail

mkdir -p logs

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/n/fs/vision-mix/rm4411/Extracting-training-data}"
export TMP_BASE="${TMP_BASE:-${PROJECT_ROOT}/tmp}"
mkdir -p "${TMP_BASE}"
if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR}" || ! -w "${TMPDIR}" ]]; then
  export TMPDIR="${TMP_BASE}"
fi
mkdir -p "${TMPDIR}"

# shellcheck source=jobscript/_slurm_prologue.sh
source "${PROJECT_ROOT}/jobscript/_slurm_prologue.sh"

export TMP_BASE="${TMP_BASE:-${PROJECT_ROOT}/tmp}"
mkdir -p "${TMP_BASE}"
if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR}" || ! -w "${TMPDIR}" ]]; then
  export TMPDIR="${TMP_BASE}"
fi
mkdir -p "${TMPDIR}"

# shellcheck source=jobscript/_mimir_bench_env_conda_torch.sh
source "${PROJECT_ROOT}/jobscript/_mimir_bench_env_conda_torch.sh"

JOB_TAG="${SLURM_JOB_ID:-manual}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  JOB_TAG="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
mkdir -p "logs/${JOB_TAG}"

MIMIR_MODEL_KEYS=(gpt_neo_2p7 pythia_2p8 pythia_1p4)
N_MODELS="${#MIMIR_MODEL_KEYS[@]}"
N_SHARDS="${MIMIR_EXAMPLE_SHARDS:-4}"
PER_SHARD="${MIMIR_EXAMPLES_PER_SHARD:-6}"
EXPECTED_TASKS=$((N_MODELS * N_SHARDS))

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${EXPECTED_TASKS}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,$((EXPECTED_TASKS - 1))]. Expected ${EXPECTED_TASKS} tasks (${N_MODELS} models × ${N_SHARDS} shards). Fix #SBATCH --array." >&2
  exit 1
fi

MODEL_IDX=$((TASK_ID % N_MODELS))
SHARD_IDX=$((TASK_ID / N_MODELS))
export MIMIR_BENCH_SINGLE_MODEL="${MIMIR_MODEL_KEYS[$MODEL_IDX]}"

# Window-based comparison (WBC): second model on GPU. Defaults match each MIMIR_BENCH_SINGLE_MODEL; set
# MIMIR_DISABLE_WBC=1 to skip, or set MIMIR_WBC_REFERENCE_MODEL explicitly before sbatch.
if [[ "${MIMIR_DISABLE_WBC:-0}" == "1" ]]; then
  unset MIMIR_WBC_REFERENCE_MODEL || true
  unset MIMIR_MAX_NEW_TOKENS_WBC || true
else
  case "${MIMIR_BENCH_SINGLE_MODEL}" in
    pythia_2p8|pythia_1p4)
      export MIMIR_WBC_REFERENCE_MODEL="${MIMIR_WBC_REFERENCE_MODEL:-EleutherAI/pythia-70m}"
      export MIMIR_WBC_SHARE_TARGET_TOKENIZER="${MIMIR_WBC_SHARE_TARGET_TOKENIZER:-1}"
      ;;
    gpt_neo_2p7)
      export MIMIR_WBC_REFERENCE_MODEL="${MIMIR_WBC_REFERENCE_MODEL:-EleutherAI/gpt-neo-125M}"
      export MIMIR_WBC_SHARE_TARGET_TOKENIZER="${MIMIR_WBC_SHARE_TARGET_TOKENIZER:-1}"
      ;;
  esac
fi
_SLOW_MAX="${MIMIR_SLOW_MAX_NEW:-32}"
if [[ -n "${MIMIR_WBC_REFERENCE_MODEL:-}" ]]; then
  export MIMIR_MAX_NEW_TOKENS_WBC="${MIMIR_MAX_NEW_TOKENS_WBC:-${_SLOW_MAX}}"
fi

EXAMPLE_OFFSET=$((SHARD_IDX * PER_SHARD))
export N_EXAMPLES="${PER_SHARD}"
# Generation budget tuned for ~6h/task with slow decoding; override via MIMIR_SLOW_MAX_NEW, MIMIR_TOP_K, MIMIR_RISK_EVERY.
export BENCH_ARGS="--example-offset ${EXAMPLE_OFFSET} --slow-max-new-tokens ${_SLOW_MAX} --top-k ${MIMIR_TOP_K:-10} --risk-every ${MIMIR_RISK_EVERY:-4}${MIMIR_EXTRA_BENCH:+ ${MIMIR_EXTRA_BENCH}}"

export BENCH_OUTPUT="${BENCH_OUTPUT:-logs/mimir_slow_shard_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json}"

echo "Array task ${TASK_ID}: model=${MIMIR_BENCH_SINGLE_MODEL} shard=${SHARD_IDX}/${N_SHARDS} offset=${EXAMPLE_OFFSET} n=${PER_SHARD} -> ${BENCH_OUTPUT}"
if [[ -n "${MIMIR_WBC_REFERENCE_MODEL:-}" ]]; then
  echo "WBC: reference=${MIMIR_WBC_REFERENCE_MODEL} max_new_tokens_wbc=${MIMIR_MAX_NEW_TOKENS_WBC:-} share_tok=${MIMIR_WBC_SHARE_TARGET_TOKENIZER:-0}"
fi

export MIMIR_BENCH_PROFILE="${MIMIR_BENCH_PROFILE:-slow_6h}"

# shellcheck source=jobscript/_mimir_bench_after_jobtag.sh
source "${PROJECT_ROOT}/jobscript/_mimir_bench_after_jobtag.sh"
