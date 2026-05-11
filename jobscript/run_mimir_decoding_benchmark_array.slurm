#!/bin/bash
#SBATCH --job-name=mimir_b_a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=all
#SBATCH --array=0-2
#SBATCH --output=logs/mimir_bench_%A_%a.out
#SBATCH --error=logs/mimir_bench_%A_%a.err
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#
# One Slurm array task per model (default: GPT-Neo 2.7B, Pythia 2.8B, Pythia 1.4B).
# Each task writes its own JSON; merge when all finish:
#   python jobscript/merge_mimir_benchmark_shards.py -o logs/mimir_bench_merged.json \\
#     logs/mimir_shard_<ARRAY_JOB_ID>_*.json
#
# Same prerequisites as run_mimir_decoding_benchmark.slurm (conda env, HF_TOKEN, etc.).
# For baseline+fast+slow within ~6h per GPU: jobscript/run_mimir_decoding_benchmark_array_slow_6h.slurm
# (12 tasks: 3 models × 4 example shards), then merge. This file stays 3 tasks (skip-slow PROFILE=6h).
#
# Optional overrides (same as monolithic job): CONDA_ENV, N_EXAMPLES, BENCH_OUTPUT,
# SKIP_SLOW, BENCH_ARGS, HF_HOME, ...
#
# Decoding / benchmark flags (passed via jobscript/_mimir_bench_after_jobtag.sh → EXTRA):
#   MIMIR_FAST_LEGACY_LOGPROB=1     --fast-legacy-logprob (old fast heuristic)
#   MIMIR_INFILLING_PENALTY_SIGN    e.g. -1 to flip infilling penalty direction
#   MIMIR_FAST_INFILLING_WINDOW/M/K  infilling hyperparameters for fast/wbc paths
#   MIMIR_WBC_REFERENCE_MODEL       HF id to enable WBC mode (loads 2nd model; raise --mem if OOM)
#   MIMIR_WBC_LAMBDA, MIMIR_WBC_INFILLING_LAMBDA, MIMIR_WBC_GATE_GAMMA, MIMIR_WBC_GATE_EVERY
#   MIMIR_MAX_NEW_TOKENS_WBC        --max-new-tokens-wbc
#   MIMIR_WBC_SHARE_TARGET_TOKENIZER=1  --wbc-share-target-tokenizer (same-vocab Pythia family)
#
# Per-task output default (override with BENCH_OUTPUT):
#   logs/mimir_shard_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json
#
# Submit:
#   mkdir -p logs && export HF_TOKEN="hf_..."
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" jobscript/run_mimir_decoding_benchmark_array.slurm

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
if [[ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#MIMIR_MODEL_KEYS[@]}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for MIMIR_MODEL_KEYS (${#MIMIR_MODEL_KEYS[@]} models). Fix #SBATCH --array." >&2
  exit 1
fi
export MIMIR_BENCH_SINGLE_MODEL="${MIMIR_MODEL_KEYS[$SLURM_ARRAY_TASK_ID]}"

export BENCH_OUTPUT="${BENCH_OUTPUT:-logs/mimir_shard_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: model=${MIMIR_BENCH_SINGLE_MODEL} -> ${BENCH_OUTPUT}"

export MIMIR_BENCH_PROFILE="${MIMIR_BENCH_PROFILE:-6h}"

# shellcheck source=jobscript/_mimir_bench_after_jobtag.sh
source "${PROJECT_ROOT}/jobscript/_mimir_bench_after_jobtag.sh"
