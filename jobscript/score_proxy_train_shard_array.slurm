#!/bin/bash
#SBATCH --job-name=proxy_mia_sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%A_%a/out.log
#SBATCH --error=logs/%A_%a/err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#
# IMPORTANT: The Slurm array size is ONLY this line — it must be ``0`` through ``NUM_SHARDS-1``.
# ``export NUM_SHARDS=64`` does NOT change how many tasks Slurm runs; edit the next line or override on the CLI.
#SBATCH --array=0-63
#
# Examples (run from repo, after ``cd`` to checkout):
#   export MODEL=qwen25_7b_base NUM_SHARDS=64
#   sbatch jobscript/score_proxy_train_shard_array.slurm
#
# 32 shards instead: set ``#SBATCH --array=0-31`` below and ``NUM_SHARDS=32``, or override without editing the file:
#   export MODEL=... NUM_SHARDS=64
#   sbatch --array=0-$((NUM_SHARDS-1)) jobscript/score_proxy_train_shard_array.slurm
#
# Merge after all array tasks finish (match NUM_SHARDS):
#   sbatch --export=ALL,MODEL=${MODEL},NUM_SHARDS=64 jobscript/merge_proxy_train_shards.slurm
#
# Rerun only failed shards (same MODEL / NUM_SHARDS as the original array):
#   sbatch --array=7,19,22 jobscript/score_proxy_train_shard_array.slurm
#   sbatch --export=ALL,MODEL=qwen25_7b_base,NUM_SHARDS=64 --array=10-15 jobscript/score_proxy_train_shard_array.slurm
# Each task overwrites its own ``*_shard$(printf '%04d' TASK_ID)_of_$(printf '%04d' NUM_SHARDS).jsonl``.

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/n/fs/vision-mix/rm4411/Extracting-training-data}"
# shellcheck source=jobscript/_slurm_prologue.sh
source "${PROJECT_ROOT}/jobscript/_slurm_prologue.sh"
conda activate "${CONDA_ENV:-base}"

mkdir -p "logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export HF_HOME=/n/fs/vision-mix/rm4411/huggingface
export TRANSFORMERS_CACHE=/n/fs/vision-mix/rm4411/huggingface
export TORCH_HOME=/n/fs/vision-mix/rm4411/torch
export XDG_CACHE_HOME=/n/fs/vision-mix/rm4411/cache
export CACHE_DIR=/n/fs/vision-mix/rm4411/cache

MODEL="${MODEL:?Set MODEL, e.g. export MODEL=qwen25_7b_base}"
NUM_SHARDS="${NUM_SHARDS:?Set NUM_SHARDS, e.g. export NUM_SHARDS=32}"

if [[ -n "${SLURM_ARRAY_TASK_MAX:-}" ]]; then
  _expected_max=$((NUM_SHARDS - 1))
  if (( SLURM_ARRAY_TASK_MAX != _expected_max )); then
    echo "FATAL: NUM_SHARDS=${NUM_SHARDS} requires Slurm array indices 0..${_expected_max} (inclusive), but this job has SLURM_ARRAY_TASK_MAX=${SLURM_ARRAY_TASK_MAX}." >&2
    echo "Fix: edit #SBATCH --array in this script to 0-${_expected_max}, or submit with: sbatch --array=0-${_expected_max} ..." >&2
    exit 1
  fi
fi

OUT_DIR="data/proxy_scored/${MODEL}/shards"
mkdir -p "${OUT_DIR}"

EXTRA=()
if [[ "${PROXY_PRIMARY_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--primary-only)
fi

python -m mia_eval.score_proxy_jsonl \
  --input data/qwen_memtrace_proxy_train.jsonl \
  --output "${OUT_DIR}/qwen_memtrace_proxy_train_scored_shard$(printf '%04d' "${SLURM_ARRAY_TASK_ID}")_of_$(printf '%04d' "${NUM_SHARDS}").jsonl" \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/qwen2p5.yaml \
  --preset "${MODEL}" \
  --batch-size "${PROXY_BATCH_SIZE:-4}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  "${EXTRA[@]}"
