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
#SBATCH --array=0-31
#
# Shards one model’s proxy scoring across 32 GPUs (tune --array upper bound to NUM_SHARDS-1).
# Submit once per MODEL after exporting NUM_SHARDS:
#   export MODEL=qwen25_7b_base
#   export NUM_SHARDS=32
#   sbatch jobscript/score_proxy_train_shard_array.slurm
#
# Then merge (CPU is enough):
#   sbatch --export=ALL,MODEL=${MODEL},NUM_SHARDS=32 jobscript/merge_proxy_train_shards.slurm

set -euo pipefail

mkdir -p "logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export HF_HOME=/n/fs/vision-mix/rm4411/huggingface
export TRANSFORMERS_CACHE=/n/fs/vision-mix/rm4411/huggingface
export TORCH_HOME=/n/fs/vision-mix/rm4411/torch
export XDG_CACHE_HOME=/n/fs/vision-mix/rm4411/cache
export CACHE_DIR=/n/fs/vision-mix/rm4411/cache

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
cd "${REPO_ROOT}" || exit 1

MODEL="${MODEL:?Set MODEL, e.g. export MODEL=qwen25_7b_base}"
NUM_SHARDS="${NUM_SHARDS:?Set NUM_SHARDS, e.g. export NUM_SHARDS=32}"

OUT_DIR="data/proxy_scored/${MODEL}/shards"
mkdir -p "${OUT_DIR}"

EXTRA=()
if [[ "${PROXY_PRIMARY_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--primary-only)
fi

# shellcheck source=jobscript/_conda.sh
source "${REPO_ROOT}/jobscript/_conda.sh"
conda activate "${CONDA_ENV:-base}"

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
