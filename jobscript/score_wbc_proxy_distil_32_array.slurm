#!/bin/bash
#SBATCH --job-name=wbc_distil_32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%A_%a/wbc_distil_out.log
#SBATCH --error=logs/%A_%a/wbc_distil_err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#SBATCH --array=0-31
#
# WBC-only scoring for Distil on the full proxy JSONL, **32-way data parallel** (one GPU per shard).
# Must keep NUM_SHARDS in sync with #SBATCH --array=0-(N-1) (here N=32).
#
#   sbatch jobscript/score_wbc_proxy_distil_32_array.slurm
#
# Env:
#   WBC_INPUT   (default data/qwen_memtrace_proxy_train.jsonl)
#   WBC_OUT_DIR (default data/wbc_only/distil_proxy_shards)
#   NUM_SHARDS  (default 32; must match array size)
#   WBC_BATCH_SIZE   (default 4)
#   WBC_CHUNK_LINES  (default 256)
#   WBC_PRESET       (default distil_qwen25_7b_instruct)
#   CONDA_ENV
#
# Merge shards after all tasks succeed (32 → use *_of_0032.jsonl; change if NUM_SHARDS differs):
#   python -m mia_eval.merge_wbc_jsonl_shards \\
#     --glob 'data/wbc_only/distil_proxy_shards/*_of_0032.jsonl' \\
#     --output data/wbc_only/distil_proxy_train_wbc_merged.jsonl \\
#     --expected-rows 12200 \\
#     --summary-json data/wbc_only/distil_proxy_train_wbc_merged_summary.json

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/n/fs/vision-mix/rm4411/Extracting-training-data}"
source "${PROJECT_ROOT}/jobscript/_slurm_prologue.sh"
conda activate "${CONDA_ENV:-base}"

mkdir -p "logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export HF_HOME=/n/fs/vision-mix/rm4411/huggingface
export TRANSFORMERS_CACHE=/n/fs/vision-mix/rm4411/huggingface
export TORCH_HOME=/n/fs/vision-mix/rm4411/torch
export XDG_CACHE_HOME=/n/fs/vision-mix/rm4411/cache
export CACHE_DIR=/n/fs/vision-mix/rm4411/cache

NUM_SHARDS="${NUM_SHARDS:-32}"
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "NUM_SHARDS must be >= 1" >&2
  exit 1
fi
EXPECTED_MAX=$((NUM_SHARDS - 1))
if [[ "${SLURM_ARRAY_TASK_ID}" -gt "${EXPECTED_MAX}" ]]; then
  echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} > ${EXPECTED_MAX}; widen array or lower NUM_SHARDS." >&2
  exit 1
fi

WBC_INPUT="${WBC_INPUT:-data/qwen_memtrace_proxy_train.jsonl}"
WBC_OUT_DIR="${WBC_OUT_DIR:-data/wbc_only/distil_proxy_shards}"
WBC_PRESET="${WBC_PRESET:-distil_qwen25_7b_instruct}"

if [[ ! -f "${WBC_INPUT}" ]]; then
  echo "Missing WBC_INPUT=${WBC_INPUT}" >&2
  exit 1
fi

mkdir -p "${WBC_OUT_DIR}"
NS_PAD=$(printf '%04d' "${NUM_SHARDS}")
SID_PAD=$(printf '%04d' "${SLURM_ARRAY_TASK_ID}")
OUT_JSONL="${WBC_OUT_DIR}/distil_proxy_wbc_shard${SID_PAD}_of_${NS_PAD}.jsonl"
SUMMARY_JSON="${WBC_OUT_DIR}/distil_proxy_wbc_shard${SID_PAD}_of_${NS_PAD}_summary.json"

python -m mia_eval.score_wbc_jsonl \
  --input "${WBC_INPUT}" \
  --output "${OUT_JSONL}" \
  --preset "${WBC_PRESET}" \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/qwen2p5.yaml \
  --num-shards "${NUM_SHARDS}" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --batch-size "${WBC_BATCH_SIZE:-4}" \
  --chunk-lines "${WBC_CHUNK_LINES:-256}" \
  --summary-json "${SUMMARY_JSON}"

echo "Wrote ${OUT_JSONL}"
