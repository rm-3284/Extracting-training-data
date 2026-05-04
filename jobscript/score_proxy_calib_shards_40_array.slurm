#!/bin/bash
#SBATCH --job-name=proxy_calib_s40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%A_%a/out.log
#SBATCH --error=logs/%A_%a/err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#SBATCH --array=0-39
#
# Uses up to 40 GPUs in parallel: 5 Qwen presets × NUM_SHARDS_PER_MODEL shards (default 8).
# Each task scores one disjoint slice of a fixed-size calibration JSONL (default 512 rows).
#
# Prerequisites:
#   CALIB_INPUT must exist (default data/qwen_memtrace_proxy_train_calib512.jsonl). Create once
#   (random sample of 512 non-empty rows; override CALIB_SEED if needed):
#     python -m mia_eval.sample_proxy_jsonl \\
#       --input data/qwen_memtrace_proxy_train.jsonl \\
#       --output data/qwen_memtrace_proxy_train_calib512.jsonl \\
#       -n 512 --seed "${CALIB_SEED:-42}"
#
# After this job completes, run the merge+thresholds stage (5 GPUs or CPU is fine):
#   DEP=$(sbatch --parsable jobscript/score_proxy_calib_shards_40_array.slurm)
#   sbatch --dependency=afterok:${DEP} jobscript/score_proxy_calib_merge_thresholds_array.slurm
#
# Env: NUM_SHARDS_PER_MODEL (default 8), CALIB_INPUT, PROXY_PRIMARY_ONLY, PROXY_BATCH_SIZE,
#      CONDA_ENV

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

NUM_SHARDS_PER_MODEL="${NUM_SHARDS_PER_MODEL:-8}"
NSM="${NUM_SHARDS_PER_MODEL}"
if [[ "$NSM" -lt 1 ]]; then
  echo "NUM_SHARDS_PER_MODEL must be >= 1" >&2
  exit 1
fi

EXPECTED_MAX=$((5 * NSM - 1))
if [[ "${SLURM_ARRAY_TASK_ID}" -gt "$EXPECTED_MAX" ]]; then
  echo "Array task id ${SLURM_ARRAY_TASK_ID} exceeds 5 models × ${NSM} shards - 1 (${EXPECTED_MAX})." >&2
  exit 1
fi

MODELS=(
  qwen25_7b_base
  qwen25_7b_instruct
  qwen25_7b_coder
  qwen25_7b_math
  distil_qwen25_7b_instruct
)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NSM))
SHARD_ID=$((SLURM_ARRAY_TASK_ID % NSM))
MODEL="${MODELS[${MODEL_IDX}]}"

CALIB_INPUT="${CALIB_INPUT:-data/qwen_memtrace_proxy_train_calib512.jsonl}"
if [[ ! -f "$CALIB_INPUT" ]]; then
  echo "Missing ${CALIB_INPUT}. Example: python -m mia_eval.sample_proxy_jsonl --input data/qwen_memtrace_proxy_train.jsonl --output ${CALIB_INPUT} -n 512 --seed \${CALIB_SEED:-42}" >&2
  exit 1
fi

OUT_DIR="data/proxy_scored/${MODEL}/calib_shards"
mkdir -p "${OUT_DIR}"
OUT_SHARD="${OUT_DIR}/calib512_scored_shard$(printf '%04d' "${SHARD_ID}")_of_$(printf '%04d' "${NSM}").jsonl"

EXTRA=()
if [[ "${PROXY_PRIMARY_ONLY:-1}" == "1" ]]; then
  EXTRA+=(--primary-only)
fi

python -m mia_eval.score_proxy_jsonl \
  --input "${CALIB_INPUT}" \
  --output "${OUT_SHARD}" \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/qwen2p5.yaml \
  --preset "${MODEL}" \
  --num-shards "${NSM}" \
  --shard-id "${SHARD_ID}" \
  --batch-size "${PROXY_BATCH_SIZE:-4}" \
  --skip-memtrace \
  "${EXTRA[@]}"

echo "Wrote ${OUT_SHARD}"
