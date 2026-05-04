#!/bin/bash
#SBATCH --job-name=proxy_mia_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%A_%a/output.log
#SBATCH --error=logs/%A_%a/error.log
#SBATCH --partition=all
#SBATCH --array=0-4
#
# One GPU job per model (5 tasks). Submit from repo root:
#   sbatch jobscript/score_proxy_train_all_models_array.slurm

set -euo pipefail

mkdir -p "logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export HF_HOME=/n/fs/vision-mix/rm4411/huggingface
export TRANSFORMERS_CACHE=/n/fs/vision-mix/rm4411/huggingface
export TORCH_HOME=/n/fs/vision-mix/rm4411/torch
export XDG_CACHE_HOME=/n/fs/vision-mix/rm4411/cache
export CACHE_DIR=/n/fs/vision-mix/rm4411/cache

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
cd "${REPO_ROOT}" || exit 1

MODELS=(
  qwen25_7b_base
  qwen25_7b_instruct
  qwen25_7b_coder
  qwen25_7b_math
  distil_qwen25_7b_instruct
)

MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
OUT_DIR="data/proxy_scored/${MODEL}"
mkdir -p "${OUT_DIR}"

EXTRA=()
if [[ "${PROXY_PRIMARY_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--primary-only)
fi

conda activate base

python -m mia_eval.score_proxy_jsonl \
  --input data/qwen_memtrace_proxy_train.jsonl \
  --output "${OUT_DIR}/qwen_memtrace_proxy_train_scored.jsonl" \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/qwen2p5.yaml \
  --preset "${MODEL}" \
  --batch-size "${PROXY_BATCH_SIZE:-4}" \
  "${EXTRA[@]}" \
  --calibration-json "${OUT_DIR}/proxy_calibration.json"
