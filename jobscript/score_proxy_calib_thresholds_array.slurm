#!/bin/bash
#SBATCH --job-name=proxy_calib_thr
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
#SBATCH --array=0-4
#
# One GPU task per Qwen preset: score CAL_N proxy rows (cheap), then write thresholds JSON
# (infilling + WBC; memtrace fixed at MEMTRACE_P_FIXED).
#
# Default: first CAL_N rows from CAL_SOURCE. For a random subset instead (recommended), create once:
#   python -m mia_eval.sample_proxy_jsonl --input data/qwen_memtrace_proxy_train.jsonl \\
#     --output data/qwen_memtrace_proxy_train_calib512.jsonl -n 512 --seed 42
# then submit with CALIB_INPUT=data/qwen_memtrace_proxy_train_calib512.jsonl (omit max-rows on full file).
#
# Optional env: CAL_N (default 512), CAL_SOURCE, CALIB_INPUT, PROXY_PRIMARY_ONLY=1, PROXY_BATCH_SIZE=4,
# MEMTRACE_P_FIXED=0.5
#
#   sbatch jobscript/score_proxy_calib_thresholds_array.slurm

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

MODELS=(
  qwen25_7b_base
  qwen25_7b_instruct
  qwen25_7b_coder
  qwen25_7b_math
  distil_qwen25_7b_instruct
)

MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
CAL_N="${CAL_N:-512}"
MT_FIX="${MEMTRACE_P_FIXED:-0.5}"

OUT_DIR="data/proxy_scored/${MODEL}"
mkdir -p "${OUT_DIR}"
CAL_JSONL="${OUT_DIR}/calib_${CAL_N}.jsonl"
THR_JSON="${OUT_DIR}/proxy_thresholds_inf_wbc_mtfix.json"

CAL_SOURCE="${CAL_SOURCE:-data/qwen_memtrace_proxy_train.jsonl}"
if [[ -n "${CALIB_INPUT:-}" ]]; then
  PROXY_IN="${CALIB_INPUT}"
  ROW_ARGS=()
else
  PROXY_IN="${CAL_SOURCE}"
  ROW_ARGS=(--max-rows "${CAL_N}")
fi

EXTRA=()
if [[ "${PROXY_PRIMARY_ONLY:-1}" == "1" ]]; then
  EXTRA+=(--primary-only)
fi

python -m mia_eval.score_proxy_jsonl \
  --input "${PROXY_IN}" \
  --output "${CAL_JSONL}" \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/qwen2p5.yaml \
  --preset "${MODEL}" \
  "${ROW_ARGS[@]}" \
  --batch-size "${PROXY_BATCH_SIZE:-4}" \
  --skip-memtrace \
  "${EXTRA[@]}"

python -m mia_eval.compute_proxy_thresholds \
  --input "${CAL_JSONL}" \
  --output "${THR_JSON}" \
  --memtrace-p-fixed "${MT_FIX}"

echo "Wrote ${THR_JSON}"
