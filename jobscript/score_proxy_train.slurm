#!/bin/bash
#SBATCH --job-name=proxy_mia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%j/output.log
#SBATCH --error=logs/%j/error.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#
# Edit --chdir (and PROJECT_ROOT in jobscript/_slurm_prologue.sh) if your clone path differs.
#
# Score ``data/qwen_memtrace_proxy_train.jsonl`` with infilling + WBC + memtrace
# (``mia_eval.score_proxy_jsonl`` — one process, shared model loads).
#
#   sbatch jobscript/score_proxy_train.slurm
#   sbatch --export=ALL,MODEL=qwen25_7b_base jobscript/score_proxy_train.slurm
#
# Conda: same as a working Slurm script — ``conda`` on PATH + ``conda shell.bash hook``
# (see jobscript/_slurm_prologue.sh). Optional: export CONDA_ENV=your_env
#
# Models: qwen25_7b_base | qwen25_7b_instruct | qwen25_7b_coder | qwen25_7b_math | distil_qwen25_7b_instruct

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/n/fs/vision-mix/rm4411/Extracting-training-data}"
# shellcheck source=jobscript/_slurm_prologue.sh
source "${PROJECT_ROOT}/jobscript/_slurm_prologue.sh"
conda activate "${CONDA_ENV:-base}"

mkdir -p "logs/${SLURM_JOB_ID:-manual}"

export HF_HOME=/n/fs/vision-mix/rm4411/huggingface
export TRANSFORMERS_CACHE=/n/fs/vision-mix/rm4411/huggingface
export TORCH_HOME=/n/fs/vision-mix/rm4411/torch
export XDG_CACHE_HOME=/n/fs/vision-mix/rm4411/cache
export CACHE_DIR=/n/fs/vision-mix/rm4411/cache

MODEL="${MODEL:-qwen25_7b_coder}"
OUT_DIR="data/proxy_scored/${MODEL}"
mkdir -p "${OUT_DIR}"

# Optional: PROXY_PRIMARY_ONLY=1 halves infilling/WBC work (sensitivity HPs skipped).
EXTRA=()
if [[ "${PROXY_PRIMARY_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--primary-only)
fi

python -m mia_eval.score_proxy_jsonl \
  --input data/qwen_memtrace_proxy_train.jsonl \
  --output "${OUT_DIR}/qwen_memtrace_proxy_train_scored.jsonl" \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/qwen2p5.yaml \
  --preset "${MODEL}" \
  --batch-size "${PROXY_BATCH_SIZE:-4}" \
  "${EXTRA[@]}" \
  --calibration-json "${OUT_DIR}/proxy_calibration.json"
