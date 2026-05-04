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
#
# Score ``data/qwen_memtrace_proxy_train.jsonl`` with infilling + WBC + memtrace
# (``mia_eval.score_proxy_jsonl`` — one process, shared model loads).
#
# Submit from repo root (so paths resolve), or set REPO_ROOT:
#   sbatch jobscript/score_proxy_train.slurm
#   sbatch --export=ALL,MODEL=qwen25_7b_base jobscript/score_proxy_train.slurm
#
# If conda is not under ~/miniconda3|~/anaconda3|..., set e.g.:
#   export CONDA_ROOT=/n/fs/.../miniconda3
# Optional env name (default base): export CONDA_ENV=cos484
#
# Models: qwen25_7b_base | qwen25_7b_instruct | qwen25_7b_coder | qwen25_7b_math | distil_qwen25_7b_instruct

set -euo pipefail

mkdir -p "logs/${SLURM_JOB_ID:-manual}"

export HF_HOME=/n/fs/vision-mix/rm4411/huggingface
export TRANSFORMERS_CACHE=/n/fs/vision-mix/rm4411/huggingface
export TORCH_HOME=/n/fs/vision-mix/rm4411/torch
export XDG_CACHE_HOME=/n/fs/vision-mix/rm4411/cache
export CACHE_DIR=/n/fs/vision-mix/rm4411/cache

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
cd "${REPO_ROOT}" || exit 1

MODEL="${MODEL:-qwen25_7b_coder}"
OUT_DIR="data/proxy_scored/${MODEL}"
mkdir -p "${OUT_DIR}"

# Optional: PROXY_PRIMARY_ONLY=1 halves infilling/WBC work (sensitivity HPs skipped).
EXTRA=()
if [[ "${PROXY_PRIMARY_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--primary-only)
fi

# shellcheck source=jobscript/_conda.sh
source "${REPO_ROOT}/jobscript/_conda.sh"
conda activate "${CONDA_ENV:-base}"

python -m mia_eval.score_proxy_jsonl \
  --input data/qwen_memtrace_proxy_train.jsonl \
  --output "${OUT_DIR}/qwen_memtrace_proxy_train_scored.jsonl" \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/qwen2p5.yaml \
  --preset "${MODEL}" \
  --batch-size "${PROXY_BATCH_SIZE:-4}" \
  "${EXTRA[@]}" \
  --calibration-json "${OUT_DIR}/proxy_calibration.json"
