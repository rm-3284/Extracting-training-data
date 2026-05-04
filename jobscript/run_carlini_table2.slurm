#!/bin/bash
#SBATCH --job-name=miaeval_qwen
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
# Carlini Table 2 + scores on ``samples_mia_gt.jsonl`` for one preset.
#   sbatch --export=ALL,MODEL=qwen25_7b_base jobscript/run_carlini_table2.slurm

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

python -m mia_eval.run_carlini_table2 \
  --config mia_eval/config/defaults.yaml \
  --experiment mia_eval/config/qwen2p5.yaml \
  --set "active_model=${MODEL}" \
  --input "mia_eval_outputs/${MODEL}/samples_mia_gt.jsonl" \
  --output "mia_eval_outputs/${MODEL}/carlini_table2.json" \
  --scores-jsonl "mia_eval_outputs/${MODEL}/carlini_table2_scores.jsonl" \
  --window 50 \
  --precision-k 10,50,100
