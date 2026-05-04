#!/bin/bash
#SBATCH --job-name=proxy_mia_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%j/merge_out.log
#SBATCH --error=logs/%j/merge_err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#
# After all shard jobs finish for one MODEL and NUM_SHARDS:
#   export MODEL=qwen25_7b_base
#   export NUM_SHARDS=32
#   sbatch jobscript/merge_proxy_train_shards.slurm
#
# Optional: export EXPECTED_ROWS=12201 for a strict row count check.

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/n/fs/vision-mix/rm4411/Extracting-training-data}"
# shellcheck source=jobscript/_slurm_prologue.sh
source "${PROJECT_ROOT}/jobscript/_slurm_prologue.sh"
conda activate "${CONDA_ENV:-base}"

mkdir -p "logs/${SLURM_JOB_ID}"

MODEL="${MODEL:?export MODEL=...}"
NUM_SHARDS="${NUM_SHARDS:?export NUM_SHARDS=...}"
EXPECTED="${EXPECTED_ROWS:-12201}"

SHARD_GLOB="data/proxy_scored/${MODEL}/shards/qwen_memtrace_proxy_train_scored_shard*_of_$(printf '%04d' "${NUM_SHARDS}").jsonl"
OUT_DIR="data/proxy_scored/${MODEL}"
mkdir -p "${OUT_DIR}"

MERGE_ARGS=(
  --glob "${SHARD_GLOB}"
  --output "${OUT_DIR}/qwen_memtrace_proxy_train_scored.jsonl"
  --strip-proxy-row-index
)

if [[ "${EXPECTED}" != "0" ]]; then
  MERGE_ARGS+=(--expected-rows "${EXPECTED}")
fi

python -m mia_eval.merge_proxy_scored_shards "${MERGE_ARGS[@]}"
