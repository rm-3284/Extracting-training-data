#!/bin/bash
#SBATCH --job-name=proxy_calib_mrg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%A_%a/out_merge.log
#SBATCH --error=logs/%A_%a/err_merge.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#SBATCH --array=0-4
#
# Merge shard JSONLs from score_proxy_calib_shards_40_array.slurm and write thresholds JSON
# (infilling + WBC; memtrace fixed at MEMTRACE_P_FIXED).
#
# Must match NUM_SHARDS_PER_MODEL used in the shard job (default 8). CAL_N rows = lines in CALIB_INPUT.
#
#   sbatch --dependency=afterok:<JOBID_SHARDS> jobscript/score_proxy_calib_merge_thresholds_array.slurm
#
# Env: NUM_SHARDS_PER_MODEL (default 8), CAL_N (default 512), CALIB_INPUT stem name for glob,
#      MEMTRACE_P_FIXED, CONDA_ENV

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/n/fs/vision-mix/rm4411/Extracting-training-data}"
source "${PROJECT_ROOT}/jobscript/_slurm_prologue.sh"
conda activate "${CONDA_ENV:-base}"

mkdir -p "logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

NSM_PAD="$(printf '%04d' "${NUM_SHARDS_PER_MODEL:-8}")"
CAL_N="${CAL_N:-512}"
MT_FIX="${MEMTRACE_P_FIXED:-0.5}"

MODELS=(
  qwen25_7b_base
  qwen25_7b_instruct
  qwen25_7b_coder
  qwen25_7b_math
  distil_qwen25_7b_instruct
)

MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
OUT_DIR="data/proxy_scored/${MODEL}"
SHARD_GLOB="${OUT_DIR}/calib_shards/calib512_scored_shard*_of_${NSM_PAD}.jsonl"
MERGED="${OUT_DIR}/calib_${CAL_N}_merged.jsonl"
THR_JSON="${OUT_DIR}/proxy_thresholds_inf_wbc_mtfix.json"

python -m mia_eval.merge_proxy_scored_shards \
  --glob "${SHARD_GLOB}" \
  --output "${MERGED}" \
  --expected-rows "${CAL_N}" \
  --strip-proxy-row-index

python -m mia_eval.compute_proxy_thresholds \
  --input "${MERGED}" \
  --output "${THR_JSON}" \
  --memtrace-p-fixed "${MT_FIX}"

echo "Wrote ${THR_JSON}"
