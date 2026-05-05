#!/bin/bash
#SBATCH --job-name=mimir_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --output=logs/mimir_bench_%j.out
#SBATCH --error=logs/mimir_bench_%j.err
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#
# MIMIR decoding benchmark: baseline + fast + slow on three open LMs (see
# memorization_detection/run_mimir_decoding_benchmark.py).
#
# Prerequisites:
#   - HF_TOKEN with MIMIR access (do not put the token in this file).
#   - Edit --chdir and PROJECT_ROOT below if your clone path differs.
#   - Edit --partition / --time / --mem for your cluster (e.g. mig, pli-lc).
#
# Submit from repo root (recommended):
#   mkdir -p logs
#   export HF_TOKEN="hf_..."
#   sbatch jobscript/run_mimir_decoding_benchmark.slurm
#
# Or pass the token at submit time:
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" jobscript/run_mimir_decoding_benchmark.slurm
#
# Optional environment overrides (all optional):
#   CONDA_ENV         conda env name (default: base)
#   N_EXAMPLES        --n-examples (default: 50)
#   BENCH_OUTPUT      --output JSON path (default: logs/${SLURM_JOB_ID}/mimir_bench.json)
#   SKIP_SLOW         set to 1 to pass --skip-slow
#   BENCH_ARGS        optional extra CLI tokens (unquoted expansion), e.g.:
#                       export BENCH_ARGS="--verbose --seed 1 --models gpt_neo_2p7"
#   HF_HOME           Hugging Face cache (defaults under PROJECT_ROOT below)

set -euo pipefail

mkdir -p logs

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/n/fs/vision-mix/rm4411/Extracting-training-data}"
# shellcheck source=jobscript/_slurm_prologue.sh
source "${PROJECT_ROOT}/jobscript/_slurm_prologue.sh"
conda activate "${CONDA_ENV:-base}"

JOB_TAG="${SLURM_JOB_ID:-manual}"
mkdir -p "logs/${JOB_TAG}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it before sbatch, or use:" >&2
  echo "  sbatch --export=ALL,HF_TOKEN=\"\$HF_TOKEN\" jobscript/run_mimir_decoding_benchmark.slurm" >&2
  exit 1
fi

# Cache on shared FS (override with HF_HOME / TRANSFORMERS_CACHE if you prefer scratch)
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.cache}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

N_EXAMPLES="${N_EXAMPLES:-50}"
OUT_JSON="${BENCH_OUTPUT:-logs/${JOB_TAG}/mimir_bench.json}"

EXTRA=(--n-examples "${N_EXAMPLES}" --output "${OUT_JSON}")
if [[ "${SKIP_SLOW:-0}" == "1" ]]; then
  EXTRA+=(--skip-slow)
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "Job ID: ${JOB_TAG}"
echo "Writing: ${OUT_JSON}"
echo "Running: python memorization_detection/run_mimir_decoding_benchmark.py ${EXTRA[*]} ${BENCH_ARGS:-}"

# shellcheck disable=SC2086
python memorization_detection/run_mimir_decoding_benchmark.py "${EXTRA[@]}" ${BENCH_ARGS:-}

echo "Finished job ${JOB_TAG}"
