#!/bin/bash
#SBATCH --job-name=carlini_def_gptneo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%j/carlini_decode_def_gpt_neo_out.log
#SBATCH --error=logs/%j/carlini_decode_def_gpt_neo_err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#
# Carlini sampling (top_k, temperature_decay, internet where enabled) **plus** decode-defense
# strategies from ``mia_eval/config/carlini_overlay_decode_defenses.yaml`` for **GPT-Neo 2.7B**
# first (run_key ``gpt_neo_2p7``). Requires ``gpt_neo_2p7`` in ``carlini_open_models.yaml``.
#
# Outputs (default merge):
#   ${PROJECT_ROOT}/mia_eval_outputs/carlini_extract_decode_defenses/gpt_neo_2p7/samples.jsonl
#   …/run_meta.json
# Manifest:
#   ${PROJECT_ROOT}/mia_eval_outputs/carlini_extract_decode_defenses/manifest.json
#
# Submit:
#   sbatch jobscript/run_carlini_decode_defenses_gpt_neo.slurm
#
# Optional env:
#   NUM_SAMPLES_PER_STRATEGY  BATCH_SIZE  CARLINI_CONFIG  CARLINI_MERGE_DECODE_DEFENSES
#   MANIFEST_PATH_DECODE_DEFENSES  CONDA_ENV  HF_HOME  CARLINI_VERBOSE

set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -z "${TMPDIR:-}" ]] || [[ ! -d "${TMPDIR}" ]]; then
  export TMPDIR="/tmp/${USER}_carlini_tmp"
fi
mkdir -p "${TMPDIR}"

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck source=jobscript/_slurm_prologue.sh
source "${PROJECT_ROOT}/jobscript/_slurm_prologue.sh"
conda activate "${CONDA_ENV:-/n/fs/vision-mix/rm4411/conda_envs/carlini}"

mkdir -p "logs/${SLURM_JOB_ID:-manual}"

export HF_HOME="${HF_HOME:-/n/fs/vision-mix/rm4411/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TORCH_HOME="${TORCH_HOME:-/n/fs/vision-mix/rm4411/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/n/fs/vision-mix/rm4411/cache}"

RUN_KEY="${RUN_KEY_DECODE_DEF:-gpt_neo_2p7}"
CONFIG="${CARLINI_CONFIG:-mia_eval/config/carlini_open_models.yaml}"
MERGE="${CARLINI_MERGE_DECODE_DEFENSES:-${PROJECT_ROOT}/mia_eval/config/carlini_overlay_decode_defenses.yaml}"
MANIFEST_PATH="${MANIFEST_PATH_DECODE_DEFENSES:-${PROJECT_ROOT}/mia_eval_outputs/carlini_extract_decode_defenses/manifest.json}"

EXTRA=()
if [[ -n "${NUM_SAMPLES_PER_STRATEGY:-}" ]]; then
  EXTRA+=(--num-samples-per-strategy "${NUM_SAMPLES_PER_STRATEGY}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  EXTRA+=(--batch-size "${BATCH_SIZE}")
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  EXTRA+=(--skip-existing)
fi

echo "RUN_KEY=${RUN_KEY}  merge=${MERGE}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Outputs: ${PROJECT_ROOT}/mia_eval_outputs/carlini_extract_decode_defenses/${RUN_KEY}/samples.jsonl"
cd "${PROJECT_ROOT}"

python -m mia_eval.run_carlini_extraction_batch \
  --config "${CONFIG}" \
  --merge-config "${MERGE}" \
  --manifest-path "${MANIFEST_PATH}" \
  --manifest-mode merge \
  --only "${RUN_KEY}" \
  "${EXTRA[@]}"
