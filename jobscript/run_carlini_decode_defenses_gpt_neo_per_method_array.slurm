#!/bin/bash
#SBATCH --job-name=def_gen_gptneo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%A_%a/carlini_def_gen_gptneo_out.log
#SBATCH --error=logs/%A_%a/carlini_def_gen_gptneo_err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#SBATCH --array=0-7
#
# One array task = one generation method, **500 samples** each (equal N per method).
# Matches ``carlini_overlay_decode_defenses.yaml`` + base Carlini sampling (top_k, internet, temp decay).
#
# Outputs per task:
#   mia_eval_outputs/carlini_extract_decode_defenses/gpt_neo_2p7/by_method/<strategy>/samples.jsonl
#   …/by_method/<strategy>/run_meta.json
#
# Submit:
#   sbatch jobscript/run_carlini_decode_defenses_gpt_neo_per_method_array.slurm
#
# Optional env:
#   NUM_SAMPLES_PER_STRATEGY  (default 500)
#   BATCH_SIZE  RUN_KEY  CARLINI_CONFIG  CARLINI_MERGE_DECODE_DEFENSES
#   SKIP_EXISTING=1  CONDA_ENV  HF_HOME
#
# Task 6 (memorization_sparse_infilling) is slow; raise --time for that index if needed.
#
# Edit #SBATCH --array=0-N to match STRATEGIES length.

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

mkdir -p "logs/${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"

export HF_HOME="${HF_HOME:-/n/fs/vision-mix/rm4411/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TORCH_HOME="${TORCH_HOME:-/n/fs/vision-mix/rm4411/torch}"

# Order must match #SBATCH --array=0-7
STRATEGIES=(
  top_k
  top_k_internet
  temperature_decay
  memorization_baseline
  memorization_cheap_logits
  memorization_contrastive
  memorization_sparse_infilling
  memorization_wbc_no_infilling
)

IDX="${SLURM_ARRAY_TASK_ID:?}"
if [[ "${IDX}" -lt 0 || "${IDX}" -ge "${#STRATEGIES[@]}" ]]; then
  echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((${#STRATEGIES[@]} - 1))]" >&2
  exit 1
fi

STRATEGY="${STRATEGIES[$IDX]}"
RUN_KEY="${RUN_KEY:-gpt_neo_2p7}"
NUM_SAMPLES="${NUM_SAMPLES_PER_STRATEGY:-500}"
CONFIG="${CARLINI_CONFIG:-mia_eval/config/carlini_open_models.yaml}"
MERGE="${CARLINI_MERGE_DECODE_DEFENSES:-${PROJECT_ROOT}/mia_eval/config/carlini_overlay_decode_defenses.yaml}"

OUT_DIR="${PROJECT_ROOT}/mia_eval_outputs/carlini_extract_decode_defenses/${RUN_KEY}/by_method/${STRATEGY}"
OUT_JSONL="${OUT_DIR}/samples.jsonl"

EXTRA=()
EXTRA+=(--num-samples "${NUM_SAMPLES}")
if [[ -n "${BATCH_SIZE:-}" ]]; then
  EXTRA+=(--batch-size "${BATCH_SIZE}")
fi

echo "Task ${IDX}: RUN_KEY=${RUN_KEY}  strategy=${STRATEGY}  n=${NUM_SAMPLES}"
echo "Output: ${OUT_JSONL}"
cd "${PROJECT_ROOT}"

if [[ "${SKIP_EXISTING:-0}" == "1" && -f "${OUT_JSONL}" ]]; then
  n=$(wc -l < "${OUT_JSONL}" | tr -d ' ')
  echo "[skip] exists ${OUT_JSONL} (${n} lines)"
  exit 0
fi

python -m mia_eval.run_carlini_extraction_per_strategy \
  --config "${CONFIG}" \
  --merge-config "${MERGE}" \
  --run-key "${RUN_KEY}" \
  --strategy "${STRATEGY}" \
  "${EXTRA[@]}"
