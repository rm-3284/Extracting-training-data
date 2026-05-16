#!/bin/bash
#SBATCH --job-name=carlini_t2_gptneo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%j/carlini_table2_gpt_neo_out.log
#SBATCH --error=logs/%j/carlini_table2_gpt_neo_err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#
# Carlini Table 2-style scores + **precision@k per JSONL ``source``** (each generation method).
# Requires labeled JSONL from ``label_carlini_extract_samples`` (``label`` on every row).
#
# Default I/O:
#   INPUT:  ${RUN_DIR}/samples_labeled.jsonl
#   OUTPUT: ${RUN_DIR}/carlini_table2.json
#
# Inside ``carlini_table2.json``, see:
#   metrics → Perplexity | Small | zlib | Lowercase | Window
#     → precision_at_k_by_source → { "top_k": {"@10": …}, "memorization_cheap_logits": …, … }
#
# Submit:
#   sbatch jobscript/run_carlini_table2_gpt_neo_decode_defenses.slurm
#
# Optional env:
#   CARLINI_TABLE2_RUN_DIR       Run directory (default: decode_defenses/gpt_neo_2p7).
#   CARLINI_TABLE2_INPUT       Override labeled JSONL path (default: ${RUN_DIR}/samples_labeled.jsonl).
#   CARLINI_TABLE2_OUTPUT        Override output JSON path (default: ${RUN_DIR}/carlini_table2.json).
#   CARLINI_TABLE2_SCORES_JSONL  If set, also write per-row scores (for ``precision_at_k_by_method``).
#   CARLINI_TABLE2_CONFIG        Default: mia_eval/config/defaults.yaml
#   CARLINI_TABLE2_PRECISION_K   Default: 10,50,100
#   CARLINI_TABLE2_ACTIVE_MODEL  Default: gpt_neo_2p7 (must match ``defaults.yaml`` / experiment bundle).
#   CONDA_ENV  HF_HOME  TORCH_HOME

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

RUN_DIR="${CARLINI_TABLE2_RUN_DIR:-${PROJECT_ROOT}/mia_eval_outputs/carlini_extract_decode_defenses/gpt_neo_2p7}"
IN_JSONL="${CARLINI_TABLE2_INPUT:-${RUN_DIR}/samples_labeled.jsonl}"
OUT_JSON="${CARLINI_TABLE2_OUTPUT:-${RUN_DIR}/carlini_table2.json}"
PK="${CARLINI_TABLE2_PRECISION_K:-10,50,100}"
CFG="${CARLINI_TABLE2_CONFIG:-mia_eval/config/defaults.yaml}"
ACTIVE="${CARLINI_TABLE2_ACTIVE_MODEL:-gpt_neo_2p7}"

EXTRA=()
if [[ -n "${CARLINI_TABLE2_SCORES_JSONL:-}" ]]; then
  EXTRA+=(--scores-jsonl "${CARLINI_TABLE2_SCORES_JSONL}")
fi

echo "INPUT=${IN_JSONL}"
echo "OUTPUT=${OUT_JSON}"
echo "active_model=${ACTIVE}  precision-k=${PK}"
cd "${PROJECT_ROOT}"

if [[ ! -f "${IN_JSONL}" ]]; then
  echo "error: labeled input not found: ${IN_JSONL}" >&2
  echo "Run labeling first (e.g. jobscript/run_carlini_label_gpt_neo_decode_defenses.slurm)." >&2
  exit 1
fi

python -m mia_eval.run_carlini_table2 \
  --config "${CFG}" \
  --set "active_model=${ACTIVE}" \
  --input "${IN_JSONL}" \
  --output "${OUT_JSON}" \
  --precision-k "${PK}" \
  "${EXTRA[@]}"
