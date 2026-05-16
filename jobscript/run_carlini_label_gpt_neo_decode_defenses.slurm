#!/bin/bash
#SBATCH --job-name=carlini_lbl_gptneo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%j/carlini_label_gpt_neo_out.log
#SBATCH --error=logs/%j/carlini_label_gpt_neo_err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#
# Shingle-label ``samples.jsonl`` → ``samples_labeled.jsonl`` for the GPT-Neo decode-defenses
# run (heavy index: raise LABEL_MAX_DOCUMENTS / LABEL_MAX_SHINGLES via env as your cluster allows).
#
# Default run dir:
#   ${PROJECT_ROOT}/mia_eval_outputs/carlini_extract_decode_defenses/gpt_neo_2p7
#
# Outputs in that directory:
#   samples_labeled.jsonl
#   training_corpus_manifest.json
#   training_shingle_index.json   (if LABEL_SAVE_INDEX=1)
#
# Submit:
#   sbatch jobscript/run_carlini_label_gpt_neo_decode_defenses.slurm
#
# Optional env:
#   CARLINI_LABEL_RUN_DIR   Override path to run directory (must contain samples.jsonl + run_meta.json).
#   LABEL_MAX_DOCUMENTS     Per-source cap (overrides every ``max_documents`` in YAML for this run); default 80000.
#   LABEL_MAX_SHINGLES      Global shingle cap (defaults YAML ``max_shingles`` if unset); default 8000000.
#   LABEL_SAVE_INDEX=1      Persist ``training_shingle_index.json`` for ``--reuse-index`` reruns.
#   LABEL_SKIP_EXISTING=1   Skip if samples_labeled.jsonl already exists.
#   CONDA_ENV  HF_HOME  …

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

RUN_DIR="${CARLINI_LABEL_RUN_DIR:-${PROJECT_ROOT}/mia_eval_outputs/carlini_extract_decode_defenses/gpt_neo_2p7}"
CORPORA_YAML="${LABEL_TRAINING_CORPORA_YAML:-${PROJECT_ROOT}/mia_eval/config/carlini_training_corpora.yaml}"

MAX_DOC="${LABEL_MAX_DOCUMENTS:-80000}"
MAX_SH="${LABEL_MAX_SHINGLES:-8000000}"

EXTRA=()
EXTRA+=(--max-documents "${MAX_DOC}")
EXTRA+=(--max-shingles "${MAX_SH}")

if [[ "${LABEL_SAVE_INDEX:-0}" == "1" ]]; then
  EXTRA+=(--save-index)
fi
if [[ "${LABEL_SKIP_EXISTING:-0}" == "1" ]]; then
  EXTRA+=(--skip-existing)
fi

echo "RUN_DIR=${RUN_DIR}"
echo "Using --max-documents ${MAX_DOC} --max-shingles ${MAX_SH} (override all YAML source caps for this index build)"
cd "${PROJECT_ROOT}"

python -m mia_eval.label_carlini_extract_samples \
  --run-dir "${RUN_DIR}" \
  --training-corpora-yaml "${CORPORA_YAML}" \
  "${EXTRA[@]}"
