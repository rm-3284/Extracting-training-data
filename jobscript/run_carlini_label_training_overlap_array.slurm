#!/bin/bash
#SBATCH --job-name=carlini_label
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%A_%a/carlini_label_out.log
#SBATCH --error=logs/%A_%a/carlini_label_err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#SBATCH --array=0-14
#
# Label ``mia_eval_outputs/carlini_extract/<run_key>/samples.jsonl`` with ``label`` 0/1 by
# streaming the **public training corpora** in ``mia_eval/config/carlini_training_corpora.yaml``
# (Hugging Face Hub) and shingle-matching (see ``mia_eval.label_carlini_extract_samples``).
#
# CPU + RAM + network only (no GPU). Keep ``#SBATCH --array=`` in sync with RUN_KEYS length.
#
#   sbatch jobscript/run_carlini_label_training_overlap_array.slurm
#
# Optional env:
#   PYTHONUNBUFFERED=1
#   PROJECT_ROOT              Override repo root (default: SLURM_SUBMIT_DIR / prologue default)
#   CONDA_ENV                 Conda env with torch, transformers, datasets, pyyaml (same as Carlini extract)
#   CARLINI_EXTRACT_ROOT      (default: mia_eval_outputs/carlini_extract)
#   TRAINING_CORPORA_YAML     (default: mia_eval/config/carlini_training_corpora.yaml)
#   LABEL_SKIP_EXISTING=1     Skip run if samples_labeled.jsonl already exists
#   LABEL_REUSE_INDEX=1       Reuse training_shingle_index.json if present (no re-stream)
#   LABEL_SAVE_INDEX=1        Save training_shingle_index.json after build (default: on)
#   HF_TOKEN                  For gated Hub datasets (e.g. bigcode/the-stack-dedup); or ``huggingface-cli login``
#   LEGACY_GENERIC_CORPORA=1  Use old C4/Pile-by-project heuristic instead of YAML corpora
#   HF_HUB_DOWNLOAD_TIMEOUT   Seconds per file download (increase if Hub read timeouts)
#
# If Slurm warns ``Unable to create TMPDIR [/scratch/...]``, Slurm resets TMPDIR to /tmp; this script
# also sets TMPDIR to /tmp/<user>_carlini_label_tmp when unset or missing.
# Single-run debug (override array index):
#   sbatch --array=3 jobscript/run_carlini_label_training_overlap_array.slurm
#
# Edit #SBATCH --chdir= and --mail-user= to your paths.

set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -z "${TMPDIR:-}" ]] || [[ ! -d "${TMPDIR}" ]]; then
  export TMPDIR="/tmp/${USER}_carlini_label_tmp"
fi
mkdir -p "${TMPDIR}"

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck source=jobscript/_slurm_prologue.sh
source "${PROJECT_ROOT}/jobscript/_slurm_prologue.sh"
conda activate "${CONDA_ENV:-/n/fs/vision-mix/rm4411/conda_envs/carlini}"

mkdir -p "logs/${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"

export HF_HOME="${HF_HOME:-/n/fs/vision-mix/rm4411/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/n/fs/vision-mix/rm4411/cache}"

# Gated datasets (StarCoder Stack, etc.)
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi

RUN_KEYS=(
  olmo2_7b_base
  olmo2_7b_instruct
  olmo1_7b_base
  olmo1_7b_instruct
  llm360_amber_base
  llm360_amber_chat
  llm360_amber_safe
  pythia_12b_base
  dolly_v2_12b
  starcoder2_15b_base
  starcoder2_15b_instruct
  redpajama_7b_base
  redpajama_7b_chat
  dclm_7b_base
  dclm_7b_instruct
)

IDX="${SLURM_ARRAY_TASK_ID:?}"
if [[ "${IDX}" -lt 0 || "${IDX}" -ge "${#RUN_KEYS[@]}" ]]; then
  echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $((${#RUN_KEYS[@]} - 1))]" >&2
  exit 1
fi

RUN_KEY="${RUN_KEYS[$IDX]}"
EXTRACT_ROOT="${CARLINI_EXTRACT_ROOT:-mia_eval_outputs/carlini_extract}"
CORPORA_YAML="${TRAINING_CORPORA_YAML:-mia_eval/config/carlini_training_corpora.yaml}"
RUN_DIR="${PROJECT_ROOT}/${EXTRACT_ROOT}/${RUN_KEY}"

cd "${PROJECT_ROOT}"

EXTRA=()
if [[ "${LABEL_SKIP_EXISTING:-0}" == "1" ]]; then
  EXTRA+=(--skip-existing)
fi
if [[ "${LABEL_REUSE_INDEX:-0}" == "1" ]]; then
  EXTRA+=(--reuse-index)
fi
if [[ "${LABEL_SAVE_INDEX:-1}" == "1" ]]; then
  EXTRA+=(--save-index)
fi
if [[ "${LEGACY_GENERIC_CORPORA:-0}" == "1" ]]; then
  EXTRA+=(--legacy-generic-corpora)
fi

echo "Task ${IDX}: RUN_KEY=${RUN_KEY} RUN_DIR=${RUN_DIR}"

if [[ ! -f "${RUN_DIR}/samples.jsonl" ]]; then
  echo "Missing ${RUN_DIR}/samples.jsonl — run Carlini extraction first." >&2
  exit 1
fi

python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("yaml") is None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", "--prefer-binary", "pyyaml"]
    )
PY

python -m mia_eval.label_carlini_extract_samples \
  --run-dir "${RUN_DIR}" \
  --training-corpora-yaml "${CORPORA_YAML}" \
  "${EXTRA[@]}"

echo "Done: ${RUN_DIR}/samples_labeled.jsonl"
