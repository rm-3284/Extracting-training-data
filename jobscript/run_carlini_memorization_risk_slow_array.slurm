#!/bin/bash
#SBATCH --job-name=memrisk_slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%A_%a/carlini_memrisk_slow_out.log
#SBATCH --error=logs/%A_%a/carlini_memrisk_slow_err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#SBATCH --array=0-14
#
# memorization_risk_slow only (500 samples/model by default).
# Outputs **only** under ``carlini_extract_memorization_risk_slow/`` (never the fast tree).
#   <repo>/mia_eval_outputs/carlini_extract_memorization_risk_slow/<run_key>/samples.jsonl
#   <repo>/mia_eval_outputs/carlini_extract_memorization_risk_slow/<run_key>/run_meta.json
#
#   sbatch jobscript/run_carlini_memorization_risk_slow_array.slurm
#
# Fast decoder (separate time limit): ``run_carlini_memorization_risk_fast_array.slurm``.
#
# Optional env: PYTHONUNBUFFERED, CARLINI_VERBOSE, CARLINI_NO_TQDM, NUM_SAMPLES_PER_STRATEGY,
# BATCH_SIZE, CARLINI_CONFIG, CONDA_ENV, INSTALL_OLMO_DEPS, INSTALL_HUGGINGFACE_HUB.
# Strategy-specific overrides (do not use ``CARLINI_MERGE_CONFIG`` / ``MANIFEST_PATH`` for both jobs):
#   CARLINI_MERGE_CONFIG_MEMRISK_SLOW, MANIFEST_PATH_MEMRISK_SLOW
#
# Edit #SBATCH --chdir= and --time= as needed (slow is usually the bottleneck). Keep --array in
# sync with RUN_KEYS length.

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
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/n/fs/vision-mix/rm4411/cache}"

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
CONFIG="${CARLINI_CONFIG:-mia_eval/config/carlini_open_models.yaml}"
MERGE="${CARLINI_MERGE_CONFIG_MEMRISK_SLOW:-${PROJECT_ROOT}/mia_eval/config/carlini_overlay_memorization_risk_slow_only.yaml}"
MANIFEST_PATH="${MANIFEST_PATH_MEMRISK_SLOW:-${PROJECT_ROOT}/mia_eval_outputs/carlini_extract_memorization_risk_slow/manifest.json}"
if [[ "${MERGE}" == *fast_only* ]]; then
  echo "error: slow array merge must not use fast_only overlay (got ${MERGE})" >&2
  exit 1
fi

EXTRA=()
if [[ -n "${NUM_SAMPLES_PER_STRATEGY:-}" ]]; then
  EXTRA+=(--num-samples-per-strategy "${NUM_SAMPLES_PER_STRATEGY}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  EXTRA+=(--batch-size "${BATCH_SIZE}")
fi

echo "Task ${IDX}: RUN_KEY=${RUN_KEY}  memorization_risk_slow  merge=${MERGE}"
echo "Outputs: ${PROJECT_ROOT}/mia_eval_outputs/carlini_extract_memorization_risk_slow/${RUN_KEY}/samples.jsonl"
cd "${PROJECT_ROOT}"

if [[ "${RUN_KEY}" == olmo* ]] && [[ "${INSTALL_OLMO_DEPS:-1}" == "1" ]]; then
  python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("hf_olmo") is None:
    print("Installing OLMo deps into active env: ai2-olmo transformers accelerate sentencepiece")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", "ai2-olmo", "transformers", "accelerate", "sentencepiece"]
    )
else:
    print("hf_olmo already available in active env")
PY
fi

if [[ "${RUN_KEY}" == dclm* ]] && [[ "${INSTALL_HUGGINGFACE_HUB:-1}" == "1" ]]; then
  python - <<'PY'
import importlib.util
import subprocess
import sys

try:
    import huggingface_hub.hf_file_system  # noqa: F401
except ImportError:
    print("Installing huggingface_hub>=0.23.5 (streaming datasets require hf_file_system)")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "--prefer-binary",
            "huggingface_hub>=0.23.5",
        ]
    )
else:
    print("huggingface_hub.hf_file_system OK")
PY
fi

python -m mia_eval.run_carlini_extraction_batch \
  --config "${CONFIG}" \
  --merge-config "${MERGE}" \
  --manifest-path "${MANIFEST_PATH}" \
  --manifest-mode merge \
  --only "${RUN_KEY}" \
  "${EXTRA[@]}"
