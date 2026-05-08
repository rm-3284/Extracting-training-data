#!/bin/bash
#SBATCH --job-name=carlini_open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%A_%a/carlini_open_out.log
#SBATCH --error=logs/%A_%a/carlini_open_err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#SBATCH --array=0-14
#
# One Carlini extraction run per array task (see ``carlini_open_models.yaml``: top_k + temperature_decay +
# top_k_internet per Carlini §5.1.2 → 1500 JSONL lines/run at 500/strategy).
# Keep array range in sync with RUN_KEYS length.
#
#   sbatch jobscript/run_carlini_open_extract_array.slurm
#
# Optional env:
#   PYTHONUNBUFFERED=1        (recommended: live log lines in *.out)
#   CARLINI_VERBOSE=1         (extra load/debug prints in generation + model_utils)
#   CARLINI_NO_TQDM=1         (disable tqdm bars if logs get cluttered)
#   NUM_SAMPLES_PER_STRATEGY  (default: use YAML, currently 500)
#   BATCH_SIZE                (default: use YAML generation.batch_size)
#   CARLINI_CONFIG            (default: mia_eval/config/carlini_open_models.yaml)
#   MANIFEST_PATH             (default: mia_eval_outputs/carlini_extract/manifest.json)
#   CONDA_ENV
#   INSTALL_OLMO_DEPS         (default: 1; auto-install/check hf_olmo for OLMo runs)
#
# DCLM tasks expect ``open_lm`` already in the conda env (``import open_lm.hf`` registers ``openlm`` with
# Transformers). One-time install: see ``mia_eval/openlm_hf_loader.py``.
# Streaming ``internet_prefix`` needs ``huggingface_hub.hf_file_system`` (upgrade hub if ImportError).
# Optional env: INSTALL_HUGGINGFACE_HUB=0 to skip the auto ``pip install`` below for dclm runs.
#
# Edit #SBATCH --chdir= to your clone if different.
#
# If Slurm logs ``Unable to create TMPDIR [/scratch/...]`` once per job, that is the scheduler;
# either create that directory or ``unset TMPDIR`` before ``sbatch`` so Slurm picks /tmp.

set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Runtime temp files (Python, HF cache staging): avoid inherited TMPDIR pointing at a missing scratch path.
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
MANIFEST_PATH="${MANIFEST_PATH:-mia_eval_outputs/carlini_extract/manifest.json}"

EXTRA=()
if [[ -n "${NUM_SAMPLES_PER_STRATEGY:-}" ]]; then
  EXTRA+=(--num-samples-per-strategy "${NUM_SAMPLES_PER_STRATEGY}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  EXTRA+=(--batch-size "${BATCH_SIZE}")
fi

echo "Task ${IDX}: RUN_KEY=${RUN_KEY}"
cd "${PROJECT_ROOT}"

# OLMo checkpoints on HF require hf_olmo runtime in the same active conda env.
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
  --manifest-path "${MANIFEST_PATH}" \
  --manifest-mode merge \
  --only "${RUN_KEY}" \
  "${EXTRA[@]}"
