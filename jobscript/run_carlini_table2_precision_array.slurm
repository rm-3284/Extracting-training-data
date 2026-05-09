#!/bin/bash
#SBATCH --job-name=carlini_patk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%A_%a/carlini_table2_out.log
#SBATCH --error=logs/%A_%a/carlini_table2_err.log
#SBATCH --partition=all
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#SBATCH --array=0-14
#
# Carlini Table-2 scoring + **precision@k** (needs integer ``label`` on each JSONL line — use
# ``samples_labeled.jsonl`` from ``label_carlini_extract_samples`` / ``run_carlini_label_training_overlap_array.slurm``).
#
# Loads target + reference LMs per ``mia_eval/config/carlini_open_table2_models.yaml``. One GPU task per model.
# Keep ``#SBATCH --array=`` in sync with RUN_KEYS length.
#
#   sbatch jobscript/run_carlini_table2_precision_array.slurm
#
# Optional env:
#   PYTHONUNBUFFERED=1
#   PROJECT_ROOT / CONDA_ENV          (see ``jobscript/_slurm_prologue.sh``)
#   CARLINI_EXTRACT_ROOT              (default: mia_eval_outputs/carlini_extract)
#   TABLE2_CONFIG                     (default: mia_eval/config/defaults.yaml)
#   TABLE2_EXPERIMENT                 (default: mia_eval/config/carlini_open_table2_models.yaml)
#   PRECISION_K                       (default: 10,50,100) — forwarded to ``--precision-k``
#   TABLE2_INPUT_PREFER               auto | labeled | raw (default: auto — same as batch driver)
#   TABLE2_SKIP_EXISTING=1            Skip if carlini_table2.json already exists
#   INSTALL_OLMO_DEPS                 (default: 1) — pip ai2-olmo etc. for olmo* keys
#
# For a **single process** that walks all run dirs (sequential on one GPU), use:
#   python -m mia_eval.run_carlini_table2_carlini_extract
#
# Edit #SBATCH --chdir= and --mail-user= to your paths.

set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -z "${TMPDIR:-}" ]] || [[ ! -d "${TMPDIR}" ]]; then
  export TMPDIR="/tmp/${USER}_carlini_table2_tmp"
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
EXTRACT_ROOT="${CARLINI_EXTRACT_ROOT:-mia_eval_outputs/carlini_extract}"
RUN_DIR="${PROJECT_ROOT}/${EXTRACT_ROOT}/${RUN_KEY}"
OUT_JSON="${RUN_DIR}/carlini_table2.json"

CFG="${TABLE2_CONFIG:-mia_eval/config/defaults.yaml}"
EXP="${TABLE2_EXPERIMENT:-mia_eval/config/carlini_open_table2_models.yaml}"
KS="${PRECISION_K:-10,50,100}"
PREFER="${TABLE2_INPUT_PREFER:-auto}"

pick_input_jsonl() {
  local d="$1" pref="$2"
  local labeled="${d}/samples_labeled.jsonl"
  local raw="${d}/samples.jsonl"
  case "${pref}" in
    labeled)
      echo "${labeled}"
      ;;
    raw)
      echo "${raw}"
      ;;
    auto)
      if [[ -f "${labeled}" ]]; then
        echo "${labeled}"
      else
        echo "${raw}"
      fi
      ;;
    *)
      echo "TABLE2_INPUT_PREFER must be auto|labeled|raw, got: ${pref}" >&2
      exit 1
      ;;
  esac
}

INPUT_JSONL="$(pick_input_jsonl "${RUN_DIR}" "${PREFER}")"

cd "${PROJECT_ROOT}"

echo "Task ${IDX}: RUN_KEY=${RUN_KEY}"
echo "  input=${INPUT_JSONL}"
echo "  output=${OUT_JSON}"

if [[ "${TABLE2_SKIP_EXISTING:-0}" == "1" && -f "${OUT_JSON}" ]]; then
  echo "[skip] exists ${OUT_JSON}"
  exit 0
fi

if [[ "${PREFER}" == labeled && ! -f "${INPUT_JSONL}" ]]; then
  echo "Missing ${INPUT_JSONL} — run labeling first or use TABLE2_INPUT_PREFER=auto." >&2
  exit 1
fi

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "Missing input JSONL: ${INPUT_JSONL}" >&2
  exit 1
fi

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

python -m mia_eval.run_carlini_table2 \
  --config "${CFG}" \
  --experiment "${EXP}" \
  --set "active_model=${RUN_KEY}" \
  --input "${INPUT_JSONL}" \
  --output "${OUT_JSON}" \
  --precision-k "${KS}"

echo "Done: ${OUT_JSON}"
