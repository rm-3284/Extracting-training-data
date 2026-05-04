# shellcheck shell=bash
# Cluster Slurm prologue: repo root, PYTHONPATH, TMPDIR, conda for non-login shells.
#
# Each ``*.slurm`` should ``cd`` / set PROJECT_ROOT, then source this file. Prefer:
#   export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/n/fs/vision-mix/rm4411/Extracting-training-data}"
#   source "${PROJECT_ROOT}/jobscript/_slurm_prologue.sh"
#   conda activate "${CONDA_ENV:-base}"
#
# (Do not use BASH_SOURCE to find the repo: Slurm may run a copy under /var/spool/slurmd/.)
#
# Override clone path: export PROJECT_ROOT=/path/to/Extracting-training-data
# Override conda install: export CONDA_ROOT=/path/to/miniconda3 (only if conda not on PATH)

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="/n/fs/vision-mix/rm4411/Extracting-training-data"
fi
if [[ ! -d "${PROJECT_ROOT}/mia_eval" && -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/mia_eval" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
fi
export PROJECT_ROOT
cd "${PROJECT_ROOT}" || {
  echo "Cannot cd to PROJECT_ROOT=${PROJECT_ROOT}" >&2
  return 1
}
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TMP_BASE="${TMP_BASE:-/n/fs/vision-mix/rm4411/tmp}"
mkdir -p "${TMP_BASE}" 2>/dev/null || true
export TMPDIR="${TMPDIR:-${TMP_BASE}}"
mkdir -p "${TMPDIR}" 2>/dev/null || export TMPDIR=/tmp

_mia_eval_conda_init() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    return 0
  fi
  local f
  for f in \
    "${CONDA_ROOT:+${CONDA_ROOT}/etc/profile.d/conda.sh}" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "${HOME}/micromamba/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    [[ -z "${f}" || ! -f "${f}" ]] && continue
    # shellcheck source=/dev/null
    source "${f}"
    return 0
  done
  echo "mia_eval jobscript: conda not on PATH and conda.sh not found. Load your conda module first or set CONDA_ROOT." >&2
  return 1
}

_mia_eval_conda_init || return 1
