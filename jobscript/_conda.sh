# shellcheck shell=bash
# Source from Slurm jobscripts (non-interactive bash has no ``conda init`` hooks):
#   source "${REPO_ROOT}/jobscript/_conda.sh"
#   conda activate base   # or your env name
#
# Override install location if needed:
#   export CONDA_ROOT=/path/to/miniconda3   # directory that contains etc/profile.d/conda.sh

_mia_eval_source_conda() {
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
  echo "mia_eval jobscript: could not find conda.sh. Set CONDA_ROOT to the conda prefix." >&2
  return 1
}

_mia_eval_source_conda || return 1
