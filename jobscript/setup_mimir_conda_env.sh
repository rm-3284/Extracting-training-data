#!/usr/bin/env bash
# Create a dedicated conda env for memorization_detection / MIMIR benchmark jobs.
#
# Usage (from repo root, on a login or interactive node with network):
#   bash jobscript/setup_mimir_conda_env.sh
#
# Optional:
#   MIMIR_ENV_PREFIX=/path/to/env   (default: <repo>/.conda/envs/mimir_bench)
#   MIMIR_TORCH=cu121|cu124|cpu     (default: cu121 — CUDA 12.1 wheels from pytorch.org)
#
# After creation:
#   conda activate /path/to/mimir_bench
#   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#
# Slurm (run_mimir_decoding_benchmark.slurm) activates the same path if CONDA_ENV is unset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PREFIX="${MIMIR_ENV_PREFIX:-${ROOT}/.conda/envs/mimir_bench}"
TORCH="${MIMIR_TORCH:-cu121}"

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
    "/usr/local/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    [[ -z "${f}" || ! -f "${f}" ]] && continue
    # shellcheck source=/dev/null
    source "${f}"
    return 0
  done
  echo "conda not found. Load your conda module first." >&2
  return 1
}

_mia_eval_conda_init || exit 1

mkdir -p "$(dirname "${PREFIX}")"

if [[ ! -x "${PREFIX}/bin/python" ]]; then
  echo "Creating conda env at ${PREFIX} (python 3.11) ..."
  conda create -y -p "${PREFIX}" python=3.11 pip
else
  echo "Env already exists at ${PREFIX}; upgrading pip packages ..."
fi

PIP="${PREFIX}/bin/pip"
PY="${PREFIX}/bin/python"

"${PIP}" install -U pip setuptools wheel

case "${TORCH}" in
  cpu)
    echo "Installing PyTorch (CPU wheels) ..."
    "${PIP}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ;;
  cu124)
    echo "Installing PyTorch (CUDA 12.4 wheels) ..."
    "${PIP}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ;;
  cu121|*)
    echo "Installing PyTorch (CUDA 12.1 wheels) ..."
    "${PIP}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ;;
esac

# Align with cluster NumPy 1.x stacks; avoids pandas/numexpr ABI issues from NumPy 2.
echo "Installing HF stack ..."
"${PIP}" install \
  "numpy>=1.26,<2" \
  "transformers>=4.36,<5" \
  "datasets>=2.19,<4" \
  "accelerate>=0.26" \
  "huggingface_hub>=0.23,<1" \
  safetensors \
  sentencepiece \
  tqdm \
  filelock \
  pyyaml \
  dill \
  multiprocess

echo ""
echo "Done."
echo "  Activate:  conda activate ${PREFIX}"
echo "  Verify:    conda activate ${PREFIX} && python -c \"import torch, datasets, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\""
echo "  Slurm:     sbatch --export=ALL,HF_TOKEN=\"\$HF_TOKEN\" jobscript/run_mimir_decoding_benchmark.slurm"
echo "           (uses this env automatically if CONDA_ENV is unset and ${PREFIX} exists)"
