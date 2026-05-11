# MIMIR benchmark: conda env, torch, HF stack sanity check.
# Source after PROJECT_ROOT, _slurm_prologue.sh, and TMPDIR fixes are applied.

export MIMIR_ENV_PREFIX="${MIMIR_ENV_PREFIX:-${PROJECT_ROOT}/.conda/envs/mimir_bench}"
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "${CONDA_ENV}"
elif [[ -x "${MIMIR_ENV_PREFIX}/bin/python" ]]; then
  conda activate "${MIMIR_ENV_PREFIX}"
else
  echo "ERROR: No conda env with PyTorch stack for this benchmark." >&2
  echo "Create it once on a login node (with network):" >&2
  echo "  cd ${PROJECT_ROOT} && bash jobscript/setup_mimir_conda_env.sh" >&2
  echo "Or submit with an existing env:" >&2
  echo "  sbatch --export=ALL,HF_TOKEN=\"\$HF_TOKEN\",CONDA_ENV=carlini jobscript/run_mimir_decoding_benchmark.slurm" >&2
  exit 1
fi

export AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
# Avoid mixing ~/.local site-packages with conda env packages (causes hub/datasets skew).
export PYTHONNOUSERSITE=1
unset PYTHONHOME || true

echo "--- conda / python ---"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "which python: $(command -v python)"
python -c "import sys; print('sys.executable:', sys.executable)"

_mimir_ensure_torch() {
  if python -c "import torch" 2>/dev/null; then
    python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
    return 0
  fi
  if [[ "${AUTO_INSTALL_DEPS:-1}" != "1" ]]; then
    echo "ERROR: torch is not installed and AUTO_INSTALL_DEPS=0." >&2
    echo "Run once: bash ${PROJECT_ROOT}/jobscript/setup_mimir_conda_env.sh" >&2
    return 1
  fi
  local idx="https://download.pytorch.org/whl/cu121"
  case "${MIMIR_TORCH:-cu121}" in
    cpu) idx="https://download.pytorch.org/whl/cpu" ;;
    cu124) idx="https://download.pytorch.org/whl/cu124" ;;
    cu121|*) idx="https://download.pytorch.org/whl/cu121" ;;
  esac
  echo "torch missing; installing into active env with: python -m pip install torch ... --index-url ${idx}"
  python -m pip install -U pip setuptools wheel
  # Avoid -U on torch so pip does not opportunistically upgrade huggingface_hub to 1.x.
  python -m pip install torch torchvision torchaudio --index-url "${idx}"
  python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
}

_mimir_ensure_torch || exit 1

# Keep transformers / hub / regex / tokenizers consistent (repairs envs where hub was bumped to 1.x).
_mimir_sync_hf_requirements() {
  if [[ "${AUTO_INSTALL_DEPS:-1}" != "1" ]]; then
    return 0
  fi
  local req="${PROJECT_ROOT}/jobscript/requirements_mimir.txt"
  if [[ ! -f "${req}" ]]; then
    echo "WARN: missing ${req}; skipping HF stack sync." >&2
    return 0
  fi
  echo "Syncing HF/transformers deps: pip install -r ${req}"
  python -m pip install -q -r "${req}"
}

_mimir_sync_hf_requirements || exit 1

python - <<'PY'
import huggingface_hub as h
import transformers
import regex

v = getattr(h, "__version__", "")
if v.startswith("1."):
    raise SystemExit(
        f"huggingface_hub {v} is incompatible with transformers<5; need hub<1. "
        "Run: python -m pip install -r jobscript/requirements_mimir.txt"
    )
print("HF stack OK: huggingface_hub", v, "transformers", transformers.__version__)
PY
