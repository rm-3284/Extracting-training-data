#!/bin/bash
#SBATCH --job-name=mimir_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --output=logs/mimir_bench_%j.out
#SBATCH --error=logs/mimir_bench_%j.err
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#
# MIMIR decoding benchmark: baseline + fast + slow on three open LMs (see
# memorization_detection/run_mimir_decoding_benchmark.py).
#
# Prerequisites:
#   - One-time: create the dedicated env (login node, network):
#       bash jobscript/setup_mimir_conda_env.sh
#     Default install path: ${PROJECT_ROOT}/.conda/envs/mimir_bench
#   - HF_TOKEN with MIMIR access (do not put the token in this file).
#   - Edit --chdir and PROJECT_ROOT below if your clone path differs.
#   - Edit --partition / --time / --mem for your cluster (e.g. mig, pli-lc).
#
# Submit from repo root (recommended):
#   mkdir -p logs
#   export HF_TOKEN="hf_..."
#   sbatch jobscript/run_mimir_decoding_benchmark.slurm
#
# Or pass the token at submit time:
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" jobscript/run_mimir_decoding_benchmark.slurm
#
# Optional environment overrides (all optional):
#   CONDA_ENV         explicit conda env name OR path (overrides default mimir_bench path)
#   MIMIR_ENV_PREFIX  path to env created by setup_mimir_conda_env.sh (default: ${PROJECT_ROOT}/.conda/envs/mimir_bench)
#   MIMIR_TORCH       cu121 | cu124 | cpu — wheel index if job must pip-install torch (default: cu121)
#   N_EXAMPLES        --n-examples (default: 50)
#   BENCH_OUTPUT      --output JSON path (default: logs/${SLURM_JOB_ID}/mimir_bench.json)
#   SKIP_SLOW         set to 1 to pass --skip-slow
#   AUTO_INSTALL_DEPS set to 0 to disable auto pip install when missing
#                     (plain `pip show` may be a different Python than `python` after conda activate;
#                      always use: conda activate ENV && python -m pip show datasets)
#   PIP_OVERLAY_DIR   writable dir for per-job overlay installs (default: ${PROJECT_ROOT}/.slurm_pip_overlay)
#   TMP_BASE          temp dir Slurm/prologue use (default: ${PROJECT_ROOT}/tmp — avoids /scratch/... if missing)
#   BENCH_ARGS        optional extra CLI tokens (unquoted expansion), e.g.:
#                       export BENCH_ARGS="--verbose --seed 1 --models gpt_neo_2p7"
#   HF_HOME           Hugging Face cache (defaults under PROJECT_ROOT below)

set -euo pipefail

mkdir -p logs

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/n/fs/vision-mix/rm4411/Extracting-training-data}"
# Writable temp (cluster may set TMPDIR=/scratch/... which does not exist for every user).
export TMP_BASE="${TMP_BASE:-${PROJECT_ROOT}/tmp}"
mkdir -p "${TMP_BASE}"
# Slurm may export TMPDIR=/scratch/... before the script runs; override if missing/unwritable.
if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR}" || ! -w "${TMPDIR}" ]]; then
  export TMPDIR="${TMP_BASE}"
fi
mkdir -p "${TMPDIR}"

# shellcheck source=jobscript/_slurm_prologue.sh
source "${PROJECT_ROOT}/jobscript/_slurm_prologue.sh"

# Prologue uses TMPDIR:-TMP_BASE; if Slurm exported TMPDIR=/scratch/... and that path is missing,
# TMPDIR can still be invalid. Force a writable TMPDIR under the repo after prologue runs.
export TMP_BASE="${TMP_BASE:-${PROJECT_ROOT}/tmp}"
mkdir -p "${TMP_BASE}"
if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR}" || ! -w "${TMPDIR}" ]]; then
  export TMPDIR="${TMP_BASE}"
fi
mkdir -p "${TMPDIR}"

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

JOB_TAG="${SLURM_JOB_ID:-manual}"
mkdir -p "logs/${JOB_TAG}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it before sbatch, or use:" >&2
  echo "  sbatch --export=ALL,HF_TOKEN=\"\$HF_TOKEN\" jobscript/run_mimir_decoding_benchmark.slurm" >&2
  exit 1
fi

# Cache on shared FS (override HF_HOME if you prefer another path)
# Hugging Face cache (HF_HOME covers transformers + hub; avoid TRANSFORMERS_CACHE — deprecated in transformers v5 roadmap)
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.cache}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

N_EXAMPLES="${N_EXAMPLES:-50}"
OUT_JSON="${BENCH_OUTPUT:-logs/${JOB_TAG}/mimir_bench.json}"

EXTRA=(--n-examples "${N_EXAMPLES}" --output "${OUT_JSON}")
if [[ "${SKIP_SLOW:-0}" == "1" ]]; then
  EXTRA+=(--skip-slow)
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "Job ID: ${JOB_TAG}"
echo "Writing: ${OUT_JSON}"
echo "Python: $(command -v python)"
# Fresh per-job pip overlay to avoid stale package conflicts.
export PIP_OVERLAY_DIR="${PIP_OVERLAY_DIR:-${PROJECT_ROOT}/.slurm_pip_overlay}"
export PIP_TARGET="${PIP_OVERLAY_DIR}/${JOB_TAG}"
rm -rf "${PIP_TARGET}"
mkdir -p "${PIP_TARGET}"
# Do NOT prepend PYTHONPATH until deps resolve; overlay numpy 2.x breaks conda numexpr/bottleneck.

echo "--- Python / pip diagnostics (same interpreter the job uses) ---"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "python -m pip: $(python -m pip --version 2>/dev/null || echo 'missing')"
python -m pip show datasets 2>/dev/null | head -5 || echo "(python -m pip show datasets: not installed for THIS python)"
echo "---"

if [[ "${AUTO_INSTALL_DEPS:-1}" == "1" ]]; then
python - <<'PY'
"""Resolve huggingface_hub + datasets without mixing NumPy 2 overlay with conda-built extensions."""
import os
import subprocess
import sys

pip_target = os.environ["PIP_TARGET"]
os.makedirs(pip_target, exist_ok=True)

print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
print("PYTHONNOUSERSITE:", os.environ.get("PYTHONNOUSERSITE"))
print("PIP_TARGET:", pip_target)


def have_hub_commitinfo():
    try:
        from huggingface_hub import CommitInfo  # noqa: F401

        return True
    except Exception:
        return False


def import_datasets_clean():
    import datasets as ds

    return ds


def install_hub_overlay():
    print("Installing huggingface_hub into overlay (--no-deps)")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "--no-deps",
            "--target",
            pip_target,
            "huggingface_hub>=0.23,<1.0",
        ]
    )


def install_datasets_overlay_numpy1():
    """
    Pin NumPy < 2 so pandas does not pull NumPy 2 and then import conda numexpr/bottleneck
    built for NumPy 1.x (AttributeError: _ARRAY_API not found).
    Also install numexpr/bottleneck into the overlay against the same NumPy.
    """
    print(
        "Installing datasets stack into overlay with numpy<2 (avoids conda C-extension mismatch)"
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "--target",
            pip_target,
            "numpy>=1.26,<2",
            "pandas>=2.0.2,<2.3",
            "pyarrow>=14,<21",
            "numexpr>=2.8.4",
            "bottleneck>=1.3.7",
            "datasets>=2.19,<4",
        ]
    )


def try_imports_with_overlay():
    sys.path.insert(0, pip_target)
    if not have_hub_commitinfo():
        return False
    import huggingface_hub as hfh

    print(
        "huggingface_hub:",
        getattr(hfh, "__version__", "?"),
        "from",
        getattr(hfh, "__file__", "?"),
    )
    ds = import_datasets_clean()
    print(
        "datasets:",
        getattr(ds, "__version__", "?"),
        "from",
        getattr(ds, "__file__", "?"),
    )
    return True


# --- huggingface_hub
if not have_hub_commitinfo():
    if os.environ.get("AUTO_INSTALL_DEPS", "1") != "1":
        raise SystemExit("huggingface_hub missing and AUTO_INSTALL_DEPS=0")
    install_hub_overlay()

sys.path.insert(0, pip_target)
import huggingface_hub as hfh

print(
    "huggingface_hub:",
    getattr(hfh, "__version__", "?"),
    "from",
    getattr(hfh, "__file__", "?"),
)

# --- datasets (prefer conda env; overlay only if needed, with numpy pin)
try:
    ds = import_datasets_clean()
    print(
        "datasets:",
        getattr(ds, "__version__", "?"),
        "from",
        getattr(ds, "__file__", "?"),
    )
except Exception as first_err:
    if os.environ.get("AUTO_INSTALL_DEPS", "1") != "1":
        raise SystemExit(f"datasets missing: {first_err}")
    print("datasets import failed:", first_err)
    print("Trying: python -m pip install -U 'datasets>=2.19' into active env ...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-U",
                "datasets>=2.19,<4",
            ]
        )
        ds = import_datasets_clean()
        print(
            "datasets:",
            getattr(ds, "__version__", "?"),
            "from",
            getattr(ds, "__file__", "?"),
        )
    except Exception as env_err:
        print("In-env install failed:", env_err)
        install_datasets_overlay_numpy1()
        if not try_imports_with_overlay():
            raise SystemExit("datasets still not importable after overlay install")
PY
  # Only prepend overlay if something was installed there (avoid masking conda with an empty dir).
  if [[ -d "${PIP_TARGET}/huggingface_hub" || -d "${PIP_TARGET}/datasets" ]]; then
    export PYTHONPATH="${PIP_TARGET}${PYTHONPATH:+:${PYTHONPATH}}"
  fi
fi

python - <<'PY'
import os
import sys

if os.environ.get("AUTO_INSTALL_DEPS", "1") != "1":
    try:
        import huggingface_hub  # noqa: F401
        import datasets  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"AUTO_INSTALL_DEPS=0 but missing dependency: {e}")
    print("AUTO_INSTALL_DEPS=0: huggingface_hub, datasets, torch OK")
    sys.exit(0)

import huggingface_hub as hfh

print("smoke: huggingface_hub OK", getattr(hfh, "__version__", "?"))

import datasets as ds

print("smoke: datasets OK", getattr(ds, "__version__", "?"))

try:
    import torch
except ImportError:
    raise SystemExit("torch missing at smoke test")

print("smoke: torch OK", torch.__version__)
PY
echo "Running: python memorization_detection/run_mimir_decoding_benchmark.py ${EXTRA[*]} ${BENCH_ARGS:-}"

# shellcheck disable=SC2086
python memorization_detection/run_mimir_decoding_benchmark.py "${EXTRA[@]}" ${BENCH_ARGS:-}

echo "Finished job ${JOB_TAG}"
