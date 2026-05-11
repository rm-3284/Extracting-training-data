# MIMIR benchmark: HF caches, pip overlay, smoke tests, run Python benchmark.
# Expects JOB_TAG and logs/${JOB_TAG} already set. Optional: MIMIR_BENCH_SINGLE_MODEL
# adds --models <key> for array shards (one model per task).
# Optional env → CLI (see run_mimir_decoding_benchmark_array*.slurm headers): MIMIR_FAST_LEGACY_LOGPROB,
# MIMIR_INFILLING_PENALTY_SIGN, MIMIR_FAST_INFILLING_*, MIMIR_RISK_SCORE_MODE, MIMIR_RISK_EXPLORE_EPS,
# MIMIR_FAST_AUX_LOGPROB_LAMBDA, MIMIR_SLOW_AUX_LOGPROB_LAMBDA, MIMIR_WBC_REFERENCE_MODEL, MIMIR_WBC_*, etc.

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it before sbatch, or use:" >&2
  echo "  sbatch --export=ALL,HF_TOKEN=\"\$HF_TOKEN\" jobscript/run_mimir_decoding_benchmark.slurm" >&2
  echo "  sbatch --export=ALL,HF_TOKEN=\"\$HF_TOKEN\" jobscript/run_mimir_decoding_benchmark_array.slurm" >&2
  echo "  sbatch --export=ALL,HF_TOKEN=\"\$HF_TOKEN\" jobscript/run_mimir_decoding_benchmark_array_slow_6h.slurm" >&2
  exit 1
fi

# Cache on shared FS (override HF_HOME if you prefer another path)
# Hugging Face cache (HF_HOME covers transformers + hub; avoid TRANSFORMERS_CACHE — deprecated in transformers v5 roadmap)
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.cache}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

# ~6h target on GPU (Slurm sets MIMIR_BENCH_PROFILE=6h by default). Use MIMIR_BENCH_PROFILE=full
# for the full slow benchmark and raise --time / split across array tasks.
if [[ "${MIMIR_BENCH_PROFILE:-}" == "6h" ]]; then
  export SKIP_SLOW="${SKIP_SLOW:-1}"
  export N_EXAMPLES="${N_EXAMPLES:-25}"
  if [[ -z "${BENCH_ARGS:-}" ]]; then
    export BENCH_ARGS="--risk-every 4 --top-k 10"
  fi
  echo "MIMIR_BENCH_PROFILE=6h: SKIP_SLOW=${SKIP_SLOW} N_EXAMPLES=${N_EXAMPLES:-} BENCH_ARGS=${BENCH_ARGS:-}"
fi

# Slow path + ~6h tasks (pair with example-sharded array jobs; budgets usually set in that Slurm script).
if [[ "${MIMIR_BENCH_PROFILE:-}" == "slow_6h" ]]; then
  export SKIP_SLOW="${SKIP_SLOW:-0}"
  echo "MIMIR_BENCH_PROFILE=slow_6h: SKIP_SLOW=${SKIP_SLOW:-0} N_EXAMPLES=${N_EXAMPLES:-} BENCH_ARGS=${BENCH_ARGS:-}"
fi

N_EXAMPLES="${N_EXAMPLES:-50}"
OUT_JSON="${BENCH_OUTPUT:-logs/${JOB_TAG}/mimir_bench.json}"

EXTRA=(--n-examples "${N_EXAMPLES}" --output "${OUT_JSON}")
if [[ "${SKIP_SLOW:-0}" == "1" ]]; then
  EXTRA+=(--skip-slow)
fi
if [[ -n "${MIMIR_BENCH_SINGLE_MODEL:-}" ]]; then
  EXTRA+=(--models "${MIMIR_BENCH_SINGLE_MODEL}")
fi

# Optional: memorization_detection/run_mimir_decoding_benchmark.py flags (see also BENCH_ARGS).
if [[ "${MIMIR_FAST_LEGACY_LOGPROB:-0}" == "1" ]]; then
  EXTRA+=(--fast-legacy-logprob)
fi
if [[ -n "${MIMIR_INFILLING_PENALTY_SIGN:-}" ]]; then
  EXTRA+=(--infilling-penalty-sign "${MIMIR_INFILLING_PENALTY_SIGN}")
fi
if [[ -n "${MIMIR_FAST_INFILLING_WINDOW:-}" ]]; then
  EXTRA+=(--fast-infilling-window "${MIMIR_FAST_INFILLING_WINDOW}")
fi
if [[ -n "${MIMIR_FAST_INFILLING_M:-}" ]]; then
  EXTRA+=(--fast-infilling-m "${MIMIR_FAST_INFILLING_M}")
fi
if [[ -n "${MIMIR_FAST_INFILLING_K:-}" ]]; then
  EXTRA+=(--fast-infilling-k "${MIMIR_FAST_INFILLING_K}")
fi
if [[ -n "${MIMIR_RISK_SCORE_MODE:-}" ]]; then
  EXTRA+=(--risk-score-mode "${MIMIR_RISK_SCORE_MODE}")
fi
if [[ -n "${MIMIR_RISK_EXPLORE_EPS:-}" ]]; then
  EXTRA+=(--risk-explore-eps "${MIMIR_RISK_EXPLORE_EPS}")
fi
if [[ -n "${MIMIR_FAST_AUX_LOGPROB_LAMBDA:-}" ]]; then
  EXTRA+=(--fast-aux-logprob-lambda "${MIMIR_FAST_AUX_LOGPROB_LAMBDA}")
fi
if [[ -n "${MIMIR_SLOW_AUX_LOGPROB_LAMBDA:-}" ]]; then
  EXTRA+=(--slow-aux-logprob-lambda "${MIMIR_SLOW_AUX_LOGPROB_LAMBDA}")
fi
# WBC loads a second HF model (high VRAM); e.g. Pythia 2.8B + EleutherAI/pythia-70m with MIMIR_WBC_SHARE_TARGET_TOKENIZER=1
if [[ -n "${MIMIR_WBC_REFERENCE_MODEL:-}" ]]; then
  EXTRA+=(--wbc-reference-model "${MIMIR_WBC_REFERENCE_MODEL}")
  [[ -n "${MIMIR_WBC_LAMBDA:-}" ]] && EXTRA+=(--wbc-lambda "${MIMIR_WBC_LAMBDA}")
  [[ -n "${MIMIR_WBC_INFILLING_LAMBDA:-}" ]] && EXTRA+=(--wbc-infilling-lambda "${MIMIR_WBC_INFILLING_LAMBDA}")
  [[ -n "${MIMIR_WBC_GATE_GAMMA:-}" ]] && EXTRA+=(--wbc-gate-gamma "${MIMIR_WBC_GATE_GAMMA}")
  [[ -n "${MIMIR_WBC_GATE_EVERY:-}" ]] && EXTRA+=(--wbc-gate-every "${MIMIR_WBC_GATE_EVERY}")
  [[ -n "${MIMIR_MAX_NEW_TOKENS_WBC:-}" ]] && EXTRA+=(--max-new-tokens-wbc "${MIMIR_MAX_NEW_TOKENS_WBC}")
  if [[ "${MIMIR_WBC_SHARE_TARGET_TOKENIZER:-0}" == "1" ]]; then
    EXTRA+=(--wbc-share-target-tokenizer)
  fi
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
