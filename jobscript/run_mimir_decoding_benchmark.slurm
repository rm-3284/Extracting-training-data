#!/bin/bash
#SBATCH --job-name=mimir_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=all
#SBATCH --output=logs/mimir_bench_%j.out
#SBATCH --error=logs/mimir_bench_%j.err
#SBATCH --chdir=/n/fs/vision-mix/rm4411/Extracting-training-data
#
# MIMIR decoding benchmark: baseline + fast (+ optional slow) on three open LMs (see
# memorization_detection/run_mimir_decoding_benchmark.py).
#
# Default targets finishing within ~6h (baseline + fast only, 25 examples, cheaper fast loop).
# Override: export MIMIR_BENCH_PROFILE=full (and raise #SBATCH --time) for slow mode + 50 examples.
#
# Long full benchmark: jobscript/run_mimir_decoding_benchmark_array_slow_6h.slurm (slow, 12×6h tasks)
# or array + PROFILE=full / longer --time, then merge shards:
#   python jobscript/merge_mimir_benchmark_shards.py -o logs/mimir_bench_merged.json \\
#     logs/mimir_shard_<ARRAY_JOB_ID>_*.json
# You can also shard examples with BENCH_ARGS e.g. --example-offset / --n-examples on each task.
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
#   MIMIR_BENCH_PROFILE  6h (default via this script) | full — 6h skips slow, caps examples unless set
#   N_EXAMPLES        --n-examples (default: 50, or 25 under PROFILE=6h if unset)
#   BENCH_OUTPUT      --output JSON path (default: logs/${SLURM_JOB_ID}/mimir_bench.json)
#   SKIP_SLOW         set to 1 to pass --skip-slow
#   AUTO_INSTALL_DEPS set to 0 to disable auto pip install when missing
#                     (plain `pip show` may be a different Python than `python` after conda activate;
#                      always use: conda activate ENV && python -m pip show datasets)
#   PIP_OVERLAY_DIR   writable dir for per-job overlay installs (default: ${PROJECT_ROOT}/.slurm_pip_overlay)
#   TMP_BASE          temp dir Slurm/prologue use (default: ${PROJECT_ROOT}/tmp — avoids /scratch/... if missing)
#   BENCH_ARGS        optional extra CLI tokens (unquoted expansion), e.g.:
#                       export BENCH_ARGS="--verbose --seed 1 --models gpt_neo_2p7"
#   MIMIR_BENCH_SINGLE_MODEL  if set, passed as --models <key> (used by array Slurm; rarely needed manually)
#   MIMIR_FAST_LEGACY_LOGPROB=1, MIMIR_INFILLING_PENALTY_SIGN, MIMIR_FAST_INFILLING_WINDOW/M/K
#   MIMIR_WBC_REFERENCE_MODEL (+ MIMIR_WBC_LAMBDA, MIMIR_WBC_GATE_EVERY, MIMIR_WBC_SHARE_TARGET_TOKENIZER=1, …)
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

# shellcheck source=jobscript/_mimir_bench_env_conda_torch.sh
source "${PROJECT_ROOT}/jobscript/_mimir_bench_env_conda_torch.sh"

JOB_TAG="${SLURM_JOB_ID:-manual}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  JOB_TAG="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
mkdir -p "logs/${JOB_TAG}"

export MIMIR_BENCH_PROFILE="${MIMIR_BENCH_PROFILE:-6h}"

# shellcheck source=jobscript/_mimir_bench_after_jobtag.sh
source "${PROJECT_ROOT}/jobscript/_mimir_bench_after_jobtag.sh"
