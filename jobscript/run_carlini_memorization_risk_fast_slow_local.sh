#!/usr/bin/env bash
# Memorization risk decoders only (fast + slow), 500 samples per strategy per model.
# Does **not** regenerate top_k / top_k_internet / temperature_decay; outputs go to
# ``mia_eval_outputs/carlini_extract_memorization_risk_fast_slow/<run_key>/`` (see overlay YAML).
#
# From repo root:
#   ./jobscript/run_carlini_memorization_risk_fast_slow_local.sh
#   ./jobscript/run_carlini_memorization_risk_fast_slow_local.sh --only dclm_7b_base
#   SKIP_EXISTING=1 ./jobscript/run_carlini_memorization_risk_fast_slow_local.sh
#   NUM_SAMPLES_PER_STRATEGY=500 ./jobscript/run_carlini_memorization_risk_fast_slow_local.sh
#
# Cluster (separate wall times for fast vs slow): ``sbatch jobscript/run_carlini_memorization_risk_fast_array.slurm``
# and ``sbatch jobscript/run_carlini_memorization_risk_slow_array.slurm``.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${CARLINI_CONFIG:-mia_eval/config/carlini_open_models.yaml}"
MERGE="${CARLINI_MERGE_CONFIG:-mia_eval/config/carlini_overlay_memorization_risk_fast_slow.yaml}"
_abs_if_needed() {
  local p="$1"
  if [[ "${p}" == /* ]]; then
    printf '%s' "${p}"
  else
    printf '%s' "${ROOT}/${p}"
  fi
}
EXTRA=(--config "$(_abs_if_needed "${CONFIG}")" --merge-config "$(_abs_if_needed "${MERGE}")")
if [[ "${SKIP_EXISTING:-0}" == 1 ]]; then
  EXTRA+=(--skip-existing)
fi
if [[ -n "${NUM_SAMPLES_PER_STRATEGY:-}" ]]; then
  EXTRA+=(--num-samples-per-strategy "${NUM_SAMPLES_PER_STRATEGY}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  EXTRA+=(--batch-size "${BATCH_SIZE}")
fi

exec python -m mia_eval.run_carlini_extraction_batch "${EXTRA[@]}" "$@"
