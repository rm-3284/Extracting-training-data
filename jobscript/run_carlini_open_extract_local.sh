#!/usr/bin/env bash
# Run all open-model Carlini extractions sequentially on one machine (needs GPU + HF cache).
# Usage from repo root:
#   ./jobscript/run_carlini_open_extract_local.sh
#   ./jobscript/run_carlini_open_extract_local.sh --only olmo2_7b_base,redpajama_7b_base
#   NUM_SAMPLES_PER_STRATEGY=500 ./jobscript/run_carlini_open_extract_local.sh
#   SKIP_EXISTING=1 ./jobscript/run_carlini_open_extract_local.sh   # add --skip-existing

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${CARLINI_CONFIG:-mia_eval/config/carlini_open_models.yaml}"
EXTRA=(--config "${CONFIG}")
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
