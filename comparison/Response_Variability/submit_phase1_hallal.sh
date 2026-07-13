#!/bin/bash
# Submit Hallal 1D block only (indices 0 .. hallal_block_size-1).
# Run this before submit_phase1_rf.sh for the 2D GRF / Pretell arms.
#
# Usage:
#   ./submit_phase1_hallal.sh              # full: 38,400 indices → array 0-1599
#   ./submit_phase1_hallal.sh --smoke      # smoke: 120 indices → array 0-4
#   ./submit_phase1_hallal.sh --array=0-10 # extra sbatch args
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ROOT="$(cd ../.. && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON:-python3}"
fi

SMOKE=0
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

export RV_SMOKE=$SMOKE
HALLAL_END=$("$PYTHON" -c "from manifest import hallal_index_end, phase1_array_tasks; print(hallal_index_end())")
N_TASKS=$("$PYTHON" -c "from manifest import phase1_array_tasks; print(phase1_array_tasks(index_offset=0, index_end=int('${HALLAL_END}')))")
LAST=$((N_TASKS - 1))

echo "Hallal 1D phase: indices 0..$((HALLAL_END - 1)) (${HALLAL_END} runs, array 0-${LAST})" >&2

exec sbatch \
  --export=ALL,RV_SMOKE=${SMOKE},RV_INDEX_OFFSET=0,RV_INDEX_MAX=${HALLAL_END} \
  --array=0-${LAST} \
  "${EXTRA[@]}" \
  phase1_savio2.sh
