#!/bin/bash
# Submit 2D GRF + de la Torre block (indices hallal_block_size .. total-1).
# Run after submit_phase1_hallal.sh completes.
#
# Usage:
#   ./submit_phase1_rf.sh              # full: 3,840 indices → array 0-159
#   ./submit_phase1_rf.sh --smoke      # smoke: 40 indices → array 0-1
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
read -r RF_START RF_END <<< "$("$PYTHON" -c "from manifest import rf_index_range; s,e=rf_index_range(); print(s, e)")"
N_TASKS=$("$PYTHON" -c "from manifest import phase1_array_tasks; print(phase1_array_tasks(index_offset=${RF_START}, index_end=${RF_END}))")
LAST=$((N_TASKS - 1))
COUNT=$((RF_END - RF_START))

echo "RF phase: indices ${RF_START}..$((RF_END - 1)) (${COUNT} runs, array 0-${LAST})" >&2

exec sbatch \
  --export=ALL,RV_SMOKE=${SMOKE},RV_INDEX_OFFSET=${RF_START},RV_INDEX_MAX=${RF_END} \
  --array=0-${LAST} \
  "${EXTRA[@]}" \
  phase1_savio2.sh
