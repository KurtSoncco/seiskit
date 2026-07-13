#!/bin/bash
# Submit 2D GRF + Pretell block (requires RV_SMOKE_2D=1 in smoke mode).
# Run after submit_phase1_hallal.sh completes.
#
# Usage:
#   ./submit_phase1_rf.sh              # full: 5,120 indices (40 RF × 2 methods × 64)
#   RV_SMOKE=1 RV_SMOKE_2D=1 ./submit_phase1_rf.sh --smoke   # smoke: 40 indices → array 0-1
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
if [ "$SMOKE" = "1" ] && [ "${RV_SMOKE_2D:-0}" != "1" ]; then
  echo "ERROR: smoke RF phase requires RV_SMOKE_2D=1 (2D arms disabled in default smoke)." >&2
  echo "  RV_SMOKE=1 RV_SMOKE_2D=1 ./submit_phase1_rf.sh --smoke" >&2
  exit 1
fi
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
