#!/bin/bash
# Submit phase1 (Savio2 + GNU Parallel). Creates logs/ before Slurm opens -o/-e.
# Usage:
#   ./submit_phase1.sh              # full campaign (31,360 indices; array size from manifest)
#   ./submit_phase1.sh --smoke       # smoke 1D only (120 indices)
#   ./submit_phase1.sh --smoke --2d  # smoke with 2D (160 indices)
#   ./submit_phase1.sh --array=0-10  # pass extra sbatch args
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ROOT="$(cd ../.. && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON:-python3}"
fi

SMOKE=0
INCLUDE_2D=0
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --2d) INCLUDE_2D=1 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

if [ "$SMOKE" = "1" ]; then
  EXPORT="RV_SMOKE=1"
  if [ "$INCLUDE_2D" = "1" ]; then
    EXPORT="RV_SMOKE=1,RV_SMOKE_2D=1"
  fi
  LAST=$("$PYTHON" -c "import os; os.environ['RV_SMOKE']='1'; os.environ['RV_SMOKE_2D']='${INCLUDE_2D}'; from manifest import phase1_array_tasks; print(phase1_array_tasks()-1)")
  echo "Smoke submit: export ${EXPORT}, array 0-${LAST}" >&2
  exec sbatch --export=ALL,"${EXPORT}" --array=0-"${LAST}" "${EXTRA[@]}" phase1_savio2.sh
fi

TOTAL=$("$PYTHON" -c "import os; os.environ['RV_SMOKE']='0'; from manifest import phase1_array_tasks, total_combinations; print(f'{phase1_array_tasks()-1} {total_combinations()}')")
LAST=${TOTAL%% *}
N_IDX=${TOTAL##* }
echo "Full submit: ${N_IDX} indices, array 0-${LAST}" >&2
exec sbatch --export=ALL,RV_SMOKE=0 --array=0-"${LAST}" "${EXTRA[@]}" phase1_savio2.sh
