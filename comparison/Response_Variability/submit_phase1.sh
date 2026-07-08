#!/bin/bash
# Submit phase1 (Savio2 + GNU Parallel). Creates logs/ before Slurm opens -o/-e.
# Usage:
#   ./submit_phase1.sh              # full TF-first campaign (1,000 indices, array 0-41)
#   ./submit_phase1.sh --smoke       # smoke test (50 indices, array 0-2)
#   ./submit_phase1.sh --array=0-10  # pass extra sbatch args
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

SMOKE=0
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

if [ "$SMOKE" = "1" ]; then
  exec sbatch --export=ALL,RV_SMOKE=1 --array=0-2 "${EXTRA[@]}" phase1_savio2.sh
fi

exec sbatch --export=ALL,RV_SMOKE=0 --array=0-41 "${EXTRA[@]}" phase1_savio2.sh
