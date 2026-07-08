#!/bin/bash
# Submit phase2 HTC reruns. Pass failed indices as --array=...
# Example: ./submit_phase2.sh --array=12,45,99
set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="${LOG_DIR:-/global/scratch/users/$USER/rv_comparison/logs}"
mkdir -p "$LOG_DIR"
echo "Phase2 logs: $LOG_DIR" >&2

exec sbatch \
  --output="$LOG_DIR/array_job_%A_task_%a.out" \
  --error="$LOG_DIR/array_job_%A_task_%a.err" \
  "$@" \
  phase2_htc.sh
