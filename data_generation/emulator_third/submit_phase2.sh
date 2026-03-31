#!/bin/bash
# Submit phase2 via sbatch (job arrays require sbatch, not srun).
# Creates logs on scratch so Slurm -o/-e are in a writeable location.
set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="${LOG_DIR:-/global/scratch/users/$USER/emulator8100_logs}"
mkdir -p "$LOG_DIR"
echo "Phase2 logs: $LOG_DIR" >&2

exec sbatch \
  --output="$LOG_DIR/array_job_%A_task_%a.out" \
  --error="$LOG_DIR/array_job_%A_task_%a.err" \
  phase2_htc.sh "$@"
