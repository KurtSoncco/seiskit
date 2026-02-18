#!/bin/bash
# Ensure logs/ exists before Slurm opens -o/-e, then submit phase1.
# Missing results: 432-455 (task 18), 1824-1847 (task 76), 2172-2831 (tasks 90-117).
# Pass --array to override script default and run only these tasks.
cd "$(dirname "$0")"
mkdir -p logs
exec sbatch --array=18,76,90-117 phase1_savio2.sh "$@"
