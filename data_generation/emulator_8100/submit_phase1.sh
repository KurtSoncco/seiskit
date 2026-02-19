#!/bin/bash
# Ensure logs/ exists before Slurm opens -o/-e, then submit phase1.
# Missing results: 108, 111
# Pass --array to override script default and run only these tasks.
cd "$(dirname "$0")"
mkdir -p logs
exec sbatch --array=115-117 phase1_savio2.sh "$@"
