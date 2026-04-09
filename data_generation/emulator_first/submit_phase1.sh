#!/bin/bash
# Ensure logs/ exists before Slurm opens -o/-e, then submit phase1.
# Pass --array to override script default and run only these tasks.
# 0-10: validation
# 11-336: training
cd "$(dirname "$0")"
mkdir -p logs
exec sbatch --array=0-10 phase1_savio2.sh "$@"
