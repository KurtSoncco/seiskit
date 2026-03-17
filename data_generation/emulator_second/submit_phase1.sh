#!/bin/bash
# Ensure logs/ exists before Slurm opens -o/-e, then submit phase1.
# Pass --array to override script default and run only these tasks.
# 0-10: validation
# 11-336: training
cd "$(dirname "$0")"
mkdir -p logs
exec sbatch --export="FORCE_RERUN=1" --array=240,242-243,245-247,249-250,252-259,261,317,322,324,326-327,329,331,334,336 phase1_savio2.sh "$@"
