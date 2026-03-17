#!/bin/bash
# Ensure logs/ exists before Slurm opens -o/-e, then submit phase1.
# Pass --array to override script default and run only these tasks.
# 0-10: validation
# 11-336: training
cd "$(dirname "$0")"
mkdir -p logs
exec sbatch --export="FORCE_RERUN=1" --array=136-141,143-152,155-157,159-162,168,195,213,218,226,228,230-231,236-238,240-243,245-247,249-250,252-261,317,322,324,326-327,329,331,334,336 phase1_savio2.sh "$@"
