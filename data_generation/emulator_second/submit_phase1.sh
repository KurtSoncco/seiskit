#!/bin/bash
# Ensure logs/ exists before Slurm opens -o/-e, then submit phase1.
# Pass --array to override script default and run only these tasks.
# 0-10: validation
# 11-336: training
cd "$(dirname "$0")"
mkdir -p logs
exec sbatch --export="FORCE_RERUN=1" --array=94-116,118-141,143-157,159-164,166-169,174,190-191,193,195,202,207,210-211,213,215,218-220,223,226,228-231,233-247,249-263,265-268,270-273,275-276,280,293,298,300,302,305,307,313,317,320,322,324,326-327,329,331,334,336 phase1_savio2.sh "$@"
