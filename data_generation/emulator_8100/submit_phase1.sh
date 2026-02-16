#!/bin/bash
# Ensure logs/ exists before Slurm opens -o/-e, then submit phase1.
cd "$(dirname "$0")"
mkdir -p logs
exec sbatch phase1_savio2.sh "$@"
