#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3_htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:00:30
#SBATCH --array=0-31
#SBATCH --output=array_job_%A_task_%a.out
#SBATCH --error=array_job_%A_task_%a.err
## Command(s) to run:
echo "I am task $SLURM_ARRAY_TASK_ID"