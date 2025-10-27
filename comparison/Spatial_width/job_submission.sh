#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=kurtwal98
#SBATCH --partition=savio3_htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:00:30
<strong>#SBATCH --array=0-31</strong>
<strong>#SBATCH --output=array_job_%A_task_%a.out</strong>
<strong>#SBATCH --error=array_job_%A_task_%a.err</strong>
## Command(s) to run:
echo "I am task $SLURM_ARRAY_TASK_ID"