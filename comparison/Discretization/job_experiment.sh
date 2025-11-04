#!/bin/bash
#SBATCH --job-name=disc_exp
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/disc_exp_%A_%a.out
#SBATCH --error=logs/disc_exp_%A_%a.err

# Get case type from command line argument (default to 2x2_4node)
CASE_TYPE=${1:-2x2_4node}

# Validate case type
if [[ ! "$CASE_TYPE" =~ ^(2x2_4node|1x1_4node|2x2_8node)$ ]]; then
    echo "Error: Invalid case type '$CASE_TYPE'. Must be one of: 2x2_4node, 1x1_4node, 2x2_8node"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the array task ID
TASK_ID=$SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "Starting discretization experiment"
echo "Case: $CASE_TYPE"
echo "Task ID: $TASK_ID"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

# Change to script directory
cd "$(dirname "$0")"

# Run the experiment
python run_experiment.py --case "$CASE_TYPE" --index "$TASK_ID"

echo "=========================================="
echo "Completed task $TASK_ID for case $CASE_TYPE"
echo "Date: $(date)"
echo "=========================================="

