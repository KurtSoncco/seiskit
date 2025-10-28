#!/bin/bash
#SBATCH --job-name=bc-width
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-13
# Total: 2 Lx variability values × 7 BC width values = 14 tasks
#SBATCH --output=array_job_%A_task_%a.out
#SBATCH --error=array_job_%A_task_%a.err

set -euo pipefail

# Clean module env and load required toolchain
module purge
module load gcc/13.2.0
module load openblas/0.3.24 

# Match thread counts to allocated CPUs (helps OpenBLAS and any OpenMP usage)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Activate your project venv (absolute path for HPC home)
source /global/home/users/kurtwal98/seiskit/.venv/bin/activate

# Make OpenSeesPy's native libs visible
export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}

# Run from the directory you submitted the job (keeps relative paths sane)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Record start time
START_TIME=$(date +%s)
START_DATE=$(date)

# Compute task parameters for display
# Index mapping: 0-6 -> Lx=800, 7-13 -> Lx=100
# BC widths: [0, 100, 200, 300, 400, 500, 1000]
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    BC_WIDTHS=(0 100 200 300 400 500 1000)
    TOTAL_BC_WIDTHS=7
    
    if [ "$SLURM_ARRAY_TASK_ID" -lt 7 ]; then
        LX_VAR=800
        BC_IDX=$SLURM_ARRAY_TASK_ID
    else
        LX_VAR=100
        BC_IDX=$((SLURM_ARRAY_TASK_ID - 7))
    fi
    BC_WIDTH=${BC_WIDTHS[$BC_IDX]}
    
    echo "============================================================================"
    echo "SLURM Array Job Information"
    echo "============================================================================"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
    echo "Experiment: Lx_variability=${LX_VAR}m, BC_width=${BC_WIDTH}m"
    echo "Start Time: $START_DATE"
    echo "Node: $SLURMD_NODENAME"
    echo "CPUs per task: $SLURM_CPUS_PER_TASK"
    echo "Working Directory: $(pwd)"
    echo "============================================================================"
    echo ""
else
    echo "============================================================================"
    echo "SLURM Array Job Information"
    echo "============================================================================"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Array Task ID: N/A (not an array task)"
    echo "Start Time: $START_DATE"
    echo "Node: $SLURMD_NODENAME"
    echo "CPUs per task: $SLURM_CPUS_PER_TASK"
    echo "Working Directory: $(pwd)"
    echo "============================================================================"
    echo ""
fi

# Execute the array task; the script reads SLURM_ARRAY_TASK_ID automatically
python run_experiment.py --index "$SLURM_ARRAY_TASK_ID"
PYTHON_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
END_DATE=$(date)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Print timing summary
echo ""
echo "============================================================================"
echo "Job Completed - Timing Summary"
echo "============================================================================"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $START_DATE"
echo "End Time: $END_DATE"
echo "Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Total Duration (seconds): ${DURATION}"
echo "Python Exit Code: $PYTHON_EXIT_CODE"
echo "============================================================================"

# Exit with Python's exit code
exit $PYTHON_EXIT_CODE

