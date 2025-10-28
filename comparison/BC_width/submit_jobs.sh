#!/bin/bash
#SBATCH --job-name=bc-width-full
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --output=full_workflow_%j.out
#SBATCH --error=full_workflow_%j.err

set -euo pipefail

# Clean module env and load required toolchain
module purge
module load gcc/13.2.0
module load openblas/0.3.24 

# Match thread counts to allocated CPUs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Activate your project venv
source /global/home/users/kurtwal98/seiskit/.venv/bin/activate

# Make OpenSeesPy's native libs visible
export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}

# Run from the directory you submitted the job
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "============================================================================"
echo "BC Width Full Workflow: Run analysis for all indices then plot"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "============================================================================"
echo ""

# Step 1: Run all array tasks sequentially (for testing/development)
echo "Step 1: Running all 14 analysis tasks..."
for i in {0..13}; do
    echo ""
    echo "--- Running task index $i ---"
    python run_experiment.py --index $i
    if [ $? -ne 0 ]; then
        echo "ERROR: Task $i failed!"
        exit 1
    fi
done

echo ""
echo "============================================================================"
echo "Step 2: Generating plots..."
echo "============================================================================"

# Step 2: Generate plots
python run_experiment.py --plot
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "Workflow completed successfully!"
    echo "============================================================================"
    echo "All plots saved in the current directory"
    echo "- transfer_functions_comparison.html"
    echo "- acceleration_time_histories_comparison.html"
    echo "- *_surface_nodes_acceleration_stacked.png"
    echo "============================================================================"
else
    echo ""
    echo "ERROR: Plotting failed!"
    exit $PYTHON_EXIT_CODE
fi

exit 0

