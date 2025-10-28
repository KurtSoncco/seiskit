#!/bin/bash
#SBATCH --job-name=bc-width-plot
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=plot_job_%j.out
#SBATCH --error=plot_job_%j.err

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

echo "============================================================================"
echo "BC Width Plotting Job"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $START_DATE"
echo "Node: $SLURMD_NODENAME"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Working Directory: $(pwd)"
echo "============================================================================"
echo ""

# Run the plotting script
echo "Running plotting script..."
python run_experiment.py --plot
PYTHON_EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
END_DATE=$(date)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================================"
echo "Plotting Job Completed - Timing Summary"
echo "============================================================================"
echo "Start Time: $START_DATE"
echo "End Time: $END_DATE"
echo "Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Total Duration (seconds): ${DURATION}"
echo "Python Exit Code: $PYTHON_EXIT_CODE"
echo "============================================================================"

exit $PYTHON_EXIT_CODE

