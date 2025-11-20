#!/bin/bash
#SBATCH --job-name=generate_lf_materials
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --array=0-99
#SBATCH --output=logs/lf_materials_%A_task_%a.out
#SBATCH --error=logs/lf_materials_%A_task_%a.err

# SLURM job script for generating LF material grids and parameters
# This script generates only the material data needed for HF generation
# (material grids and parameters), without running simulations
#
# Usage:
#   # Generate materials for 100 test samples (default test set size)
#   sbatch job_generate_lf_materials.sh
#
#   # To customize, edit the --array range above (e.g., --array=0-199 for 200 samples)
#   # and adjust --start_idx in the Python command below

set -euo pipefail

# Simple early logging
echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Task=${SLURM_ARRAY_TASK_ID:-<local>} Host=$(hostname)" >&2

# Load modules
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "$(date -Is) | MODULE | Purging modules..." >&2
    module purge
    echo "$(date -Is) | MODULE | Loading gcc/13.2.0 openblas/0.3.24..." >&2
    module load gcc/13.2.0 openblas/0.3.24
    echo "$(date -Is) | MODULE | Module load complete." >&2
fi

# Activate your project venv (absolute path for HPC home)
source /global/home/users/kurtwal98/seiskit/.venv/bin/activate

# Prevent Python bytecode (.pyc) file writing
export PYTHONDONTWRITEBYTECODE=1

# Match thread counts to allocated CPUs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Run from the directory you submitted the job
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Define key paths
PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/generate_lf_materials_SLURM.py"
PER_TASK_TIMEOUT_SECONDS="${PER_TASK_TIMEOUT_SECONDS:-1800}"  # 30 minutes

# Get the array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}

# Add fail-fast checks for key files
command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python binary not found at ${PYTHON_BIN}" >&2; exit 2; }
test -r "${RUNNER_PY}" >/dev/null || { echo "ERROR: Runner script not readable at ${RUNNER_PY}" >&2; exit 2; }

# Preflight check
echo "$(date -Is) | PREFLIGHT | Verifying Python..." >&2
timeout 30s ${PYTHON_BIN} - <<'PYEOF'
import sys
print('PYTHON_OK', sys.version.split()[0])
try:
    import numpy as np
    print('NUMPY_OK')
except Exception as e:
    print(f'NUMPY_IMPORT_FAIL: {e}')
    sys.exit(3)
PYEOF
PRE_RC=$?
if [ ${PRE_RC} -ne 0 ]; then
  echo "$(date -Is) | ERROR | Preflight check failed (Code: ${PRE_RC}). Exiting." >&2
  exit ${PRE_RC}
fi
echo "$(date -Is) | PREFLIGHT | Preflight success." >&2

# Record start time
START_TIME=$(date +%s)
START_DATE=$(date)

echo "============================================================================" >&2
echo "LF Material Generation - Task ${TASK_ID} | Job ${SLURM_JOB_ID:-<local>} | Node: ${SLURMD_NODENAME:-$(hostname)}" >&2
echo "Start: $START_DATE" >&2
echo "============================================================================" >&2

# Execute the material generation
echo "$(date -Is) | RUN | Launching material generation for task ${TASK_ID}..." >&2

# Default parameters (adjust as needed):
# --start_idx: Starting index for simulation IDs (default: 1000, assuming 1000 train + 100 val)
# --data_dir: Data directory (default: data)
# --hx: LF element size (default: 10.0 m)
# --hx_hf: HF element size (default: 1.0 m)
# --Lx: Domain width (default: 150.0 m)
# --Lz: Domain height (default: 150.0 m)

# Allow override via environment variable (set in submit script or sbatch command)
START_IDX="${START_IDX:-1000}"

timeout "${PER_TASK_TIMEOUT_SECONDS}"s \
    ${PYTHON_BIN} -u "${RUNNER_PY}" \
        --data_dir data \
        --start_idx "${START_IDX}" \
        --hx 10.0 \
        --hx_hf 1.0 \
        --Lx 150.0 \
        --Lz 150.0

PYTHON_EXIT_CODE=$?
echo "$(date -Is) | RUN | Python script finished. Exit Code: ${PYTHON_EXIT_CODE}" >&2

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "============================================================================" >&2
echo "Task ${TASK_ID} Completed" >&2
echo "Start Time: $START_DATE" >&2
echo "End Time: $(date)" >&2
echo "Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s (${DURATION}s)" >&2
echo "Python Exit Code: $PYTHON_EXIT_CODE" >&2
echo "============================================================================" >&2

# Exit with Python's exit code
exit $PYTHON_EXIT_CODE

