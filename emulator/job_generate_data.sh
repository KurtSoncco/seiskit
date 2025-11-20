#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/array_job_%A_task_%a.out
#SBATCH --error=logs/array_job_%A_task_%a.err

# SLURM job script for unified data generation (materials + HF)
# This script can generate materials, HF simulations, or both
#
# Usage:
#   # Generate both materials and HF (default)
#   sbatch job_generate_data.sh
#
#   # To customize, edit the --array range above and parameters below

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

# Make OpenSeesPy's native libs visible
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}
fi

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
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/generate_data_unified_SLURM.py"
PER_TASK_TIMEOUT_SECONDS="${PER_TASK_TIMEOUT_SECONDS:-14400}"  # 4 hours

# Get the array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}

# Add fail-fast checks for key files
command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python binary not found at ${PYTHON_BIN}" >&2; exit 2; }
test -r "${RUNNER_PY}" >/dev/null || { echo "ERROR: Runner script not readable at ${RUNNER_PY}" >&2; exit 2; }

# Use scratch-backed TMPDIR, not node /tmp
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export TMPDIR="/global/scratch/users/$USER/tmp/job_${SLURM_JOB_ID}_task_${TASK_ID}"
    mkdir -p "$TMPDIR"
    trap 'rm -rf "$TMPDIR"' EXIT
else
    export TMPDIR="${SLURM_SUBMIT_DIR:-$PWD}/.tmp/task_${TASK_ID}"
    mkdir -p "${TMPDIR}"
    trap 'rm -rf "${TMPDIR}"' EXIT
fi

# Preflight check
echo "$(date -Is) | PREFLIGHT | Verifying Python and OpenSees..." >&2
timeout 30s ${PYTHON_BIN} - <<'PYEOF'
import sys
print('PYTHON_OK', sys.version.split()[0])
try:
    import openseespy.opensees as ops  # noqa
    print('OPENSEES_OK')
except Exception as e:
    print(f'OPENSEES_IMPORT_FAIL: {e}')
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
echo "Unified Data Generation - Task ${TASK_ID} | Job ${SLURM_JOB_ID:-<local>} | Node: ${SLURMD_NODENAME:-$(hostname)}" >&2
echo "Start: $START_DATE" >&2
echo "TMPDIR: $TMPDIR" >&2
echo "============================================================================" >&2

# Execute the data generation
echo "$(date -Is) | RUN | Launching data generation for task ${TASK_ID}..." >&2

# Default parameters (can be overridden via environment variables)
# --mode: "lf" (materials+LF) or "both" (materials+LF+HF, default)
# --start_idx: Starting index for simulation IDs (default: 1000)
# --data_dir: Data directory (default: data)
# --hx: LF element size (default: 10.0 m)
# --hx_hf: HF element size (default: 1.0 m)
# --Lx: Domain width (default: 150.0 m)
# --Lz: Domain height (default: 150.0 m)
# --duration: Simulation duration (default: 25.0 s)
# --dt_hf: HF time step (default: 0.01 s)

MODE="${MODE:-both}"
START_IDX="${START_IDX:-1000}"

timeout "${PER_TASK_TIMEOUT_SECONDS}"s \
    ${PYTHON_BIN} -u "${RUNNER_PY}" \
        --data_dir data \
        --mode "${MODE}" \
        --start_idx "${START_IDX}" \
        --hx 10.0 \
        --hx_hf 1.0 \
        --Lx 150.0 \
        --Lz 150.0 \
        --duration 25.0 \
        --dt_hf 0.01

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

