#!/bin/bash
#SBATCH --job-name=disc_exp
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-29%7  # 30 combinations: 3 base cases × 2 rH_CV × 5 seeds
#SBATCH --output=logs/array_job_%A_task_%a.out
#SBATCH --error=logs/array_job_%A_task_%a.err
#SBATCH --exclude=n0029.savio3

## Command(s) to run:
# Run complete sweep for each case type (30 tasks each):
# sbatch job_experiment.sh 2x2_4node  # Runs indices 0-29 for 2x2_4node case
# sbatch job_experiment.sh 1x1_4node  # Runs indices 0-29 for 1x1_4node case

set -euo pipefail

# Get case type from command line argument (default to 2x2_4node)
CASE_TYPE=${1:-2x2_4node}

# Validate case type
if [[ ! "$CASE_TYPE" =~ ^(2x2_4node|1x1_4node)$ ]]; then
    echo "Error: Invalid case type '$CASE_TYPE'. Must be one of: 2x2_4node, 1x1_4node" >&2
    exit 1
fi

# Simple early logging
echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Task=${SLURM_ARRAY_TASK_ID:-<local>} Host=$(hostname) Case=${CASE_TYPE}" >&2

# Load modules directly. Remove all locking logic.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "$(date -Is) | MODULE | Purging modules..." >&2
    module purge
    echo "$(date -Is) | MODULE | Loading gcc/13.2.0 openblas/0.3.24..." >&2
    module load gcc/13.2.0 openblas/0.3.24
    echo "$(date -Is) | MODULE | Module load complete." >&2
fi

# Force single-threaded math libs (matches --cpus-per-task=1)
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

# Activate your project venv (absolute path for HPC home)
source /global/home/users/kurtwal98/seiskit/.venv/bin/activate

# Prevent Python bytecode (.pyc) file writing
export PYTHONDONTWRITEBYTECODE=1

# Make OpenSeesPy's native libs visible
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}
fi

# Run from the directory you submitted the job
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Define key paths
PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/run_experiment.py"
PER_TASK_TIMEOUT_SECONDS="${PER_TASK_TIMEOUT_SECONDS:-10000}" # 2h 47m (job is 3h)

# Get the array task ID (for local runs, can be passed as second argument)
TASK_ID=${SLURM_ARRAY_TASK_ID:-${2:-0}}

# Add fail-fast checks for key files
command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python binary not found at ${PYTHON_BIN}" >&2; exit 2; }
test -r "${RUNNER_PY}" >/dev/null || { echo "ERROR: Runner script not readable at ${RUNNER_PY}" >&2; exit 2; }

# Use scratch-backed TMPDIR, not node /tmp
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # We use $USER instead of hardcoding your username
    export TMPDIR="/global/scratch/users/$USER/tmp/job_${SLURM_JOB_ID}_task_${TASK_ID}"
    mkdir -p "$TMPDIR"
    # Set a trap to clean up the TMPDIR on script exit (normal or error)
    trap 'rm -rf "$TMPDIR"' EXIT
else
    # Running locally - use a local temp directory
    export TMPDIR="${SLURM_SUBMIT_DIR:-$PWD}/.tmp/task_${TASK_ID}"
    mkdir -p "${TMPDIR}"
    trap 'rm -rf "$TMPDIR"' EXIT
fi

# Simplify preflight check, shorten timeout
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
echo "Task ${TASK_ID} | Job ${SLURM_JOB_ID:-<local>} | Node: ${SLURMD_NODENAME:-$(hostname)}" >&2
echo "Case: $CASE_TYPE" >&2
echo "Start: $START_DATE" >&2
echo "TMPDIR: $TMPDIR" >&2
echo "============================================================================" >&2

# Remove nested srun. Execute Python directly.
echo "$(date -Is) | RUN | Launching Python task ${TASK_ID}, case ${CASE_TYPE}..." >&2

timeout "${PER_TASK_TIMEOUT_SECONDS}"s \
    ${PYTHON_BIN} -u "${RUNNER_PY}" --case "${CASE_TYPE}" --index "${TASK_ID}"

PYTHON_EXIT_CODE=$?
echo "$(date -Is) | RUN | Python script finished. Exit Code: ${PYTHON_EXIT_CODE}" >&2

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "============================================================================" >&2
echo "Task ${TASK_ID} Completed" >&2
echo "Total Duration (seconds): ${DURATION}" >&2
echo "Python Exit Code: $PYTHON_EXIT_CODE" >&2
echo "============================================================================" >&2

# Exit with Python's exit code
exit $PYTHON_EXIT_CODE