#!/bin/bash
#SBATCH --job-name=parametric-study
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-44%10
#SBATCH --output=logs/array_job_%A_task_%a.out
#SBATCH --error=logs/array_job_%A_task_%a.err
#SBATCH --exclude=n0029

set -euo pipefail

# Simple early logging
echo "$(date -Is) | START | Job=${SLURM_JOB_ID} Task=${SLURM_ARRAY_TASK_ID} Host=$(hostname)" >&2

# Load modules directly. Remove all locking logic.
echo "$(date -Is) | MODULE | Purging modules..." >&2
module purge
echo "$(date -Is) | MODULE | Loading gcc/13.2.0 openblas/0.3.24..." >&2
module load gcc/13.2.0 openblas/0.3.24
echo "$(date -Is) | MODULE | Module load complete." >&2

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
export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}

# Run from the directory you submitted the job
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Define key paths
PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/run_experiment.py"
PER_TASK_TIMEOUT_SECONDS="${PER_TASK_TIMEOUT_SECONDS:-6300}" # 1h 45m (job is 2h)

# 3. Add fail-fast checks for key files
command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python binary not found at ${PYTHON_BIN}"; exit 2; }
test -r "${RUNNER_PY}" >/dev/null || { echo "ERROR: Runner script not readable at ${RUNNER_PY}"; exit 2; }

# 4. Use scratch-backed TMPDIR, not node /tmp
# We use $USER instead of hardcoding your username
export TMPDIR="/global/scratch/users/$USER/tmp/job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
# Set a trap to clean up the TMPDIR on script exit (normal or error)
trap 'rm -rf "$TMPDIR"' EXIT

# 5. Simplify preflight check, shorten timeout
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
echo "Task ${SLURM_ARRAY_TASK_ID} | Job ${SLURM_JOB_ID} | Node: $SLURMD_NODENAME" >&2
echo "Start: $START_DATE" >&2
echo "TMPDIR: $TMPDIR" >&2
echo "============================================================================" >&2

# 6. Remove nested srun. Execute Python directly.
echo "$(date -Is) | RUN | Launching Python task ${SLURM_ARRAY_TASK_ID}..." >&2

timeout "${PER_TASK_TIMEOUT_SECONDS}"s \
    ${PYTHON_BIN} -u "${RUNNER_PY}" --index "${SLURM_ARRAY_TASK_ID}"

PYTHON_EXIT_CODE=$?
echo "$(date -Is) | RUN | Python script finished. Exit Code: ${PYTHON_EXIT_CODE}" >&2

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "============================================================================" >&2
echo "Task ${SLURM_ARRAY_TASK_ID} Completed" >&2
echo "Total Duration (seconds): ${DURATION}" >&2
echo "Python Exit Code: $PYTHON_EXIT_CODE" >&2
echo "============================================================================" >&2

# Exit with Python's exit code
exit $PYTHON_EXIT_CODE