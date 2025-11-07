#!/bin/bash
#SBATCH --job-name=parametric-study
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%10
# Allow scheduler to pack tasks per node; no explicit ntasks-per-node or exclusive
#SBATCH --output=logs/array_job_%A_task_%a.out
#SBATCH --error=logs/array_job_%A_task_%a.err

set -x
set -euo pipefail

# Stagger module loading to reduce Lmod contention (random delay 0-2s)
# This prevents all tasks from hitting the module system simultaneously
sleep $((RANDOM % 3))

# Clean module env and load required toolchain
module purge
module load gcc/13.2.0
module load openblas/0.3.24 

# Force single-threaded math libs to avoid oversubscription and races
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

# Activate your project venv (absolute path for HPC home)
source /global/home/users/kurtwal98/seiskit/.venv/bin/activate

# Prevent Python bytecode (.pyc) file writing to avoid NFS contention
# When multiple tasks import openseespy simultaneously, they fight for file locks
export PYTHONDONTWRITEBYTECODE=1

# Make OpenSeesPy's native libs visible
export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}

# Run from the directory you submitted the job (keeps relative paths sane)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Record start time
START_TIME=$(date +%s)
START_DATE=$(date)

# Use explicit venv python to avoid PATH/env issues
PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"

# Print minimal job info
echo "============================================================================"
echo "Task ${SLURM_ARRAY_TASK_ID} | Job ${SLURM_JOB_ID} | Node: $SLURMD_NODENAME"
echo "Start: $START_DATE"
echo "============================================================================"
echo ""

# Quick preflight: verify Python env and OpenSees availability (fail fast if broken)
echo "[PRE] $(date) - Verifying Python and OpenSees imports..."
timeout 30s ${PYTHON_BIN} - <<'PYEOF'
import sys
print('PYTHON_OK', sys.version.split()[0])
try:
    import openseespy.opensees as ops  # noqa
    print('OPENSEES_OK')
except Exception as e:
    print('OPENSEES_IMPORT_FAIL', e)
    sys.exit(3)
PYEOF
PRE_RC=$?
echo "[PRE] $(date) - Preflight exit code: ${PRE_RC}"
if [ ${PRE_RC} -ne 0 ]; then
  echo "[PRE] Preflight failed; exiting task ${SLURM_ARRAY_TASK_ID}"
  echo "[PRE] Check error logs for details: logs/array_job_${SLURM_JOB_ID:-<job_id>}_task_${SLURM_ARRAY_TASK_ID}.err"
  exit ${PRE_RC}
fi

# Execute the array task via srun with CPU binding; the script reads SLURM_ARRAY_TASK_ID automatically
echo "[RUN] $(date) - Launching srun for task ${SLURM_ARRAY_TASK_ID}"

# Removed heartbeat to avoid pre-step cgroup/binding interference and misleading liveness

# Set timeout to 1.75 hours (6300s) - less than job walltime of 2 hours
# This ensures SLURM kills the job before srun timeout, preventing hangs
PER_TASK_TIMEOUT_SECONDS="${PER_TASK_TIMEOUT_SECONDS:-6300}"

# Use node-local scratch space for TMPDIR to avoid NFS contention
# $SLURM_TMP is fast, node-local storage that's automatically cleaned up
if [ -n "$SLURM_TMP" ]; then
    # Running under SLURM - use node-local scratch
    export TMPDIR="${SLURM_TMP}/task_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "${TMPDIR}"
else
    # Running locally - use a local temp directory
    export TMPDIR="${SLURM_SUBMIT_DIR:-$PWD}/.tmp/task_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "${TMPDIR}"
fi

# Resolve absolute path to the runner within the SLURM submit directory
# We already cd'ed into ${SLURM_SUBMIT_DIR} above, but use absolute path to avoid spool dir confusion
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/run_experiment.py"

# Add --mpi=none to avoid PMI initialization delays/hangs
# Note: Job no longer uses --exclusive, so multiple tasks can share nodes efficiently
srun --export=ALL \
     --ntasks=1 \
     --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     --cpu-bind=cores \
     --mpi=none \
     --kill-on-bad-exit=1 \
     timeout "${PER_TASK_TIMEOUT_SECONDS}"s \
     ${PYTHON_BIN} -u "${RUNNER_PY}" --index "${SLURM_ARRAY_TASK_ID}"
PYTHON_EXIT_CODE=$?
echo "[RUN] $(date) - Python exit code: ${PYTHON_EXIT_CODE}"

# (no heartbeat to stop)

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
