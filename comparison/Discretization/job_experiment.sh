#!/bin/bash
#SBATCH --job-name=disc_exp
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=20:00:00
#SBATCH --array=0-9%5
# Allow scheduler to pack tasks per node; no explicit ntasks-per-node or exclusive
#SBATCH --output=logs/disc_exp_%A_%a.out
#SBATCH --error=logs/disc_exp_%A_%a.err

set -x
set -euo pipefail

# Get case type from command line argument (default to 2x2_4node)
CASE_TYPE=${1:-2x2_4node}

# Validate case type
if [[ ! "$CASE_TYPE" =~ ^(2x2_4node|1x1_4node|2x2_8node)$ ]]; then
    echo "Error: Invalid case type '$CASE_TYPE'. Must be one of: 2x2_4node, 1x1_4node, 2x2_8node"
    exit 1
fi

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

# Get the array task ID
TASK_ID=$SLURM_ARRAY_TASK_ID

# Print minimal job info
echo "============================================================================"
echo "Task ${TASK_ID} | Job ${SLURM_JOB_ID} | Node: $SLURMD_NODENAME"
echo "Case: $CASE_TYPE"
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
  echo "[PRE] Preflight failed; exiting task ${TASK_ID}"
  exit ${PRE_RC}
fi

# Give each array task an isolated temp directory to avoid shared-TMP contention
# Use retry logic to handle concurrent creation
export TMPDIR="${SLURM_SUBMIT_DIR:-$PWD}/.tmp/task_${TASK_ID}"
for i in 0.1 0.2 0.3 0.4 0.5; do
  mkdir -p "${TMPDIR}" && break || sleep "$i"
done

# Resolve absolute path to the runner within the SLURM submit directory
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/run_experiment.py"

# Safety timeout slightly below SLURM limit (seconds); override with PER_TASK_TIMEOUT_SECONDS
PER_TASK_TIMEOUT_SECONDS="${PER_TASK_TIMEOUT_SECONDS:-70000}"

# Execute the array task via srun with CPU binding
echo "[RUN] $(date) - Launching srun for task ${TASK_ID}, case ${CASE_TYPE}"

# Add --mpi=none to avoid PMI initialization delays/hangs
srun --export=ALL \
     --ntasks=1 \
     --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     --cpu-bind=cores \
     --mpi=none \
     --kill-on-bad-exit=1 \
     timeout "${PER_TASK_TIMEOUT_SECONDS}"s \
     ${PYTHON_BIN} -u "${RUNNER_PY}" --case "${CASE_TYPE}" --index "${TASK_ID}"
PYTHON_EXIT_CODE=$?
echo "[RUN] $(date) - Python exit code: ${PYTHON_EXIT_CODE}"

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
echo "Task ID: $TASK_ID"
echo "Case: $CASE_TYPE"
echo "Start Time: $START_DATE"
echo "End Time: $END_DATE"
echo "Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Total Duration (seconds): ${DURATION}"
echo "Python Exit Code: $PYTHON_EXIT_CODE"
echo "============================================================================"

# Exit with Python's exit code
exit $PYTHON_EXIT_CODE

