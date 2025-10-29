#!/bin/bash
#SBATCH --job-name=parametric-study
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --array=0-44%30
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

# Use explicit venv python to avoid PATH/env issues
PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"

# Print job info
echo "============================================================================"
echo "SLURM Array Job Information"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $START_DATE"
echo "Node: $SLURMD_NODENAME"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Working Directory: $(pwd)"
echo "Host: $(hostname)"
echo "Modules:"
module list 2>&1 | sed 's/^/  /'
echo "Python path: ${PYTHON_BIN}"
${PYTHON_BIN} -V
echo "NumPy version: $(${PYTHON_BIN} - <<'PYEOF'
import numpy as np
print(np.__version__)
PYEOF
)"
echo "============================================================================"
echo ""

# Quick preflight: verify Python env and OpenSees availability (fail fast if broken)
echo "[PRE] $(date) - Verifying Python and OpenSees imports..."
timeout 120s ${PYTHON_BIN} - <<'PYEOF'
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
  exit ${PRE_RC}
fi

# Execute the array task via srun with CPU binding; the script reads SLURM_ARRAY_TASK_ID automatically
echo "[RUN] $(date) - Launching srun for task ${SLURM_ARRAY_TASK_ID}"

# Heartbeat every 60s to show liveness
(
  while true; do echo "[HEARTBEAT] $(date) - task ${SLURM_ARRAY_TASK_ID} running"; sleep 60; done
) &
HB_PID=$!

# Safety timeout slightly below SLURM limit (seconds); override with PER_TASK_TIMEOUT_SECONDS
PER_TASK_TIMEOUT_SECONDS="${PER_TASK_TIMEOUT_SECONDS:-6900}"

srun --export=ALL \
     --ntasks=1 \
     --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     --cpu-bind=cores \
     --kill-on-bad-exit=1 \
     timeout "${PER_TASK_TIMEOUT_SECONDS}"s \
     ${PYTHON_BIN} -u run_experiment.py
SRUN_RC=$?
echo "[RUN] $(date) - srun exit code: ${SRUN_RC}"
PYTHON_EXIT_CODE=$?

# Stop heartbeat
kill ${HB_PID} 2>/dev/null || true

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
