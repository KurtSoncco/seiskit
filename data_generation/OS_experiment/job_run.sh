#!/bin/bash
#SBATCH --job-name=OS_Pilot
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=05:00:00
#SBATCH --qos=savio_normal
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --mem=60G

# Run 32 tasks in parallel on 1 node via gnu-parallel.
# Submit: sbatch job_run.sh

set -euo pipefail
mkdir -p logs

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Host=$(hostname)" >&2

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "$(date -Is) | MODULE | Purging modules..." >&2
    module purge
    echo "$(date -Is) | MODULE | Loading gnu-parallel, gcc, openblas..." >&2
    module load parallel
    module load gcc/13.2.0 openblas/0.3.24
    echo "$(date -Is) | MODULE | Module load complete." >&2
fi

source /global/home/users/kurtwal98/seiskit/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

if [ -n "${SLURM_JOB_ID:-}" ]; then
    export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"


PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/run_experiment.py"

command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python binary not found at ${PYTHON_BIN}" >&2; exit 2; }
test -r "${RUNNER_PY}" >/dev/null || { echo "ERROR: Runner script not readable at ${RUNNER_PY}" >&2; exit 2; }

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

export PYTHON_BIN RUNNER_PY
echo "$(date -Is) | RUN | Launching 32 tasks via gnu-parallel..." >&2

seq 0 31 | parallel -j 32 --joblog logs/joblog.txt --tag \
  'idx={}; export TMPDIR=/global/scratch/users/$USER/tmp/job_${SLURM_JOB_ID}_task_$idx; mkdir -p "$TMPDIR"; trap "rm -rf \"$TMPDIR\"" EXIT; "$PYTHON_BIN" -u "$RUNNER_PY" --index "$idx"'

PARALLEL_RC=$?
echo "$(date -Is) | RUN | parallel finished. Exit code: ${PARALLEL_RC}" >&2

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "$(date -Is) | ACCOUNTING | Job RSS and CPU efficiency:" >&2
    sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,MaxVMSize,AveCPU,Elapsed,State >&2
fi

exit $PARALLEL_RC
