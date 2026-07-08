#!/bin/bash
# Single-index Savio3 job (Parametric_study pattern). For production use phase1_savio2.sh.
#SBATCH --job-name=rv-single
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0
#SBATCH --output=logs/array_job_%A_task_%a.out
#SBATCH --error=logs/array_job_%A_task_%a.err
#SBATCH --exclude=n0029.savio3

set -euo pipefail

echo "$(date -Is) | START | Job=${SLURM_JOB_ID} Task=${SLURM_ARRAY_TASK_ID} Host=$(hostname)" >&2

echo "$(date -Is) | MODULE | Purging modules..." >&2
module purge
echo "$(date -Is) | MODULE | Loading gcc/13.2.0 openblas/0.3.24..." >&2
module load gcc/13.2.0 openblas/0.3.24
echo "$(date -Is) | MODULE | Module load complete." >&2

export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

source /global/home/users/kurtwal98/seiskit/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/run_experiment.py"
PER_TASK_TIMEOUT_SECONDS="${PER_TASK_TIMEOUT_SECONDS:-6300}"

command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python binary not found at ${PYTHON_BIN}"; exit 2; }
test -r "${RUNNER_PY}" >/dev/null || { echo "ERROR: Runner script not readable at ${RUNNER_PY}"; exit 2; }

export TMPDIR="/global/scratch/users/$USER/tmp/job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

export RV_OUTDIR=/global/scratch/users/$USER/rv_comparison/opensees_runs/${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}
export RV_H5_DIR=/global/scratch/users/$USER/rv_comparison/h5
mkdir -p "$RV_OUTDIR" "$RV_H5_DIR"

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

START_TIME=$(date +%s)
echo "$(date -Is) | RUN | Launching index ${SLURM_ARRAY_TASK_ID}..." >&2

timeout "${PER_TASK_TIMEOUT_SECONDS}"s \
    ${PYTHON_BIN} -u "${RUNNER_PY}" --index "${SLURM_ARRAY_TASK_ID}"

PYTHON_EXIT_CODE=$?
END_TIME=$(date +%s)
echo "$(date -Is) | RUN | Finished. Duration=$((END_TIME - START_TIME))s exit=${PYTHON_EXIT_CODE}" >&2
exit $PYTHON_EXIT_CODE
