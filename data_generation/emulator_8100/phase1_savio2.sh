#!/bin/bash
#SBATCH --job-name=em8100_phase1
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --constraint=savio2_c24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --array=0-336
#SBATCH --output=logs/array_job_%A_task_%a.out
#SBATCH --error=logs/array_job_%A_task_%a.err

# Phase 1: 8,088 sims (indices 0-8087). Savio2 is whole-node; use all 24 cores
# by running 24 concurrent sims per array element via GNU Parallel.
# 337 array elements × 24 sims = 8088. Submit: sbatch phase1_savio2.sh
#
# Set FORCE_RERUN=1 to re-run even when output exists (passes --force to run_experiment.py).
FORCE_RERUN=${FORCE_RERUN:-0}
mkdir -p logs
set -euo pipefail

# Avoid oversubscribing BLAS/OpenMP; each sim is single-threaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Task=${SLURM_ARRAY_TASK_ID:-<local>} Host=$(hostname)" >&2

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

# Chunking: 24 sims per array element (0-based indices). Last chunk clamped to TOTAL.
CHUNK=24
TOTAL=8088
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START=$((TASK_ID * CHUNK))
END=$((START + CHUNK))
if [ "${END}" -gt "${TOTAL}" ]; then
  END=${TOTAL}
fi
COUNT=$((END - START))
EXTRA_ARGS=""
[ "${FORCE_RERUN}" = "1" ] && EXTRA_ARGS="--force"
export PYTHON_BIN RUNNER_PY EXTRA_ARGS

# Scratch-based paths: avoid $HOME for GNU Parallel and temp files (hardening).
export PARALLEL_HOME=/global/scratch/users/$USER/.parallel
export TMPDIR=/global/scratch/users/$USER/tmp/job_${SLURM_JOB_ID:-0}_task_${TASK_ID}_root
RESULTS_DIR="logs/per_idx/${SLURM_JOB_ID:-0}/${TASK_ID}"
mkdir -p "$PARALLEL_HOME" "$TMPDIR" "$RESULTS_DIR"

# Per-sim timeout (4h) so one stuck run doesn't hang the slot; job time is 5h.
SIM_TIMEOUT=14400
export SIM_TIMEOUT

echo "$(date -Is) | RUN | Task ${TASK_ID}: indices ${START}..$((END-1)) (${COUNT} sims) via parallel -j 24 (timeout ${SIM_TIMEOUT}s each)..." >&2

seq ${START} $((END - 1)) | parallel -j 24 \
  --joblog logs/joblog_task_${TASK_ID}.txt \
  --results "$RESULTS_DIR" \
  --tag \
  'idx={}; idx_tmp="$TMPDIR/idx_$idx"; mkdir -p "$idx_tmp"; export TMPDIR="$idx_tmp"; trap "rm -rf \"$idx_tmp\"" EXIT; timeout "$SIM_TIMEOUT" "$PYTHON_BIN" -u "$RUNNER_PY" --index "$idx" $EXTRA_ARGS'

PARALLEL_RC=$?
echo "$(date -Is) | RUN | parallel finished. Exit code: ${PARALLEL_RC}" >&2

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "$(date -Is) | ACCOUNTING | Job RSS and CPU efficiency:" >&2
    sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,MaxVMSize,AveCPU,Elapsed,State >&2
fi

exit $PARALLEL_RC
