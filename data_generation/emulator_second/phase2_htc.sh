#!/bin/bash
#SBATCH --job-name=em8100_phase2_second
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio2_htc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/array_job_%A_task_%a.out
#SBATCH --error=logs/array_job_%A_task_%a.err

# Phase 2: 12 sims (remainder after Phase 1). Array 0-11 → 0-based indices 8088-8099.
# One sim per array task. Submit after Phase 1 (or anytime; idempotent unless FORCE_RERUN=1).
#
# Set FORCE_RERUN=1 to re-run even when output exists (passes --force to run_experiment.py).
# Example: FORCE_RERUN=1 sbatch phase2_htc.sh
FORCE_RERUN=${FORCE_RERUN:-0}
[ -d logs ] || mkdir -p logs
set -euo pipefail
trap 'sacct -X -j "${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" -P --format=JobID,State,ExitCode,MaxRSS,AveCPU 2>/dev/null || true' EXIT

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-} Task=${SLURM_ARRAY_TASK_ID:-} Host=$(hostname)" >&2

# Run from venv; skip module purge to avoid srun/sacct issues on some systems.
# Load gcc/openblas only if OpenSees reports missing shared libs.
source /global/home/users/kurtwal98/seiskit/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

# Only set if OpenSees/python needs this shared lib path
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}
fi

# Pin threads/placement (1 CPU per task; OpenMP/libs may spawn threads otherwise).
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Diagnostic: CPU model, MHz, NUMA.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "$(date -Is) | CPU/NUMA | lscpu (model, MHz):" >&2
    lscpu 2>/dev/null | egrep 'Model name|MHz' || true
    echo "$(date -Is) | CPU/NUMA | numactl --hardware:" >&2
    numactl --hardware 2>/dev/null || true
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# H5 output to scratch (same as phase1; avoids 50 GB home quota).
export EMULATOR_8100_H5_DIR=/global/scratch/users/$USER/emulator8100_h5_second
export EMULATOR_8100_H5_LOSSY=1
export EMULATOR_8100_H5_DOWNSAMPLE=2

PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/run_experiment.py"

command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python binary not found at ${PYTHON_BIN}" >&2; exit 2; }
test -r "${RUNNER_PY}" >/dev/null || { echo "ERROR: Runner script not readable at ${RUNNER_PY}" >&2; exit 2; }

# Warm venv site-packages stat cache to reduce first-import latency
python - <<'PYEOF' || true
import pkgutil
for m in ('openseespy','numpy','scipy'):
    try: pkgutil.find_loader(m)
    except Exception: pass
print('SITE_WARMED')
PYEOF

sleep $(( (RANDOM % 10) + ${SLURM_ARRAY_TASK_ID:-0} % 3 ))
echo "$(date -Is) | PREFLIGHT | Verifying Python and OpenSees (timeout=90s)..." >&2
timeout 90s "${PYTHON_BIN}" - <<'PYEOF'
import sys
print('PYTHON_OK', sys.version.split()[0], flush=True)
try:
    import openseespy.opensees as ops  # noqa
    print('OPENSEES_OK', flush=True)
except Exception as e:
    print(f'OPENSEES_IMPORT_FAIL: {e}', flush=True)
    raise
PYEOF
PRE_RC=$?
if [ ${PRE_RC} -ne 0 ]; then
  echo "$(date -Is) | ERROR | Preflight check failed (Code: ${PRE_RC}). Exiting." >&2
  exit ${PRE_RC}
fi
echo "$(date -Is) | PREFLIGHT | Preflight success." >&2

# Per-sim timeout (~1.9h) so a stuck run does not burn the 2h slot
SIM_TIMEOUT=7000

# Array 0-11 → 0-based index 8088..8099 (run_experiment.py uses 0-based)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
INDEX=$((8088 + TASK_ID))
EXTRA_ARGS=""
[ "${FORCE_RERUN}" = "1" ] && EXTRA_ARGS="--force"

# srun binds the single core on HTC; timeout fails stuck sims before wall time
exec srun --exclusive -c 1 --cpu-bind=cores timeout "$SIM_TIMEOUT" "${PYTHON_BIN}" -u "${RUNNER_PY}" --index "${INDEX}" ${EXTRA_ARGS}
