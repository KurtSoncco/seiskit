#!/bin/bash
#SBATCH --job-name=em8100_phase2
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3_htc
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/array_job_%A_task_%a.out
#SBATCH --error=logs/array_job_%A_task_%a.err

# Phase 2: 12 sims (remainder after Phase 1). Array 0-11 → 0-based indices 8088-8099.
# One sim per array task. Submit after Phase 1 (or anytime; idempotent unless FORCE_RERUN=1).
#
# Set FORCE_RERUN=1 to re-run even when output exists (passes --force to run_experiment.py).
# Example: FORCE_RERUN=1 sbatch phase2_htc.sh
FORCE_RERUN=${FORCE_RERUN:-0}
mkdir -p logs
set -euo pipefail

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-} Task=${SLURM_ARRAY_TASK_ID:-} Host=$(hostname)" >&2

# Load only if venv needs system BLAS; omit if wheels provide it
if [ -n "${SLURM_JOB_ID:-}" ]; then
    module purge
    # module load gcc/13.2.0 openblas/0.3.24
fi

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
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Diagnostic: CPU model, MHz, NUMA.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "$(date -Is) | CPU/NUMA | lscpu (model, MHz):" >&2
    lscpu 2>/dev/null | egrep 'Model name|MHz' || true
    echo "$(date -Is) | CPU/NUMA | numactl --hardware:" >&2
    numactl --hardware 2>/dev/null || true
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# H5 output to scratch (same as phase1; avoids 50 GB home quota).
export EMULATOR_8100_H5_DIR=/global/scratch/users/$USER/emulator_second_h5
export EMULATOR_8100_H5_LOSSY=1
export EMULATOR_8100_H5_DOWNSAMPLE=2

PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/run_experiment.py"

# Array 0-11 → 0-based index 8088..8099 (run_experiment.py uses 0-based)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
INDEX=$((8088 + TASK_ID))
EXTRA_ARGS=""
[ "${FORCE_RERUN}" = "1" ] && EXTRA_ARGS="--force"

# srun binds the single core on HTC
srun -c 1 "${PYTHON_BIN}" -u "${RUNNER_PY}" --index "${INDEX}" ${EXTRA_ARGS}
