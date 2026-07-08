#!/bin/bash
#SBATCH --job-name=rv_phase2
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3_htc
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --array=0
#SBATCH --output=logs/array_job_%A_task_%a.out
#SBATCH --error=logs/array_job_%A_task_%a.err

# Phase 2: single-index reruns on HTC (failed indices from phase1 joblog).
# Array task ID = experiment index (override with sbatch --array=12,45,99).
#
# FORCE_RERUN=1 to re-run even when HDF5 exists.
# Example: FORCE_RERUN=1 sbatch --array=0,10,20 phase2_htc.sh
FORCE_RERUN=${FORCE_RERUN:-0}
mkdir -p logs
set -euo pipefail

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-} Task=${SLURM_ARRAY_TASK_ID:-} Host=$(hostname)" >&2

if [ -n "${SLURM_JOB_ID:-}" ]; then
    module purge
fi

source /global/home/users/kurtwal98/seiskit/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

if [ -n "${SLURM_JOB_ID:-}" ]; then
    export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}
fi

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export RV_OUTDIR=/global/scratch/users/$USER/rv_comparison/opensees_runs/htc_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}
export RV_H5_DIR=/global/scratch/users/$USER/rv_comparison/h5
mkdir -p "$RV_OUTDIR" "$RV_H5_DIR"

PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/run_experiment.py"

INDEX=${SLURM_ARRAY_TASK_ID:-0}
EXTRA_ARGS=""
[ "${FORCE_RERUN}" = "1" ] && EXTRA_ARGS="--force"

export TMPDIR="/global/scratch/users/$USER/tmp/htc_${SLURM_JOB_ID:-0}_${INDEX}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "$(date -Is) | RUN | index=${INDEX} RV_H5_DIR=${RV_H5_DIR}" >&2
srun -c 1 "${PYTHON_BIN}" -u "${RUNNER_PY}" --index "${INDEX}" ${EXTRA_ARGS}
