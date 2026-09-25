#!/bin/bash
# Resume Pretell Campbell Q–Vs ensembles on Savio (remaining indices only).
#
# From this directory on Savio:
#   mkdir -p logs
#   sbatch submit_pretell_savio.sh
#
#SBATCH --job-name=pretell_vs_2d
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --constraint=savio2_c24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pretell_vs_2d_%j.out
#SBATCH --error=logs/pretell_vs_2d_%j.err

set -euo pipefail

REPO=/global/home/users/kurtwal98/seiskit
SCRIPT_DIR=$REPO/neural-operator/data/hallal_vs_2d
SCRATCH=/global/scratch/users/kurtwal98/neural_operator_data

export NO_BOX_ROOT=$SCRATCH
# Checkpoint on scratch (same publish dir) so resume survives login-node issues
export PRETELL_CKPT_DIR=$SCRATCH/hallal_vs_2d
export HALLAL_N_JOBS=${HALLAL_N_JOBS:-${SLURM_CPUS_PER_TASK:-24}}
export PRETELL_N_SAMPLES=${PRETELL_N_SAMPLES:-200}
export PRETELL_BATCH=${PRETELL_BATCH:-128}
export HDF5_PLUGIN_PATH=${HDF5_PLUGIN_PATH:-$REPO/.venv/lib/python3.11/site-packages/hdf5plugin/plugins}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$REPO:${PYTHONPATH:-}

PYTHON=$REPO/.venv/bin/python
cd "$SCRIPT_DIR"
mkdir -p logs

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-} Host=$(hostname)"
echo "NO_BOX_ROOT=$NO_BOX_ROOT"
echo "PRETELL_CKPT_DIR=$PRETELL_CKPT_DIR"
echo "n_jobs=$HALLAL_N_JOBS  n_samples=$PRETELL_N_SAMPLES  batch=$PRETELL_BATCH"
$PYTHON - <<'PY'
import h5py, numpy as np
from pathlib import Path
p = Path("/global/scratch/users/kurtwal98/neural_operator_data/hallal_vs_2d/pretell_ensembles.h5")
if p.is_file():
    with h5py.File(p, "r") as f:
        v = f["valid"][:]
        print(f"checkpoint valid={int(v.sum())}/{len(v)}")
else:
    print("WARNING: no pretell_ensembles.h5 checkpoint — will start from zero")
PY

# Resume by default (no --force). Set FORCE_RERUN=1 to wipe.
EXTRA=()
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  EXTRA+=(--force)
fi

$PYTHON -u add_pretell.py --n-jobs "$HALLAL_N_JOBS" "${EXTRA[@]}"
echo "$(date -Is) | DONE"
ls -lah "$SCRATCH/hallal_vs_2d/pretell_ensembles.h5"
