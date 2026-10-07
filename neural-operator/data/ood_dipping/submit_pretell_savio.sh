#!/bin/bash
# Pretell strip-column ensembles for ood_dipping on Savio.
#
# From this directory on Savio:
#   mkdir -p logs
#   sbatch submit_pretell_savio.sh
#
#SBATCH --job-name=ood_dip_pretell
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --constraint=savio2_c24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ood_dip_pretell_%j.out
#SBATCH --error=logs/ood_dip_pretell_%j.err

set -euo pipefail

REPO=/global/home/users/kurtwal98/seiskit
SCRIPT_DIR=$REPO/neural-operator/data/ood_dipping
SCRATCH=/global/scratch/users/kurtwal98/neural_operator_data

export NO_BOX_ROOT=$SCRATCH
export OOD_H5_DIR=$SCRATCH/ood_dipping/h5
export OOD_MANIFEST=$SCRATCH/ood_dipping/manifest.csv
export OOD_PRETELL_OUT=$SCRATCH/ood_dipping/pretell_comparison
export OOD_TF_CENTER=$SCRATCH/ood_dipping/toro_comparison/tf_2d_center.h5
export PRETELL_CKPT_DIR=$SCRATCH/ood_dipping/pretell_comparison
export HALLAL_N_JOBS=${HALLAL_N_JOBS:-${SLURM_CPUS_PER_TASK:-24}}
export PRETELL_N_SAMPLES=${PRETELL_N_SAMPLES:-200}
export PRETELL_BATCH=${PRETELL_BATCH:-64}
export HDF5_PLUGIN_PATH=${HDF5_PLUGIN_PATH:-$REPO/.venv/lib/python3.11/site-packages/hdf5plugin/plugins}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$REPO:${PYTHONPATH:-}

PYTHON=$REPO/.venv/bin/python
cd "$SCRIPT_DIR"
mkdir -p logs "$OOD_PRETELL_OUT"

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-} Host=$(hostname)"
echo "OOD_H5_DIR=$OOD_H5_DIR"
echo "OOD_TF_CENTER=$OOD_TF_CENTER"
echo "OOD_PRETELL_OUT=$OOD_PRETELL_OUT"
echo "n_jobs=$HALLAL_N_JOBS  n_samples=$PRETELL_N_SAMPLES  batch=$PRETELL_BATCH"

if [[ ! -f "$OOD_TF_CENTER" ]]; then
  echo "ERROR: missing $OOD_TF_CENTER — sync toro_comparison/tf_2d_center.h5 first"
  exit 1
fi
H5_N=$(ls -1 "$OOD_H5_DIR"/run_*.h5 2>/dev/null | wc -l)
echo "H5 count=$H5_N"
if [[ "$H5_N" -lt 900 ]]; then
  echo "ERROR: expected ~960 H5s under $OOD_H5_DIR"
  exit 1
fi

$PYTHON - <<'PY'
import h5py
from pathlib import Path
p = Path("/global/scratch/users/kurtwal98/neural_operator_data/ood_dipping/pretell_comparison/pretell_ensembles.h5")
if p.is_file():
    with h5py.File(p, "r") as f:
        v = f["valid"][:]
        print(f"checkpoint valid={int(v.sum())}/{len(v)}")
else:
    print("no pretell_ensembles.h5 yet — starting fresh")
PY

EXTRA=()
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  EXTRA+=(--force)
fi
if [[ "${OOD_SMOKE:-0}" == "1" ]]; then
  EXTRA+=(--smoke)
fi

$PYTHON -u add_pretell.py --n-jobs "$HALLAL_N_JOBS" "${EXTRA[@]}"
echo "$(date -Is) | DONE"
ls -lah "$OOD_PRETELL_OUT"
