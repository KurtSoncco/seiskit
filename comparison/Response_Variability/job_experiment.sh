#!/bin/bash
#SBATCH --job-name=rv-comparison
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-499%20
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/h5

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Optional: set on submit
# export RV_OUTDIR=/scratch/$USER/rv_comparison
# export RV_H5_DIR=/scratch/$USER/rv_comparison/h5

if [ -f "${HOME}/seiskit/.venv/bin/activate" ]; then
  source "${HOME}/seiskit/.venv/bin/activate"
elif [ -f "../../../.venv/bin/activate" ]; then
  source "../../../.venv/bin/activate"
fi

python run_experiment.py --index "${SLURM_ARRAY_TASK_ID}"
