#!/bin/bash
# Single-node Savio job for Hallal vs neural-operator 2-layer comparison.
#
# Prefers local joblib; use this when Box I/O or Haskell ensembles are slow.
#
# Submit from repo root (or this directory):
#   mkdir -p neural-operator/data/hallal_vs_2d/logs
#   sbatch neural-operator/data/hallal_vs_2d/submit_savio.sh
#
# Smoke:
#   HALLAL_N_REAL=4 sbatch --export=ALL,HALLAL_SMOKE=1 ...
#
#SBATCH --job-name=hallal_vs_2d
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --constraint=savio2_c24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hallal_vs_2d_%j.out
#SBATCH --error=logs/hallal_vs_2d_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HALLAL_N_JOBS="${HALLAL_N_JOBS:-${SLURM_CPUS_PER_TASK:-24}}"

# Box mount on Savio (override if your mount path differs)
export NO_BOX_ROOT="${NO_BOX_ROOT:-/global/home/users/${USER}/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data}"

VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-local} Host=$(hostname) n_jobs=${HALLAL_N_JOBS}"
echo "REPO=${REPO_ROOT}"
echo "NO_BOX_ROOT=${NO_BOX_ROOT}"

cd "${SCRIPT_DIR}"
EXTRA=()
if [[ "${HALLAL_SMOKE:-0}" == "1" ]]; then
  EXTRA+=(--smoke)
fi
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  EXTRA+=(--force)
fi

"${PYTHON_BIN}" -u run_comparison.py --n-jobs "${HALLAL_N_JOBS}" "${EXTRA[@]}"
echo "$(date -Is) | DONE"
