#!/bin/bash
# Run one three-layer experiment index on TACC Stampede3.
#
#   INDEX=0 sbatch neural-operator/experiments/three_layer/stampede3_single_index.sh
#SBATCH -J three_layer_idx
#SBATCH -o stampede3_three_layer_idx.o%j
#SBATCH -e stampede3_three_layer_idx.e%j
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH -A ECS24003

set -euo pipefail

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Host=$(hostname)" >&2

PROJECT_ROOT="${PROJECT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/neural-operator/experiments/three_layer}"
RUNNER_PY="${RUNNER_PY:-${SCRIPT_DIR}/run_experiment.py}"
MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/manifest.csv}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

INDEX="${INDEX:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_BASE="${RUN_BASE:-${SCRATCH:-${PROJECT_ROOT}}/opensees_three_layer_test}"

export EXP_OUTDIR="${EXP_OUTDIR:-${RUN_BASE}/raw_runs}"
export EXP_H5_DIR="${EXP_H5_DIR:-${RUN_BASE}/h5}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_BASE}" "${EXP_OUTDIR}" "${EXP_H5_DIR}"

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

OPENSEES_LIB_DIR="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import site
for base in list(site.getsitepackages()) + [site.getusersitepackages()]:
    candidate = Path(base) / "openseespylinux" / "lib"
    if candidate.exists():
        print(candidate)
        break
PY
)"
if [ -n "${OPENSEES_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="${OPENSEES_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

echo "$(date -Is) | RUN | INDEX=${INDEX} FORCE_RERUN=${FORCE_RERUN}" >&2

CMD=(
  "${PYTHON_BIN}" -u "${RUNNER_PY}"
  --manifest-path "${MANIFEST_PATH}"
  --index "${INDEX}"
)
if [ "${FORCE_RERUN}" = "1" ]; then
  CMD+=(--force)
fi

"${CMD[@]}"
echo "$(date -Is) | DONE | Index ${INDEX} finished" >&2
