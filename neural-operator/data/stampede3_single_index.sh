#!/bin/bash
# Run one Sobol index on TACC Stampede3.
#
# Default:
#   sbatch stampede3_single_index.sh
#
# Common overrides:
#   INDEX=0 sbatch stampede3_single_index.sh
#   INDEX=17 FORCE_RERUN=1 sbatch stampede3_single_index.sh
#   INDEX=0 RUN_BASE="$SCRATCH/opensees_sobol_full" sbatch stampede3_single_index.sh
#
# This script is intentionally separate from the full launcher so you can:
# 1. verify one index runs end-to-end,
# 2. inspect logs/HDF5/raw outputs,
# 3. iterate on how many independent runs a node can sustain.
#SBATCH -J sobol_idx
#SBATCH -o stampede3_sobol_idx.o%j
#SBATCH -e stampede3_sobol_idx.e%j
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ECS24003

set -euo pipefail

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Host=$(hostname)" >&2

PROJECT_ROOT="${PROJECT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/neural-operator/data}"
RUNNER_PY="${RUNNER_PY:-${SCRIPT_DIR}/run_experiment.py}"
MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/sobol_manifest.csv}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

INDEX="${INDEX:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_BASE="${RUN_BASE:-${SCRATCH:-${PROJECT_ROOT}}/opensees_sobol_test}"

export SOBOL_OUTDIR="${SOBOL_OUTDIR:-${RUN_BASE}/raw_runs}"
export SOBOL_H5_DIR="${SOBOL_H5_DIR:-${RUN_BASE}/h5}"
export SOBOL_TIMING_DB="${SOBOL_TIMING_DB:-${RUN_BASE}/sobol_timing.db}"
export SOBOL_H5_LOSSY="${SOBOL_H5_LOSSY:-1}"
export SOBOL_H5_DOWNSAMPLE="${SOBOL_H5_DOWNSAMPLE:-2}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_BASE}" "${SOBOL_OUTDIR}" "${SOBOL_H5_DIR}"

if [ ! -d "${SCRIPT_DIR}" ]; then
  echo "ERROR: script directory not found at ${SCRIPT_DIR}" >&2
  exit 2
fi
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv activation script not found at ${VENV_PATH}/bin/activate" >&2
  exit 2
fi
if [ ! -r "${RUNNER_PY}" ]; then
  echo "ERROR: runner script not readable at ${RUNNER_PY}" >&2
  exit 2
fi
if [ ! -r "${MANIFEST_PATH}" ]; then
  echo "ERROR: manifest not readable at ${MANIFEST_PATH}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

OPENSEES_LIB_DIR="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import site

search_paths = []
try:
    search_paths.extend(site.getsitepackages())
except Exception:
    pass
try:
    search_paths.append(site.getusersitepackages())
except Exception:
    pass

for base in search_paths:
    candidate = Path(base) / "openseespylinux" / "lib"
    if candidate.exists():
        print(candidate)
        break
PY
)"
if [ -n "${OPENSEES_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="${OPENSEES_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

echo "$(date -Is) | PATHS | PROJECT_ROOT=${PROJECT_ROOT}" >&2
echo "$(date -Is) | PATHS | SCRIPT_DIR=${SCRIPT_DIR}" >&2
echo "$(date -Is) | PATHS | RUNNER_PY=${RUNNER_PY}" >&2
echo "$(date -Is) | PATHS | MANIFEST_PATH=${MANIFEST_PATH}" >&2
echo "$(date -Is) | PATHS | PYTHON_BIN=${PYTHON_BIN}" >&2
echo "$(date -Is) | PATHS | RUN_BASE=${RUN_BASE}" >&2
echo "$(date -Is) | RUN | INDEX=${INDEX} FORCE_RERUN=${FORCE_RERUN}" >&2

echo "$(date -Is) | PREFLIGHT | Verifying Python and OpenSees..." >&2
timeout 60s "${PYTHON_BIN}" - <<'PY'
import sys

print("PYTHON_OK", sys.version.split()[0])
try:
    import openseespy.opensees as ops  # noqa: F401
    print("OPENSEES_OK")
except Exception as exc:
    print(f"OPENSEES_IMPORT_FAIL: {exc}")
    raise SystemExit(3)
PY

CMD=(
  "${PYTHON_BIN}" -u "${RUNNER_PY}"
  --manifest-path "${MANIFEST_PATH}"
  --index "${INDEX}"
)

if [ "${FORCE_RERUN}" = "1" ]; then
  CMD+=(--force)
fi

echo "$(date -Is) | EXEC | ${CMD[*]}" >&2
"${CMD[@]}"

echo "$(date -Is) | DONE | Index ${INDEX} finished" >&2
