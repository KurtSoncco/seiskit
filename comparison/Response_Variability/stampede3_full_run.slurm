#!/bin/bash
# Full Response_Variability campaign on TACC Stampede3 (pylauncher).
#
# Default split (recommended): Hallal 1D + Pretell OpenSees only — skips grf_2d
# (run the GIFNO surrogate locally). Clear skip with RV_SKIP_METHODS= for all indices.
#
# Submit:
#   ./submit_stampede3_opensees.sh
#   # or:
#   sbatch comparison/Response_Variability/stampede3_full_run.slurm
#
# Full OpenSees production (Hallal + Pretell):
#   FORCE_RERUN=1 sbatch -N 8 --ntasks-per-node=48 -t 48:00:00 \
#     comparison/Response_Variability/stampede3_full_run.slurm
#
# Include grf_2d too (needs torch + GIFNO checkpoint on Stampede):
#   RV_SKIP_METHODS= FORCE_RERUN=1 sbatch ... stampede3_full_run.slurm
#
# Smoke Hallal+Pretell:
#   RV_SMOKE=1 RV_SMOKE_2D=1 FORCE_RERUN=1 sbatch -N 2 --ntasks-per-node=16 -t 12:00:00 \
#     comparison/Response_Variability/stampede3_full_run.slurm
#
# Hallal 1D only:
#   RV_SKIP_METHODS=grf_2d,pretell sbatch ... stampede3_full_run.slurm
#SBATCH -J rv_opensees
#SBATCH -o rv_stampede3_full.o%j
#SBATCH -e rv_stampede3_full.e%j
#SBATCH -p skx
#SBATCH -N 8
#SBATCH --ntasks-per-node=48
#SBATCH -t 48:00:00
#SBATCH -A ECS24003

set -euo pipefail

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Host=$(hostname)" >&2

module reset
module load pylauncher

PROJECT_ROOT="${PROJECT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/comparison/Response_Variability}"
REPO_ROOT="${PROJECT_ROOT}"

RUNNER_PY="${RUNNER_PY:-${SCRIPT_DIR}/run_experiment.py}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

export RV_SMOKE="${RV_SMOKE:-0}"
export RV_SMOKE_2D="${RV_SMOKE_2D:-1}"
export RV_REGEN_SOBOL="${RV_REGEN_SOBOL:-0}"
# No surrogate on Stampede by default (grf_2d skipped; Pretell uses OpenSees).
export RV_USE_SURROGATE_2D="${RV_USE_SURROGATE_2D:-0}"
# Skip GIFNO arm unless explicitly cleared: RV_SKIP_METHODS=
export RV_SKIP_METHODS="${RV_SKIP_METHODS-grf_2d}"
FORCE_RERUN="${FORCE_RERUN:-0}"

RUN_BASE="${RUN_BASE:-${SCRATCH:-${REPO_ROOT}}/rv_comparison}"
COMMAND_FILE="${COMMAND_FILE:-${RUN_BASE}/commands_${SLURM_JOB_ID:-manual}.txt}"
LAUNCHER_WORKDIR="${LAUNCHER_WORKDIR:-${RUN_BASE}/pylauncher_${SLURM_JOB_ID:-manual}}"
PER_TASK_LOG_DIR="${PER_TASK_LOG_DIR:-${RUN_BASE}/per_index_logs}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER}/rv_${SLURM_JOB_ID:-manual}}"

export RV_OUTDIR="${RV_OUTDIR:-${RUN_BASE}/opensees_runs}"
export RV_H5_DIR="${RV_H5_DIR:-${RUN_BASE}/h5}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_BASE}" "${RV_OUTDIR}" "${RV_H5_DIR}" "${PER_TASK_LOG_DIR}" "${TMP_ROOT}"

if [ ! -d "${SCRIPT_DIR}" ]; then
  echo "ERROR: script directory not found at ${SCRIPT_DIR}" >&2
  exit 2
fi
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv not found at ${VENV_PATH}/bin/activate" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python not found at ${PYTHON_BIN}" >&2; exit 2; }
test -r "${RUNNER_PY}" || { echo "ERROR: Runner not readable at ${RUNNER_PY}" >&2; exit 2; }

OPENSEES_LIB_DIR="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import site

for base in list(site.getsitepackages()) + ([site.getusersitepackages()] if hasattr(site, "getusersitepackages") else []):
    candidate = Path(base) / "openseespylinux" / "lib"
    if candidate.exists():
        print(candidate)
        break
PY
)"
if [ -n "${OPENSEES_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="${OPENSEES_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

echo "$(date -Is) | PREFLIGHT | Python + pylauncher + OpenSees..." >&2
timeout 60s "${PYTHON_BIN}" - <<'PY'
import pylauncher  # noqa: F401
import sys

print("PYTHON_OK", sys.version.split()[0])
import openseespy.opensees as ops  # noqa: F401
print("OPENSEES_OK")
PY

cd "${SCRIPT_DIR}"
TOTAL_TASKS="$("${PYTHON_BIN}" -c "from manifest import total_combinations; print(total_combinations())")"
INDEX_OFFSET="${RV_INDEX_OFFSET:-0}"
INDEX_MAX="${RV_INDEX_MAX:-${TOTAL_TASKS}}"
echo "$(date -Is) | CONFIG | RV_SMOKE=${RV_SMOKE} SMOKE_2D=${RV_SMOKE_2D} SKIP=${RV_SKIP_METHODS:-<none>} OFFSET=${INDEX_OFFSET} MAX=${INDEX_MAX} TOTAL=${TOTAL_TASKS}" >&2

export COMMAND_FILE LAUNCHER_WORKDIR PER_TASK_LOG_DIR TMP_ROOT
export RUNNER_PY PYTHON_BIN FORCE_RERUN INDEX_OFFSET INDEX_MAX SCRIPT_DIR
export RV_SKIP_METHODS

N_CMDS="$("${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
from manifest import index_to_params, parse_method_list

command_file = Path(os.environ["COMMAND_FILE"])
command_file.parent.mkdir(parents=True, exist_ok=True)
log_dir = Path(os.environ["PER_TASK_LOG_DIR"])
log_dir.mkdir(parents=True, exist_ok=True)
tmp_root = Path(os.environ["TMP_ROOT"])
tmp_root.mkdir(parents=True, exist_ok=True)

python_bin = os.environ["PYTHON_BIN"]
runner_py = os.environ["RUNNER_PY"]
script_dir = os.environ["SCRIPT_DIR"]
index_offset = int(os.environ["INDEX_OFFSET"])
index_max = int(os.environ["INDEX_MAX"])
force = os.environ["FORCE_RERUN"] == "1"
skip = parse_method_list(os.environ.get("RV_SKIP_METHODS")) or set()

n_written = 0
with command_file.open("w", encoding="utf-8") as handle:
    for idx in range(index_offset, index_max):
        method = index_to_params(idx).method
        if method in skip:
            continue
        idx_label = f"{idx:05d}"
        tmpdir = tmp_root / f"idx_{idx_label}"
        stdout_path = log_dir / f"idx_{idx_label}.out"
        stderr_path = log_dir / f"idx_{idx_label}.err"

        command = (
            f'cd "{script_dir}" && '
            f'mkdir -p "{tmpdir}" && '
            f'export TMPDIR="{tmpdir}" && '
            f'export RV_USE_SURROGATE_2D=0 && '
            f'"{python_bin}" -u "{runner_py}" --index {idx}'
        )
        if force:
            command += " --force"
        command += f' > "{stdout_path}" 2> "{stderr_path}"'
        handle.write(command + "\n")
        n_written += 1

print(n_written)
PY
)"

echo "$(date -Is) | COMMANDS | Wrote ${N_CMDS} tasks → ${COMMAND_FILE} (skipped methods: ${RV_SKIP_METHODS:-none})" >&2
if [ "${N_CMDS}" = "0" ]; then
  echo "ERROR: no launcher commands generated" >&2
  exit 3
fi

NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
TASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-48}}"
echo "$(date -Is) | LAUNCH | ${NODES} node(s) × ${TASKS_PER_NODE} tasks/node" >&2

rm -rf "${LAUNCHER_WORKDIR}"
mkdir -p "${LAUNCHER_WORKDIR}"
export TACC_PYLAUNCHER_WORKDIR="${LAUNCHER_WORKDIR}"

"${PYTHON_BIN}" - <<'PY'
import os
import pylauncher

pylauncher.ClassicLauncher(
    os.environ["COMMAND_FILE"],
    workdir=os.environ["LAUNCHER_WORKDIR"],
    debug="host+job",
)
PY

echo "$(date -Is) | DONE | Completed all launcher tasks" >&2
