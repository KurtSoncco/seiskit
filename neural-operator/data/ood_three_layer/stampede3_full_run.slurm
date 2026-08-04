#!/bin/bash
# Tier B ood_three_layer full run on TACC Stampede3 (pylauncher).
#
# Smoke (8 h — max Lz≈34 m; ~50 m cases take ~3–4 h):
#   OOD_PHYSICS_COUNT=2 OOD_SEED_LEVELS=2 OOD_OVERWRITE_MANIFEST=1 FORCE_RERUN=1 \
#     sbatch -N 1 --ntasks-per-node=4 -t 08:00:00 \
#     neural-operator/data/ood_three_layer/stampede3_full_run.slurm
#
# Production (960 tasks):
#   sbatch -N 4 --ntasks-per-node=48 -t 48:00:00 \
#     neural-operator/data/ood_three_layer/stampede3_full_run.slurm
#SBATCH -J ood_3layer
#SBATCH -o stampede3_ood_three_layer.o%j
#SBATCH -e stampede3_ood_three_layer.e%j
#SBATCH -p skx
#SBATCH -N 4
#SBATCH --ntasks-per-node=48
#SBATCH -t 48:00:00
#SBATCH -A ECS24003

set -euo pipefail

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Host=$(hostname)" >&2

module reset
module load pylauncher

PROJECT_ROOT="${PROJECT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/neural-operator/data/ood_three_layer}"
REPO_ROOT="${PROJECT_ROOT}"

RUNNER_PY="${RUNNER_PY:-${SCRIPT_DIR}/run_experiment.py}"
MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/manifest.csv}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

OOD_PHYSICS_COUNT="${OOD_PHYSICS_COUNT:-32}"
OOD_SEED_LEVELS="${OOD_SEED_LEVELS:-30}"
OOD_SAMPLER_SEED="${OOD_SAMPLER_SEED:-42}"
OOD_RF_SEED_SEED="${OOD_RF_SEED_SEED:-43}"
OOD_OVERWRITE_MANIFEST="${OOD_OVERWRITE_MANIFEST:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

RUN_BASE="${RUN_BASE:-${SCRATCH:-${REPO_ROOT}}/opensees_ood_three_layer}"
COMMAND_FILE="${COMMAND_FILE:-${RUN_BASE}/commands_${SLURM_JOB_ID:-manual}.txt}"
LAUNCHER_WORKDIR="${LAUNCHER_WORKDIR:-${RUN_BASE}/pylauncher_${SLURM_JOB_ID:-manual}}"
PER_TASK_LOG_DIR="${PER_TASK_LOG_DIR:-${RUN_BASE}/per_index_logs}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER}/ood_three_layer_${SLURM_JOB_ID:-manual}}"

export OOD_OUTDIR="${OOD_OUTDIR:-${RUN_BASE}/raw_runs}"
export OOD_H5_DIR="${OOD_H5_DIR:-${RUN_BASE}/h5}"
export EXP_OUTDIR="${OOD_OUTDIR}"
export EXP_H5_DIR="${OOD_H5_DIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_BASE}" "${OOD_OUTDIR}" "${OOD_H5_DIR}" "${PER_TASK_LOG_DIR}" "${TMP_ROOT}"

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

export SCRIPT_DIR MANIFEST_PATH OOD_PHYSICS_COUNT OOD_SEED_LEVELS
export OOD_SAMPLER_SEED OOD_RF_SEED_SEED OOD_OVERWRITE_MANIFEST

TOTAL_TASKS="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

script_dir = Path(os.environ["SCRIPT_DIR"])
spec = importlib.util.spec_from_file_location("ood_tl_manifest", script_dir / "manifest.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["ood_tl_manifest"] = mod
spec.loader.exec_module(mod)

manifest = mod.ensure_manifest(
    path=os.environ["MANIFEST_PATH"],
    physics_count=int(os.environ["OOD_PHYSICS_COUNT"]),
    seed_levels=int(os.environ["OOD_SEED_LEVELS"]),
    sampler_seed=int(os.environ["OOD_SAMPLER_SEED"]),
    rf_seed_seed=int(os.environ["OOD_RF_SEED_SEED"]),
    overwrite=os.environ.get("OOD_OVERWRITE_MANIFEST", "0") == "1",
)
print(len(manifest))
PY
)"

if [ "${TOTAL_TASKS}" -le 0 ]; then
  echo "ERROR: Manifest contains no tasks." >&2
  exit 3
fi

echo "$(date -Is) | MANIFEST | ${MANIFEST_PATH} -> ${TOTAL_TASKS} tasks" >&2
echo "$(date -Is) | OUTPUTS | RUN_BASE=${RUN_BASE} H5=${OOD_H5_DIR}" >&2

export COMMAND_FILE LAUNCHER_WORKDIR PER_TASK_LOG_DIR TMP_ROOT
export RUNNER_PY PYTHON_BIN FORCE_RERUN TOTAL_TASKS MANIFEST_PATH

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

command_file = Path(os.environ["COMMAND_FILE"])
command_file.parent.mkdir(parents=True, exist_ok=True)
log_dir = Path(os.environ["PER_TASK_LOG_DIR"])
tmp_root = Path(os.environ["TMP_ROOT"])
log_dir.mkdir(parents=True, exist_ok=True)
tmp_root.mkdir(parents=True, exist_ok=True)

python_bin = os.environ["PYTHON_BIN"]
runner_py = os.environ["RUNNER_PY"]
manifest_path = os.environ["MANIFEST_PATH"]
total_tasks = int(os.environ["TOTAL_TASKS"])
force = os.environ.get("FORCE_RERUN", "0") == "1"

with command_file.open("w", encoding="utf-8") as handle:
    for idx in range(total_tasks):
        idx_label = f"{idx:05d}"
        tmpdir = tmp_root / f"idx_{idx_label}"
        stdout_path = log_dir / f"idx_{idx_label}.out"
        stderr_path = log_dir / f"idx_{idx_label}.err"
        command = (
            f'mkdir -p "{tmpdir}" && '
            f'export TMPDIR="{tmpdir}" && '
            f'export OOD_OUTDIR="{os.environ["OOD_OUTDIR"]}" && '
            f'export OOD_H5_DIR="{os.environ["OOD_H5_DIR"]}" && '
            f'"{python_bin}" -u "{runner_py}" '
            f'--manifest-path "{manifest_path}" '
            f'--index {idx}'
        )
        if force:
            command += " --force"
        command += f' > "{stdout_path}" 2> "{stderr_path}"'
        handle.write(command + "\n")
PY

echo "$(date -Is) | COMMANDS | Wrote ${COMMAND_FILE}" >&2

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

echo "$(date -Is) | DONE | ood_three_layer launcher finished" >&2
