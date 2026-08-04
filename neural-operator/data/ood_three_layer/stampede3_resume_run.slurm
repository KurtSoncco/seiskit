#!/bin/bash
# Resume ood_three_layer: only indices missing run_{idx}.h5.
#SBATCH -J ood_3layer_r
#SBATCH -o stampede3_ood_three_layer_resume.o%j
#SBATCH -e stampede3_ood_three_layer_resume.e%j
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
OOD_OVERWRITE_MANIFEST="${OOD_OVERWRITE_MANIFEST:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

RUN_BASE="${RUN_BASE:-${SCRATCH:-${REPO_ROOT}}/opensees_ood_three_layer}"
COMMAND_FILE="${COMMAND_FILE:-${RUN_BASE}/commands_resume_${SLURM_JOB_ID:-manual}.txt}"
LAUNCHER_WORKDIR="${LAUNCHER_WORKDIR:-${RUN_BASE}/pylauncher_resume_${SLURM_JOB_ID:-manual}}"
PER_TASK_LOG_DIR="${PER_TASK_LOG_DIR:-${RUN_BASE}/per_index_logs}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER}/ood_three_layer_resume_${SLURM_JOB_ID:-manual}}"

export OOD_OUTDIR="${OOD_OUTDIR:-${RUN_BASE}/raw_runs}"
export OOD_H5_DIR="${OOD_H5_DIR:-${RUN_BASE}/h5}"
export EXP_OUTDIR="${OOD_OUTDIR}"
export EXP_H5_DIR="${OOD_H5_DIR}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

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

export SCRIPT_DIR MANIFEST_PATH OOD_PHYSICS_COUNT OOD_SEED_LEVELS OOD_OVERWRITE_MANIFEST OOD_H5_DIR FORCE_RERUN

read -r TOTAL_TASKS PENDING_COUNT COMPLETE_COUNT <<< "$("${PYTHON_BIN}" - <<'PY'
import importlib.util, os, sys
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
    overwrite=os.environ.get("OOD_OVERWRITE_MANIFEST", "0") == "1",
)
h5_dir = Path(os.environ["OOD_H5_DIR"])
force = os.environ.get("FORCE_RERUN", "0") == "1"
pending = [i for i in range(len(manifest)) if force or not (h5_dir / f"run_{i}.h5").exists()]
print(len(manifest), len(pending), len(manifest) - len(pending))
PY
)"

if [ "${PENDING_COUNT}" -le 0 ]; then
  echo "$(date -Is) | DONE | No pending (complete=${COMPLETE_COUNT}/${TOTAL_TASKS})" >&2
  exit 0
fi

echo "$(date -Is) | RESUME | complete=${COMPLETE_COUNT}/${TOTAL_TASKS} pending=${PENDING_COUNT}" >&2

export COMMAND_FILE LAUNCHER_WORKDIR PER_TASK_LOG_DIR TMP_ROOT RUNNER_PY PYTHON_BIN MANIFEST_PATH

"${PYTHON_BIN}" - <<'PY'
import importlib.util, os, sys
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
    overwrite=os.environ.get("OOD_OVERWRITE_MANIFEST", "0") == "1",
)
h5_dir = Path(os.environ["OOD_H5_DIR"])
force = os.environ.get("FORCE_RERUN", "0") == "1"
pending = [i for i in range(len(manifest)) if force or not (h5_dir / f"run_{i}.h5").exists()]
command_file = Path(os.environ["COMMAND_FILE"])
log_dir = Path(os.environ["PER_TASK_LOG_DIR"])
tmp_root = Path(os.environ["TMP_ROOT"])
command_file.parent.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
tmp_root.mkdir(parents=True, exist_ok=True)
python_bin = os.environ["PYTHON_BIN"]
runner_py = os.environ["RUNNER_PY"]
manifest_path = os.environ["MANIFEST_PATH"]
with command_file.open("w", encoding="utf-8") as handle:
    for idx in pending:
        idx_label = f"{idx:05d}"
        tmpdir = tmp_root / f"idx_{idx_label}"
        command = (
            f'mkdir -p "{tmpdir}" && export TMPDIR="{tmpdir}" && '
            f'export OOD_OUTDIR="{os.environ["OOD_OUTDIR"]}" && '
            f'export OOD_H5_DIR="{os.environ["OOD_H5_DIR"]}" && '
            f'"{python_bin}" -u "{runner_py}" --manifest-path "{manifest_path}" --index {idx}'
        )
        if force:
            command += " --force"
        command += f' > "{log_dir / f"idx_{idx_label}.out"}" 2> "{log_dir / f"idx_{idx_label}.err"}"'
        handle.write(command + "\n")
print(f"wrote {len(pending)} commands")
PY

rm -rf "${LAUNCHER_WORKDIR}"
mkdir -p "${LAUNCHER_WORKDIR}"
export TACC_PYLAUNCHER_WORKDIR="${LAUNCHER_WORKDIR}"

"${PYTHON_BIN}" - <<'PY'
import os, pylauncher
pylauncher.ClassicLauncher(os.environ["COMMAND_FILE"], workdir=os.environ["LAUNCHER_WORKDIR"], debug="host+job")
PY

echo "$(date -Is) | DONE | Resume finished" >&2
