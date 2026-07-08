#!/bin/bash
# Resume dipping experiment: only indices without case_{idx}.h5 in EXP_H5_DIR.
#
# Submit:
#   sbatch -N 4 --ntasks-per-node=48 -t 48:00:00 \
#     neural-operator/experiments/dipping/stampede3_resume_run.slurm
#SBATCH -J dipping_resume
#SBATCH -o stampede3_dipping_resume.o%j
#SBATCH -e stampede3_dipping_resume.e%j
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
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/neural-operator/experiments/dipping}"
REPO_ROOT="${PROJECT_ROOT}"

RUNNER_PY="${RUNNER_PY:-${SCRIPT_DIR}/run_experiment.py}"
MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/manifest.csv}"
VENV_PATH="${VENV_PATH:-/work2/09739/kurtsoncco1406/stampede3/seiskit/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

EXP_SAMPLES_PER_ANGLE="${EXP_SAMPLES_PER_ANGLE:-15}"
EXP_SEEDS_PER_ANGLE="${EXP_SEEDS_PER_ANGLE:-15}"
EXP_OVERWRITE_MANIFEST="${EXP_OVERWRITE_MANIFEST:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

RUN_BASE="${RUN_BASE:-${SCRATCH:-${REPO_ROOT}}/opensees_dipping}"
COMMAND_FILE="${COMMAND_FILE:-${RUN_BASE}/commands_resume_${SLURM_JOB_ID:-manual}.txt}"
LAUNCHER_WORKDIR="${LAUNCHER_WORKDIR:-${RUN_BASE}/pylauncher_resume_${SLURM_JOB_ID:-manual}}"
PER_TASK_LOG_DIR="${PER_TASK_LOG_DIR:-${RUN_BASE}/per_index_logs}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER}/dipping_resume_${SLURM_JOB_ID:-manual}}"

export EXP_OUTDIR="${EXP_OUTDIR:-${RUN_BASE}/raw_runs}"
export EXP_H5_DIR="${EXP_H5_DIR:-${RUN_BASE}/h5}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_BASE}" "${EXP_OUTDIR}" "${EXP_H5_DIR}" "${PER_TASK_LOG_DIR}" "${TMP_ROOT}"

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

export SCRIPT_DIR MANIFEST_PATH EXP_SAMPLES_PER_ANGLE EXP_SEEDS_PER_ANGLE
export EXP_OVERWRITE_MANIFEST EXP_H5_DIR FORCE_RERUN

read -r TOTAL_TASKS PENDING_COUNT COMPLETE_COUNT <<< "$("${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

script_dir = Path(os.environ["SCRIPT_DIR"])
spec = importlib.util.spec_from_file_location("dipping_manifest", script_dir / "manifest.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["dipping_manifest"] = mod
spec.loader.exec_module(mod)

manifest = mod.ensure_manifest(
    path=os.environ["MANIFEST_PATH"],
    samples_per_angle=int(os.environ["EXP_SAMPLES_PER_ANGLE"]),
    seeds_per_angle=int(os.environ["EXP_SEEDS_PER_ANGLE"]),
    overwrite=os.environ.get("EXP_OVERWRITE_MANIFEST", "0") == "1",
)
h5_dir = Path(os.environ["EXP_H5_DIR"])
force = os.environ.get("FORCE_RERUN", "0") == "1"
pending = []
complete = 0
for idx in range(len(manifest)):
    h5 = h5_dir / f"case_{idx}.h5"
    if force or not h5.exists():
        pending.append(idx)
    else:
        complete += 1
print(len(manifest), len(pending), complete)
PY
)"

if [ "${PENDING_COUNT}" -le 0 ]; then
  echo "$(date -Is) | DONE | No pending indices (complete=${COMPLETE_COUNT}/${TOTAL_TASKS})" >&2
  exit 0
fi

echo "$(date -Is) | RESUME | complete=${COMPLETE_COUNT}/${TOTAL_TASKS} pending=${PENDING_COUNT}" >&2

export COMMAND_FILE LAUNCHER_WORKDIR PER_TASK_LOG_DIR TMP_ROOT
export RUNNER_PY PYTHON_BIN MANIFEST_PATH

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

script_dir = Path(os.environ["SCRIPT_DIR"])
spec = importlib.util.spec_from_file_location("dipping_manifest", script_dir / "manifest.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["dipping_manifest"] = mod
spec.loader.exec_module(mod)

manifest = mod.ensure_manifest(
    path=os.environ["MANIFEST_PATH"],
    samples_per_angle=int(os.environ["EXP_SAMPLES_PER_ANGLE"]),
    seeds_per_angle=int(os.environ["EXP_SEEDS_PER_ANGLE"]),
    overwrite=os.environ.get("EXP_OVERWRITE_MANIFEST", "0") == "1",
)
h5_dir = Path(os.environ["EXP_H5_DIR"])
force = os.environ.get("FORCE_RERUN", "0") == "1"
pending = [idx for idx in range(len(manifest)) if force or not (h5_dir / f"case_{idx}.h5").exists()]

command_file = Path(os.environ["COMMAND_FILE"])
command_file.parent.mkdir(parents=True, exist_ok=True)
log_dir = Path(os.environ["PER_TASK_LOG_DIR"])
log_dir.mkdir(parents=True, exist_ok=True)
tmp_root = Path(os.environ["TMP_ROOT"])
tmp_root.mkdir(parents=True, exist_ok=True)

python_bin = os.environ["PYTHON_BIN"]
runner_py = os.environ["RUNNER_PY"]
manifest_path = os.environ["MANIFEST_PATH"]

with command_file.open("w", encoding="utf-8") as handle:
    for idx in pending:
        idx_label = f"{idx:05d}"
        tmpdir = tmp_root / f"idx_{idx_label}"
        stdout_path = log_dir / f"idx_{idx_label}.out"
        stderr_path = log_dir / f"idx_{idx_label}.err"
        command = (
            f'mkdir -p "{tmpdir}" && '
            f'export TMPDIR="{tmpdir}" && '
            f'export EXP_OUTDIR="{os.environ["EXP_OUTDIR"]}" && '
            f'export EXP_H5_DIR="{os.environ["EXP_H5_DIR"]}" && '
            f'"{python_bin}" -u "{runner_py}" '
            f'--manifest-path "{manifest_path}" '
            f'--index {idx}'
        )
        if force:
            command += " --force"
        command += f' > "{stdout_path}" 2> "{stderr_path}"'
        handle.write(command + "\n")

print(f"wrote {len(pending)} commands to {command_file}")
PY

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

echo "$(date -Is) | DONE | Resume launcher finished" >&2
