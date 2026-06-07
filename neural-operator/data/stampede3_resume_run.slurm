#!/bin/bash
# Resume Sobol production run: only indices without run_{idx}.h5 in SOBOL_H5_DIR.
#
# Prerequisite on Stampede3 (after downloading partial results locally):
#   mkdir -p "$SCRATCH/opensees_sobol_full/h5"
#   unzip -o ~/results_1.zip -d "$SCRATCH/opensees_sobol_full"   # creates h5/run_*.h5
#   cp ~/sobol_timing.db "$SCRATCH/opensees_sobol_full/sobol_timing.db"
#
# Submit (recommended: 16 nodes x 48 cores, 48 h — ~4560 tasks need ~65–70 h @ 384 slots):
#   cd $PROJECT_ROOT
#   sbatch -N 16 --ntasks-per-node=48 -t 48:00:00 neural-operator/data/stampede3_resume_run.slurm
#
# Do NOT set SOBOL_OVERWRITE_MANIFEST=1 or FORCE_RERUN=1 unless you intend to rebuild/rerun.
#SBATCH -J sobol_resume
#SBATCH -o stampede3_sobol_resume.o%j
#SBATCH -e stampede3_sobol_resume.e%j
#SBATCH -p skx
#SBATCH -N 16
#SBATCH --ntasks-per-node=48
#SBATCH -t 48:00:00
#SBATCH -A ECS24003

set -euo pipefail

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Host=$(hostname)" >&2

module reset
module load pylauncher

PROJECT_ROOT="${PROJECT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/neural-operator/data}"
REPO_ROOT="${PROJECT_ROOT}"

RUNNER_PY="${RUNNER_PY:-${SCRIPT_DIR}/run_experiment.py}"
MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/sobol_manifest.csv}"
VENV_PATH="${VENV_PATH:-/work2/09739/kurtsoncco1406/stampede3/seiskit/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

SOBOL_NUM_SAMPLES="${SOBOL_NUM_SAMPLES:-256}"
SOBOL_RF_SEEDS_PER_SAMPLE="${SOBOL_RF_SEEDS_PER_SAMPLE:-30}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SOBOL_OVERWRITE_MANIFEST="${SOBOL_OVERWRITE_MANIFEST:-0}"

RUN_BASE="${RUN_BASE:-${SCRATCH:-${REPO_ROOT}}/opensees_sobol_full}"
COMMAND_FILE="${COMMAND_FILE:-${RUN_BASE}/commands_resume_${SLURM_JOB_ID:-manual}.txt}"
LAUNCHER_WORKDIR="${LAUNCHER_WORKDIR:-${RUN_BASE}/pylauncher_resume_${SLURM_JOB_ID:-manual}}"
PER_TASK_LOG_DIR="${PER_TASK_LOG_DIR:-${RUN_BASE}/per_index_logs}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER}/sobol_resume_${SLURM_JOB_ID:-manual}}"

export SOBOL_OUTDIR="${SOBOL_OUTDIR:-${RUN_BASE}/raw_runs}"
export SOBOL_H5_DIR="${SOBOL_H5_DIR:-${RUN_BASE}/h5}"
export SOBOL_TIMING_DB="${SOBOL_TIMING_DB:-${RUN_BASE}/sobol_timing.db}"
export SOBOL_H5_LOSSY="${SOBOL_H5_LOSSY:-1}"
export SOBOL_H5_DOWNSAMPLE="${SOBOL_H5_DOWNSAMPLE:-2}"
export SOBOL_MAX_TIME_PER_BATCH="${SOBOL_MAX_TIME_PER_BATCH:-28800}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_BASE}" "${SOBOL_OUTDIR}" "${SOBOL_H5_DIR}" "${PER_TASK_LOG_DIR}" "${TMP_ROOT}"

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

export SCRIPT_DIR MANIFEST_PATH SOBOL_NUM_SAMPLES SOBOL_RF_SEEDS_PER_SAMPLE SOBOL_OVERWRITE_MANIFEST
export SOBOL_H5_DIR FORCE_RERUN

read -r TOTAL_TASKS PENDING_COUNT COMPLETE_COUNT <<< "$("${PYTHON_BIN}" - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["SCRIPT_DIR"])
from sobol import ensure_manifest

manifest = ensure_manifest(
    path=os.environ["MANIFEST_PATH"],
    sample_count=int(os.environ["SOBOL_NUM_SAMPLES"]),
    seeds_per_sample=int(os.environ["SOBOL_RF_SEEDS_PER_SAMPLE"]),
    overwrite=os.environ.get("SOBOL_OVERWRITE_MANIFEST", "0") == "1",
)
h5_dir = Path(os.environ["SOBOL_H5_DIR"])
force = os.environ.get("FORCE_RERUN", "0") == "1"
pending = []
complete = 0
for idx in range(len(manifest)):
    h5 = h5_dir / f"run_{idx}.h5"
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
echo "$(date -Is) | OUTPUTS | RUN_BASE=${RUN_BASE} H5=${SOBOL_H5_DIR}" >&2

export COMMAND_FILE LAUNCHER_WORKDIR PER_TASK_LOG_DIR TMP_ROOT
export RUNNER_PY PYTHON_BIN FORCE_RERUN MANIFEST_PATH

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

sys_path = os.environ["SCRIPT_DIR"]
import sys
sys.path.insert(0, sys_path)
from sobol import ensure_manifest

manifest = ensure_manifest(
    path=os.environ["MANIFEST_PATH"],
    sample_count=int(os.environ["SOBOL_NUM_SAMPLES"]),
    seeds_per_sample=int(os.environ["SOBOL_RF_SEEDS_PER_SAMPLE"]),
    overwrite=os.environ.get("SOBOL_OVERWRITE_MANIFEST", "0") == "1",
)
h5_dir = Path(os.environ["SOBOL_H5_DIR"])
force = os.environ.get("FORCE_RERUN", "0") == "1"
pending = [idx for idx in range(len(manifest)) if force or not (h5_dir / f"run_{idx}.h5").exists()]

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

NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
TASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-48}}"
SLOTS=$((NODES * TASKS_PER_NODE))
echo "$(date -Is) | LAUNCH | ${NODES} x ${TASKS_PER_NODE} = ${SLOTS} slots | ${PENDING_COUNT} pending commands" >&2

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
