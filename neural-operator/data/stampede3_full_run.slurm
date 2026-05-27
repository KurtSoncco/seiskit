#!/bin/bash
# Full Sobol run on TACC Stampede3 using pylauncher for many independent serial tasks.
#
# Submit from anywhere with:
#   sbatch neural-operator/data/stampede3_full_run.slurm
#
# Recommended first smoke test:
#   sbatch -p skx-dev -N 1 -t 02:00:00 neural-operator/data/stampede3_full_run.slurm
#
# Defaults below target the full production run on SKX nodes.
# 8 SKX nodes => up to 8 * 48 = 384 concurrent serial simulations.
# Adjust -N and -t after your first timing measurements.
#SBATCH -J sobol_full
#SBATCH -o stampede3_sobol_full.o%j
#SBATCH -e stampede3_sobol_full.e%j
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
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/neural-operator/data}"
REPO_ROOT="${PROJECT_ROOT}"

RUNNER_PY="${RUNNER_PY:-${SCRIPT_DIR}/run_experiment.py}"
MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/sobol_manifest.csv}"
VENV_PATH="${VENV_PATH:-/work2/09739/kurtsoncco1406/stampede3/seiskit/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

SOBOL_NUM_SAMPLES="${SOBOL_NUM_SAMPLES:-256}"
SOBOL_RF_SEEDS_PER_SAMPLE="${SOBOL_RF_SEEDS_PER_SAMPLE:-30}"
SOBOL_SAMPLER_METHOD="${SOBOL_SAMPLER_METHOD:-sobol}"
SOBOL_SAMPLER_SEED="${SOBOL_SAMPLER_SEED:-42}"
SOBOL_RF_SEED_SEED="${SOBOL_RF_SEED_SEED:-42}"
FORCE_RERUN="${FORCE_RERUN:-0}"

RUN_BASE="${RUN_BASE:-${SCRATCH:-${REPO_ROOT}}/opensees_sobol_full}"
COMMAND_FILE="${COMMAND_FILE:-${RUN_BASE}/commands_${SLURM_JOB_ID:-manual}.txt}"
LAUNCHER_WORKDIR="${LAUNCHER_WORKDIR:-${RUN_BASE}/pylauncher_${SLURM_JOB_ID:-manual}}"
PER_TASK_LOG_DIR="${PER_TASK_LOG_DIR:-${RUN_BASE}/per_index_logs}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER}/sobol_${SLURM_JOB_ID:-manual}}"

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
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_BASE}" "${SOBOL_OUTDIR}" "${SOBOL_H5_DIR}" "${PER_TASK_LOG_DIR}" "${TMP_ROOT}"

if [ ! -d "${SCRIPT_DIR}" ]; then
  echo "ERROR: script directory not found at ${SCRIPT_DIR}" >&2
  echo "Set SCRIPT_DIR=/path/to/neural-operator/data or PROJECT_ROOT=/path/to/repo before sbatch." >&2
  exit 2
fi
if [ ! -d "${REPO_ROOT}" ]; then
  echo "ERROR: repo root not found at ${REPO_ROOT}" >&2
  echo "Set PROJECT_ROOT=/path/to/repo before sbatch." >&2
  exit 2
fi
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv activation script not found at ${VENV_PATH}/bin/activate" >&2
  echo "Set VENV_PATH=/path/to/venv before sbatch if your Stampede3 path is different." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

command -v "${PYTHON_BIN}" >/dev/null || {
  echo "ERROR: Python binary not found at ${PYTHON_BIN}" >&2
  echo "Set VENV_PATH=/path/to/venv or PYTHON_BIN=/path/to/python before sbatch." >&2
  exit 2
}
test -r "${RUNNER_PY}" || {
  echo "ERROR: Runner script not readable at ${RUNNER_PY}" >&2
  exit 2
}

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

echo "$(date -Is) | PREFLIGHT | Verifying Python, pylauncher, paramiko, and OpenSees..." >&2
echo "$(date -Is) | PATHS | PROJECT_ROOT=${PROJECT_ROOT}" >&2
echo "$(date -Is) | PATHS | SCRIPT_DIR=${SCRIPT_DIR}" >&2
echo "$(date -Is) | PATHS | RUNNER_PY=${RUNNER_PY}" >&2
echo "$(date -Is) | PATHS | PYTHON_BIN=${PYTHON_BIN}" >&2
timeout 60s "${PYTHON_BIN}" - <<'PY'
import pylauncher  # noqa: F401
import paramiko  # noqa: F401
import sys

print("PYTHON_OK", sys.version.split()[0])
print("PYLAUNCHER_OK")
print("PARAMIKO_OK")
try:
    import openseespy.opensees as ops  # noqa: F401
    print("OPENSEES_OK")
except Exception as exc:
    print(f"OPENSEES_IMPORT_FAIL: {exc}")
    raise SystemExit(3)
PY

export SCRIPT_DIR MANIFEST_PATH SOBOL_NUM_SAMPLES SOBOL_RF_SEEDS_PER_SAMPLE
export SOBOL_SAMPLER_METHOD SOBOL_SAMPLER_SEED SOBOL_RF_SEED_SEED

TOTAL_TASKS="$("${PYTHON_BIN}" - <<'PY'
import os
import sys

sys.path.insert(0, os.environ["SCRIPT_DIR"])
from sobol import ensure_manifest

manifest = ensure_manifest(
    path=os.environ["MANIFEST_PATH"],
    sample_count=int(os.environ["SOBOL_NUM_SAMPLES"]),
    seeds_per_sample=int(os.environ["SOBOL_RF_SEEDS_PER_SAMPLE"]),
    method=os.environ["SOBOL_SAMPLER_METHOD"],
    sampler_seed=int(os.environ["SOBOL_SAMPLER_SEED"]),
    rf_seed_seed=int(os.environ["SOBOL_RF_SEED_SEED"]),
    overwrite=False,
)
print(len(manifest))
PY
)"

if [ "${TOTAL_TASKS}" -le 0 ]; then
  echo "ERROR: Manifest contains no tasks." >&2
  exit 3
fi

echo "$(date -Is) | MANIFEST | ${MANIFEST_PATH} -> ${TOTAL_TASKS} tasks" >&2
echo "$(date -Is) | OUTPUTS | RUN_BASE=${RUN_BASE}" >&2
echo "$(date -Is) | OUTPUTS | H5=${SOBOL_H5_DIR}" >&2
echo "$(date -Is) | OUTPUTS | RAW=${SOBOL_OUTDIR}" >&2

export COMMAND_FILE LAUNCHER_WORKDIR PER_TASK_LOG_DIR TMP_ROOT
export RUNNER_PY PYTHON_BIN FORCE_RERUN TOTAL_TASKS

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

command_file = Path(os.environ["COMMAND_FILE"])
command_file.parent.mkdir(parents=True, exist_ok=True)
log_dir = Path(os.environ["PER_TASK_LOG_DIR"])
log_dir.mkdir(parents=True, exist_ok=True)
tmp_root = Path(os.environ["TMP_ROOT"])
tmp_root.mkdir(parents=True, exist_ok=True)

python_bin = os.environ["PYTHON_BIN"]
runner_py = os.environ["RUNNER_PY"]
manifest_path = os.environ["MANIFEST_PATH"]
total_tasks = int(os.environ["TOTAL_TASKS"])
force = os.environ["FORCE_RERUN"] == "1"

with command_file.open("w", encoding="utf-8") as handle:
    for idx in range(total_tasks):
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
PY

echo "$(date -Is) | COMMANDS | Wrote ${COMMAND_FILE}" >&2
echo "$(date -Is) | LAUNCH | Using ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}} node(s) via pylauncher" >&2

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
