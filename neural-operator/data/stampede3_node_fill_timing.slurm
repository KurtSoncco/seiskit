#!/bin/bash
# Fill one SKX node (48 cores) with independent serial Sobol indices for timing.
#
# Submit:
#   sbatch neural-operator/data/stampede3_node_fill_timing.slurm
#
# Override indices (space-separated manifest indices):
#   INDICES="0 210 5940" sbatch ...
#
# Default picks 48 indices stratified by H_discretized (one replicate each).
#SBATCH -J sobol_fill48
#SBATCH -o stampede3_fill48.o%j
#SBATCH -e stampede3_fill48.e%j
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -t 12:00:00
#SBATCH -A ECS24003

set -euo pipefail

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Host=$(hostname)" >&2

PROJECT_ROOT="${PROJECT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/neural-operator/data}"
RUNNER_PY="${RUNNER_PY:-${SCRIPT_DIR}/run_experiment.py}"
MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/sobol_manifest.csv}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

RUN_BASE="${RUN_BASE:-${SCRATCH:-${PROJECT_ROOT}}/opensees_sobol_fill48}"
FORCE_RERUN="${FORCE_RERUN:-0}"

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
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_BASE}" "${SOBOL_OUTDIR}" "${SOBOL_H5_DIR}"

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

export SCRIPT_DIR MANIFEST_PATH RUNNER_PY PYTHON_BIN FORCE_RERUN

INDICES="${INDICES:-$("${PYTHON_BIN}" - <<'PY'
import os
import sys

sys.path.insert(0, os.environ["SCRIPT_DIR"])
from sobol import load_manifest_csv

manifest = load_manifest_csv(os.environ["MANIFEST_PATH"])
# One row per sample_id (replicate_id==0) for stratified H coverage.
by_sample = {}
for entry in manifest:
    if entry.replicate_id == 0:
        by_sample[entry.sample_id] = entry

entries = sorted(by_sample.values(), key=lambda e: e.H_discretized)
n = len(entries)
slots = 48
if n < slots:
    raise SystemExit(f"Need at least {slots} unique samples, got {n}")

chosen = []
for slot in range(slots):
    pos = int(round(slot * (n - 1) / (slots - 1)))
    chosen.append(entries[pos].index)
print(" ".join(str(i) for i in chosen))
PY
)}"

read -r -a INDEX_LIST <<< "${INDICES}"
if [ "${#INDEX_LIST[@]}" -ne 48 ]; then
  echo "ERROR: expected 48 indices, got ${#INDEX_LIST[@]}" >&2
  exit 3
fi

echo "$(date -Is) | INDICES | ${INDICES}" >&2
echo "$(date -Is) | RUN_BASE | ${RUN_BASE}" >&2

srun --exclusive -n 48 bash -c '
  set -euo pipefail
  rank="${SLURM_PROCID:-0}"
  indices=('"${INDICES}"')
  idx="${indices[$rank]}"
  log_dir="'"${RUN_BASE}"'/srun_logs"
  mkdir -p "$log_dir"
  export SOBOL_OUTDIR SOBOL_H5_DIR SOBOL_TIMING_DB SOBOL_H5_LOSSY SOBOL_H5_DOWNSAMPLE SOBOL_MAX_TIME_PER_BATCH
  export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS PYTHONPATH
  cmd=(
    "'"${PYTHON_BIN}"'" -u "'"${RUNNER_PY}"'"
    --manifest-path "'"${MANIFEST_PATH}"'"
    --index "$idx"
  )
  if [ "'"${FORCE_RERUN}"'" = "1" ]; then
    cmd+=(--force)
  fi
  echo "$(date -Is) | rank=$rank idx=$idx start" >&2
  "${cmd[@]}" > "$log_dir/idx_${idx}.out" 2> "$log_dir/idx_${idx}.err"
  echo "$(date -Is) | rank=$rank idx=$idx done" >&2
'

echo "$(date -Is) | DONE | 48 tasks finished" >&2
echo "$(date -Is) | SUMMARY | Query ${SOBOL_TIMING_DB}:" >&2
"${PYTHON_BIN}" - <<'PY'
import os
import sqlite3
from pathlib import Path

db = Path(os.environ["SOBOL_TIMING_DB"])
conn = sqlite3.connect(db)
rows = conn.execute(
    """
    SELECT idx, H_discretized, Lz_discretized, duration,
           total_time_sec, field_sec, model_sec, analysis_sec,
           substr(status, 1, 40)
    FROM sobol_timing_per_index
    ORDER BY total_time_sec DESC
    """
).fetchall()
conn.close()
if not rows:
    print("  (no rows in timing DB yet)")
else:
    totals = [r[4] for r in rows if r[4]]
    print(f"  rows={len(rows)} min={min(totals):.0f}s max={max(totals):.0f}s median={sorted(totals)[len(totals)//2]:.0f}s")
    for row in rows[:5]:
        print(" ", row)
    if len(rows) > 5:
        print("  ...")
PY
