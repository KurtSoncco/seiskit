#!/bin/bash
# One task with 24 CPUs per array element so GNU Parallel can run 24 sims in parallel on the node.
#SBATCH --job-name=em8100_phase1_second
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --constraint=savio2_c24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --array=0-8099%50
#SBATCH --output=logs/array_job_%A_task_%a.out
#SBATCH --error=logs/array_job_%A_task_%a.err
#SBATCH --exclude=n0087.savio2
# Create logs/ before submitting (Slurm does not create it). From this dir: mkdir -p logs; sbatch phase1_savio2.sh  (or use ./submit_phase1.sh).
# To put Slurm stdout/err under per_idx too, use:
#   --output=logs/per_idx/job_%A/task_%a/slurm.out --error=logs/per_idx/job_%A/task_%a/slurm.err
# and precreate dirs before submit (Slurm does not create them): for a in $(seq 0 336); do mkdir -p logs/per_idx/job_<JOB_ID>/task_$a; done

# Phase 1: 8,088 sims (indices 0-8087). Savio2 whole-node; GNU Parallel runs
# N sims per array element. Array 0-10 = validation; for full run use: --array=0-336
# 337 array elements × 24 sims = 8088. Submit: sbatch phase1_savio2.sh
#
# Hardening: per-index logs, joblog, retries, timeout, scratch TMPDIR, pre-flight
# writes, post-run failure summary. Rerun only failed indices via joblog.
#
# FORCE_RERUN=1 to re-run even when output exists.
# CONCURRENCY: 24 runs per node (override with CONCURRENCY=N if needed).
FORCE_RERUN=${FORCE_RERUN:-0}
CONCURRENCY="${CONCURRENCY:-${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-24}}}"
mkdir -p logs
set -euo pipefail

# Create/write-check result dir and co-locate joblog (deterministic per-task logs).
# Write to local scratch during execution to avoid shared FS contention; copy back at end.
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
RESULTS_BASE_FINAL="logs/per_idx/job_${SLURM_JOB_ID:-0}/task_${SLURM_ARRAY_TASK_ID:-0}"
# Node-local RESULTS_BASE when SLURM_TMPDIR unset; copy back at end.
if [ -n "${SLURM_TMPDIR:-}" ]; then
  RESULTS_BASE="${SLURM_TMPDIR}/parallel_results/job_${SLURM_JOB_ID:-0}/task_${SLURM_ARRAY_TASK_ID:-0}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  RESULTS_BASE="/tmp/em8100_${USER}_job_${SLURM_JOB_ID}_task_${TASK_ID}"
else
  RESULTS_BASE="$RESULTS_BASE_FINAL"
fi
mkdir -p "$RESULTS_BASE" || { echo "ERROR mkdir $RESULTS_BASE" >&2; exit 12; }
touch "$RESULTS_BASE/.w" || { echo "ERROR write $RESULTS_BASE" >&2; exit 13; }
rm -f "$RESULTS_BASE/.w"
# Joblog at final location from start (tiny; no need to stage with node-local results).
mkdir -p "$RESULTS_BASE_FINAL"
JOBLOG="$RESULTS_BASE_FINAL/joblog.tsv"

# Disable nested threading for each payload (export before parallel so all jobs inherit).
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# If using BLIS or Accelerate, set BLIS_NUM_THREADS=1 / VECLIB_MAXIMUM_THREADS=1 as needed.

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Task=${SLURM_ARRAY_TASK_ID:-<local>} Host=$(hostname)" >&2
echo "$(date -Is) | RESOURCES | NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-} CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-} CONCURRENCY=${CONCURRENCY}" >&2

# Diagnostic: CPU model, MHz, NUMA (consistent governor/affinity check).
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "$(date -Is) | CPU/NUMA | lscpu (model, MHz):" >&2
    lscpu 2>/dev/null | egrep 'Model name|MHz' || true
    echo "$(date -Is) | CPU/NUMA | numactl --hardware:" >&2
    numactl --hardware 2>/dev/null || true
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "$(date -Is) | MODULE | Purging modules..." >&2
    module purge
    echo "$(date -Is) | MODULE | Loading gnu-parallel, gcc, openblas..." >&2
    module load parallel
    module load gcc/13.2.0 openblas/0.3.24
    echo "$(date -Is) | MODULE | Module load complete." >&2
fi

source /global/home/users/kurtwal98/seiskit/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

if [ -n "${SLURM_JOB_ID:-}" ]; then
    export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
start_epoch=$(date +%s)

# Stage hot shared inputs once per array task to local scratch (reduces shared FS contention).
if [ -n "${SLURM_TMPDIR:-}" ]; then
  mkdir -p "$SLURM_TMPDIR/data"
  # Optional: rsync or ln -s large static inputs here, then set EMULATOR_8100_DATADIR="$SLURM_TMPDIR/data" for payloads.
  # Example: rsync -a "${SLURM_SUBMIT_DIR:-.}/static_inputs/" "$SLURM_TMPDIR/data/" 2>/dev/null || true
  export EMULATOR_8100_DATADIR="${EMULATOR_8100_DATADIR:-$SLURM_TMPDIR/data}"
fi

PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"
RUNNER_PY="${SLURM_SUBMIT_DIR:-$PWD}/run_experiment.py"

command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python binary not found at ${PYTHON_BIN}" >&2; exit 2; }
test -r "${RUNNER_PY}" >/dev/null || { echo "ERROR: Runner script not readable at ${RUNNER_PY}" >&2; exit 2; }

echo "$(date -Is) | PREFLIGHT | Verifying Python and OpenSees..." >&2
timeout 30s ${PYTHON_BIN} - <<'PYEOF'
import sys
print('PYTHON_OK', sys.version.split()[0])
try:
    import openseespy.opensees as ops  # noqa
    print('OPENSEES_OK')
except Exception as e:
    print(f'OPENSEES_IMPORT_FAIL: {e}')
    sys.exit(3)
PYEOF
PRE_RC=$?
if [ ${PRE_RC} -ne 0 ]; then
  echo "$(date -Is) | ERROR | Preflight check failed (Code: ${PRE_RC}). Exiting." >&2
  exit ${PRE_RC}
fi
echo "$(date -Is) | PREFLIGHT | Preflight success." >&2

# Chunking: 24 sims per array element (0-based indices). Last chunk clamped to TOTAL.
CHUNK=24
TOTAL=8088
START=$((TASK_ID * CHUNK))
END=$((START + CHUNK))
if [ "${END}" -gt "${TOTAL}" ]; then
  END=${TOTAL}
fi
COUNT=$((END - START))
EXTRA_ARGS=""
[ "${FORCE_RERUN}" = "1" ] && EXTRA_ARGS="--force"
export PYTHON_BIN RUNNER_PY EXTRA_ARGS

# Scratch-based paths: avoid $HOME (common source of odd early terminations).
# Optional: stage common inputs to local node scratch for faster reads:
#   cp "$SLURM_SUBMIT_DIR"/run_experiment.py "$SLURM_TMPDIR"/; RUNNER_PY="$SLURM_TMPDIR/run_experiment.py"
#   run from "$SLURM_TMPDIR", then at end: rsync -a "$SLURM_TMPDIR"/results/ "$SLURM_SUBMIT_DIR"/results/
export PARALLEL_HOME=/global/scratch/users/$USER/.parallel
export TMPDIR=/global/scratch/users/$USER/tmp/job_${SLURM_JOB_ID:-0}_task_${TASK_ID}_root
# Per-task scratch for OpenSees output; then aggregate+compress to archives/ (run post-step in Python).
export EMULATOR_8100_OUTDIR=/global/scratch/users/$USER/opensees_runs/${SLURM_JOB_ID:-0}_${TASK_ID}
# H5 output to scratch (avoids 50 GB home quota; 8100 files need ~80-160 GB with lossy compression).
export EMULATOR_8100_H5_DIR=/global/scratch/users/$USER/emulator8100_h5_second
export EMULATOR_8100_H5_LOSSY=1
export EMULATOR_8100_H5_DOWNSAMPLE=2
# Parallel joblog and per-index results under RESULTS_BASE (joblog.tsv + <index>/{stdout,stderr,seq}).
mkdir -p "$PARALLEL_HOME" "$TMPDIR" "$EMULATOR_8100_OUTDIR/archives" "$EMULATOR_8100_H5_DIR"
mkdir -p results/h5

# Pre-flight writes: fail fast if we cannot write logs or parallel state (permission/quota).
touch "logs/write_test_${SLURM_JOB_ID:-0}_${TASK_ID}" || { echo "$(date -Is) | ERROR | cannot write logs/" >&2; exit 10; }
touch "$PARALLEL_HOME/write_test_${SLURM_JOB_ID:-0}_${TASK_ID}" || { echo "$(date -Is) | ERROR | cannot write PARALLEL_HOME: $PARALLEL_HOME" >&2; exit 11; }
rm -f "logs/write_test_${SLURM_JOB_ID:-0}_${TASK_ID}" "$PARALLEL_HOME/write_test_${SLURM_JOB_ID:-0}_${TASK_ID}"

# Per-sim timeout (4h) so one stuck run doesn't hang the slot; job time is 5h.
SIM_TIMEOUT=14400
export SIM_TIMEOUT

# Retries: transient glitches get one automatic retry.
PARALLEL_RETRIES=1

# Diagnostics: verify parallel config and expected concurrency (timeout so a bad node does not hang).
echo "$(date -Is) | DIAG | Checking parallel version and config..." >&2
timeout 10s parallel --version >&2 || true
echo "$(date -Is) | DIAG | Will launch ${CONCURRENCY} concurrent jobs" >&2
echo "$(date -Is) | DIAG | Results base: $RESULTS_BASE" >&2
if [ -n "${SLURM_TMPDIR:-}" ]; then
  echo "$(date -Is) | DIAG | Using local scratch: $SLURM_TMPDIR" >&2
fi
echo "$(date -Is) | DIAG | START=${START} END=${END} COUNT=${COUNT} (indices ${START}..$((END-1)))" >&2

echo "$(date -Is) | RUN | Task ${TASK_ID}: indices ${START}..$((END-1)) (${COUNT} sims) -j ${CONCURRENCY} timeout=${SIM_TIMEOUT}s retries=${PARALLEL_RETRIES}..." >&2

if [ "$START" -ge "$END" ]; then
  echo "$(date -Is) | SUMMARY | Nothing to do (START=$START END=$END)" >&2
  exit 0
fi

# Parallel tmpdir: use same node-local base when we have one, else /tmp.
if [ "$RESULTS_BASE" != "$RESULTS_BASE_FINAL" ]; then
  PARALLEL_TMPDIR="${RESULTS_BASE}/parallel_tmp"
else
  PARALLEL_TMPDIR="${SLURM_TMPDIR:-/tmp}/parallel_tmp_${SLURM_JOB_ID:-0}_${TASK_ID}"
fi
mkdir -p "$PARALLEL_TMPDIR"

set +e
seq "${START}" $((END - 1)) | parallel -j "${CONCURRENCY}" \
  --tmpdir "$PARALLEL_TMPDIR" \
  --line-buffer \
  --verbose \
  --retries "${PARALLEL_RETRIES}" \
  --joblog "${JOBLOG}" \
  --results "${RESULTS_BASE}/shard_{= \$_ = int(\$_/1000) =}/{}" \
  --tagstring "idx={} slot={#} host=$(hostname)" \
  'slot={#}; idx={};
   base_tmp="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}";
   idx_tmp="$base_tmp/idx_$idx";
   mkdir -p "$idx_tmp";
   export TMPDIR="$idx_tmp";
   trap "rm -rf \"$idx_tmp\"" EXIT;
   if command -v taskset >/dev/null 2>&1; then
     exec taskset -c $((slot-1)) timeout "$SIM_TIMEOUT" "$PYTHON_BIN" -u "$RUNNER_PY" --index "$idx" $EXTRA_ARGS;
   else
     exec timeout "$SIM_TIMEOUT" "$PYTHON_BIN" -u "$RUNNER_PY" --index "$idx" $EXTRA_ARGS;
   fi'
PARALLEL_RC=$?
set -e
echo "$(date -Is) | SUMMARY | parallel rc=$PARALLEL_RC" >&2

# Copy results from node-local storage back to final location when we used scratch or /tmp.
if [ "$RESULTS_BASE" != "$RESULTS_BASE_FINAL" ] && [ -d "$RESULTS_BASE" ]; then
  echo "$(date -Is) | COPY | Copying results from $RESULTS_BASE to $RESULTS_BASE_FINAL..." >&2
  mkdir -p "$RESULTS_BASE_FINAL"
  rsync -aW --no-compress "$RESULTS_BASE/" "$RESULTS_BASE_FINAL/" 2>/dev/null || {
    echo "$(date -Is) | WARNING | rsync failed, trying cp..." >&2
    cp -r "$RESULTS_BASE"/* "$RESULTS_BASE_FINAL/" 2>/dev/null || true
  }
fi

# Summaries: make failures actionable (counts + exact indices to requeue).
OK_COUNT=0
FAILED_COUNT=0
if [ -r "${JOBLOG}" ]; then
  FAILED_LINES=$(awk 'NR>1 && $7!=0 {print}' "${JOBLOG}" 2>/dev/null || true)
  FAILED_COUNT=$(echo "${FAILED_LINES}" | grep -c . 2>/dev/null || echo 0)
  OK_COUNT=$(awk 'NR>1 && $7==0 {c++} END {print c+0}' "${JOBLOG}" 2>/dev/null || echo 0)
  echo "$(date -Is) | SUMMARY | OK=${OK_COUNT} failed=${FAILED_COUNT} parallel_rc=${PARALLEL_RC}" >&2
  if [ "${FAILED_COUNT}" -gt 0 ]; then
    echo "$(date -Is) | FAILED ROWS (joblog):" >&2
    echo "${FAILED_LINES}" >&2
    echo "$(date -Is) | INDICES TO RERUN (use for sbatch --array= or single-index resubmit):" >&2
    FAILED_IDX=$(awk -v start="${START}" 'NR>1 && $7!=0 {print start+$1-1}' "${JOBLOG}" | sort -n | uniq)
    echo "${FAILED_IDX}" | tr '\n' ' ' && echo "" >&2
    RERUN_ARRAY=$(echo "${FAILED_IDX}" | awk -v chunk="${CHUNK}" '{print int($0/chunk)}' | sort -n | uniq | paste -sd, -)
    if [ -n "${RERUN_ARRAY}" ]; then
      echo "Rerun failed indices: sbatch --array=${RERUN_ARRAY} phase1_savio2.sh" >&2
    fi
  fi
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "$(date -Is) | ACCOUNTING | Per-element MaxRSS/State:" >&2
  sacct -X -j "${SLURM_JOB_ID}_${TASK_ID}" --format=JobID,MaxRSS,State,ExitCode -P 2>/dev/null || true
  # Diagnostic: check how many Python processes were running (should be ~CONCURRENCY during peak).
  echo "$(date -Is) | DIAG | Checking for concurrent Python processes..." >&2
  ps aux 2>/dev/null | grep -E "[p]ython.*run_experiment" | wc -l | xargs -I{} echo "$(date -Is) | DIAG | Found {} Python run_experiment processes" >&2 || true
fi

# Append one row per array task to shared SQLite timing DB (atomic, queryable). Uses venv Python + stdlib sqlite3 (uv sync).
end_epoch=$(date +%s)
wall_s=$((end_epoch - start_epoch))
TIMING_DB="${SLURM_SUBMIT_DIR:-.}/timing.db"
cpu_val=""
max_rss_val=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
  sacct_line=$(sacct -X -j "${SLURM_JOB_ID}_${TASK_ID}" --format=AveCPU,MaxRSS -n -P --noheader 2>/dev/null | head -1)
  if [ -n "${sacct_line}" ]; then
    c=$(echo "$sacct_line" | cut -d'|' -f1)
    m=$(echo "$sacct_line" | cut -d'|' -f2)
    [ -n "$c" ] && cpu_val="$c"
    if [ -n "$m" ]; then
      max_rss_mb=$(echo "$m" | awk '/K$/{gsub(/K/,""); print $0/1024} /M$/{gsub(/M/,""); print $0+0} /G$/{gsub(/G/,""); print $0*1024}')
      [ -n "$max_rss_mb" ] && max_rss_val="$max_rss_mb"
    fi
  fi
fi
"${PYTHON_BIN}" - "${TIMING_DB}" "${SLURM_JOB_ID:-0}" "${TASK_ID}" "${HOSTNAME:-unknown}" "${start_epoch}" "${end_epoch}" "${wall_s}" "${cpu_val}" "${max_rss_val}" "${PARALLEL_RC}" "${OK_COUNT}" "${FAILED_COUNT}" <<'PYTIMING' 2>/dev/null || true
import sqlite3
import sys
import json
_, db, jobid, taskid, host, start_s, end_s, wall_s, cpu_s, max_rss_mb, exitcode, ok, failed = sys.argv
conn = sqlite3.connect(db)
conn.execute("""
CREATE TABLE IF NOT EXISTS timing(ts TEXT, jobid INT, taskid INT, host TEXT, start_s INT, end_s INT, wall_s INT, cpu_s REAL, max_rss_mb REAL, gpu_util REAL, exit INT, meta TEXT)
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_task ON timing(taskid)")
meta = json.dumps({"ok": int(ok), "failed": int(failed)})
conn.execute(
    """INSERT INTO timing(ts, jobid, taskid, host, start_s, end_s, wall_s, cpu_s, max_rss_mb, gpu_util, exit, meta)
       VALUES(datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
    (int(jobid), int(taskid), host, int(start_s), int(end_s), int(wall_s),
     float(cpu_s) if cpu_s else None, float(max_rss_mb) if max_rss_mb else None, int(exitcode), meta)
)
conn.commit()
conn.close()
PYTIMING

exit $PARALLEL_RC
