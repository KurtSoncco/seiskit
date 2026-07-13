#!/bin/bash
# Stampede3: Hallal 1D + Pretell OpenSees only (skips grf_2d / GIFNO).
# Run grf_2d locally with the surrogate — see README.
#
# Usage (from this directory or repo root):
#   ./submit_stampede3_opensees.sh
#   FORCE_RERUN=1 ./submit_stampede3_opensees.sh -N 8 --ntasks-per-node=48 -t 48:00:00
#   RV_SMOKE=1 RV_SMOKE_2D=1 ./submit_stampede3_opensees.sh -N 2 --ntasks-per-node=16 -t 12:00:00
#
# Extra args are forwarded to sbatch.
set -euo pipefail
cd "$(dirname "$0")"

export RV_SMOKE="${RV_SMOKE:-0}"
export RV_SMOKE_2D="${RV_SMOKE_2D:-1}"
export RV_USE_SURROGATE_2D=0
export RV_SKIP_METHODS="${RV_SKIP_METHODS:-grf_2d}"
export FORCE_RERUN="${FORCE_RERUN:-0}"

ROOT="$(cd ../.. && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON:-python3}"
fi

COUNTS="$("$PYTHON" -c "
from manifest import indices_for_methods, METHODS
skip = set((('${RV_SKIP_METHODS}' or '').split(',')))
skip.discard('')
keep = [m for m in METHODS if m not in skip]
idxs = indices_for_methods(keep)
from collections import Counter
from manifest import index_to_params
c = Counter(index_to_params(i).method for i in idxs)
print(len(idxs), dict(c))
")"
echo "Stampede3 OpenSees split: ${COUNTS}" >&2
echo "  RV_SKIP_METHODS=${RV_SKIP_METHODS}" >&2
echo "  Run grf_2d locally: ./submit_local.sh --all --2d --grf-only --n-seeds N" >&2

exec sbatch \
  --export=ALL,RV_SMOKE,RV_SMOKE_2D,RV_USE_SURROGATE_2D,RV_SKIP_METHODS,FORCE_RERUN \
  "$@" \
  stampede3_full_run.slurm
