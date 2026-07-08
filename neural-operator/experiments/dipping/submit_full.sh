#!/bin/bash
# Convenience wrapper to submit the full dipping production run on Stampede3.
#
# Usage (from repo root on Stampede3):
#   bash neural-operator/experiments/dipping/submit_full.sh
#
# Smoke test:
#   EXP_SAMPLES_PER_ANGLE=1 EXP_SEEDS_PER_ANGLE=2 EXP_OVERWRITE_MANIFEST=1 FORCE_RERUN=1 \
#     bash neural-operator/experiments/dipping/submit_full.sh smoke
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODE="${1:-production}"

if [ "${MODE}" = "smoke" ]; then
  EXP_SAMPLES_PER_ANGLE="${EXP_SAMPLES_PER_ANGLE:-1}" \
  EXP_SEEDS_PER_ANGLE="${EXP_SEEDS_PER_ANGLE:-2}" \
  EXP_OVERWRITE_MANIFEST="${EXP_OVERWRITE_MANIFEST:-1}" \
  FORCE_RERUN="${FORCE_RERUN:-1}" \
  sbatch -N 1 --ntasks-per-node=4 -t 4:00:00 \
    "${SCRIPT_DIR}/stampede3_full_run.slurm"
elif [ "${MODE}" = "production" ]; then
  sbatch -N 4 --ntasks-per-node=48 -t 48:00:00 \
    "${SCRIPT_DIR}/stampede3_full_run.slurm"
else
  echo "Usage: $0 [smoke|production]" >&2
  exit 2
fi
