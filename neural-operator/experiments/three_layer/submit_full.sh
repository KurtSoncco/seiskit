#!/bin/bash
# Convenience wrapper to submit the full three-layer production run on Stampede3.
#
# Usage (from repo root on Stampede3):
#   bash neural-operator/experiments/three_layer/submit_full.sh
#
# Smoke test:
#   EXP_TOPOLOGY_COUNT=2 EXP_RF_SEEDS=2 EXP_OVERWRITE_MANIFEST=1 FORCE_RERUN=1 \
#     bash neural-operator/experiments/three_layer/submit_full.sh smoke
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-production}"

if [ "${MODE}" = "smoke" ]; then
  EXP_TOPOLOGY_COUNT="${EXP_TOPOLOGY_COUNT:-2}" \
  EXP_RF_SEEDS="${EXP_RF_SEEDS:-2}" \
  EXP_OVERWRITE_MANIFEST="${EXP_OVERWRITE_MANIFEST:-1}" \
  FORCE_RERUN="${FORCE_RERUN:-1}" \
  sbatch -N 1 --ntasks-per-node=4 -t 4:00:00 \
    "${SCRIPT_DIR}/stampede3_full_run.slurm"
elif [ "${MODE}" = "production" ]; then
  sbatch -N 6 --ntasks-per-node=48 -t 48:00:00 \
    "${SCRIPT_DIR}/stampede3_full_run.slurm"
else
  echo "Usage: $0 [smoke|production]" >&2
  exit 2
fi
