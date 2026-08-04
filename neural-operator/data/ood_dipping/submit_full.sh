#!/bin/bash
# Submit ood_dipping on Stampede3: smoke | production
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-production}"

if [ "${MODE}" = "smoke" ]; then
  # Dipping cases are deeper (H≥25 m + dip drop); 4 h was short for idx 0/1 (~1–2k s left).
  OOD_PHYSICS_COUNT="${OOD_PHYSICS_COUNT:-2}" \
  OOD_SEED_LEVELS="${OOD_SEED_LEVELS:-2}" \
  OOD_OVERWRITE_MANIFEST="${OOD_OVERWRITE_MANIFEST:-1}" \
  FORCE_RERUN="${FORCE_RERUN:-1}" \
  sbatch -N 1 --ntasks-per-node=4 -t 12:00:00 \
    "${SCRIPT_DIR}/stampede3_full_run.slurm"
elif [ "${MODE}" = "production" ]; then
  # Extra walltime vs flat 3-layer: dipping Lz grows with |dip|.
  sbatch -N 4 --ntasks-per-node=48 -t 48:00:00 \
    "${SCRIPT_DIR}/stampede3_full_run.slurm"
else
  echo "Usage: $0 [smoke|production]" >&2
  exit 2
fi
