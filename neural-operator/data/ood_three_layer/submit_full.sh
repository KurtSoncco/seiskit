#!/bin/bash
# Submit ood_three_layer on Stampede3: smoke | production
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-production}"

if [ "${MODE}" = "smoke" ]; then
  # Max Lz ≈ 34 m (H1+H2≤24 + 10 m bedrock). At ~50 m OpenSees is ~3–4 h,
  # so budget 8 h smoke walltime with margin for the slowest of 4 concurrent tasks.
  OOD_PHYSICS_COUNT="${OOD_PHYSICS_COUNT:-2}" \
  OOD_SEED_LEVELS="${OOD_SEED_LEVELS:-2}" \
  OOD_OVERWRITE_MANIFEST="${OOD_OVERWRITE_MANIFEST:-1}" \
  FORCE_RERUN="${FORCE_RERUN:-1}" \
  sbatch -N 1 --ntasks-per-node=4 -t 08:00:00 \
    "${SCRIPT_DIR}/stampede3_full_run.slurm"
elif [ "${MODE}" = "production" ]; then
  # ~3 h/case × ~5 launcher waves on 4×48 cores → well under 48 h.
  sbatch -N 4 --ntasks-per-node=48 -t 48:00:00 \
    "${SCRIPT_DIR}/stampede3_full_run.slurm"
else
  echo "Usage: $0 [smoke|production]" >&2
  exit 2
fi
