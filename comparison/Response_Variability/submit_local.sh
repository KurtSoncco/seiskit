#!/bin/bash
# Quick validation: one seed per method (indices 0,10,20,30,40). Full smoke = 50 cases.
# Usage: ./submit_local.sh [max_parallel]

set -euo pipefail
cd "$(dirname "$0")"
MAX_JOBS="${1:-2}"
export RV_SMOKE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p results/h5 logs

echo "Quick validation (5 methods x 1 seed). 2D cases ~5 min each on laptop."

for idx in 0 10 20 30 40; do
  echo "=== index $idx ==="
  python run_experiment.py --index "$idx" --force 2>&1 | tee "logs/smoke_${idx}.log" | tail -3
done

python analyze_response.py
python plot_comparison.py
echo "Validation complete."
