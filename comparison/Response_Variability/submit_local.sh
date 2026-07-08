#!/bin/bash
# Quick local validation before HPC submit.
# Default: one seed per method (indices 0,10,20,30,40).
# Usage:
#   ./submit_local.sh
#   ./submit_local.sh --analyze-only

set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../.. && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON:-python3}"
fi

export RV_SMOKE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p results/h5 logs

ANALYZE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --analyze-only) ANALYZE_ONLY=1 ;;
  esac
done

run_idx() {
  local idx="$1"
  echo "=== index ${idx} ==="
  "$PYTHON" -u run_experiment.py --index "$idx" 2>&1 | tee "logs/smoke_${idx}.log" | tail -5
}

if [ "$ANALYZE_ONLY" = "0" ]; then
  echo "Quick validation (5 methods × 1 seed). 2D cases ~5 min each."
  for idx in 0 10 20 30 40; do
    run_idx "$idx"
  done
fi

echo "=== analyze ==="
"$PYTHON" analyze_response.py --h5-dir results/h5 --out-dir results/analysis
echo "=== plot ==="
"$PYTHON" plot_comparison.py --h5-dir results/h5 --out-dir results/figures
echo "Local analysis complete."
echo "  CSV: results/analysis/method_comparison_summary.csv"
echo "  TF:  results/figures/af_method_subplots_Vs1230_M1.png"
echo "  Sa:  results/figures/method_subplots_Vs1230_M1.png"
echo "  Profiles: results/figures/hallal_profiles_Vs1230.png"
echo "  2D GRF:   results/figures/grf2d_explainability_Vs1230_s1.png"
