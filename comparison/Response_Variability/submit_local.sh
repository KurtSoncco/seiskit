#!/bin/bash
# Quick local validation before HPC submit.
# Default: one seed per method at Sobol #0 (indices 0,10,20,120,125).
# Full smoke (160 indices): RV_SMOKE=1 ./submit_local.sh --all
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
RUN_ALL=0
for arg in "$@"; do
  case "$arg" in
    --analyze-only) ANALYZE_ONLY=1 ;;
    --all) RUN_ALL=1 ;;
  esac
done

run_idx() {
  local idx="$1"
  echo "=== index ${idx} ==="
  "$PYTHON" -u run_experiment.py --index "$idx" 2>&1 | tee "logs/smoke_${idx}.log" | tail -5
}

if [ "$ANALYZE_ONLY" = "0" ]; then
  if [ "$RUN_ALL" = "1" ]; then
    TOTAL=$("$PYTHON" -c "from manifest import total_combinations; print(total_combinations())")
    echo "Full smoke run: ${TOTAL} indices (RV_SMOKE=1)."
    for idx in $(seq 0 $((TOTAL - 1))); do
      run_idx "$idx"
    done
  else
    echo "Quick validation (5 methods × 1 seed @ Sobol #0). 2D cases ~5 min each."
    for idx in 0 10 20 120 125; do
      run_idx "$idx"
    done
  fi
fi

echo "=== analyze ==="
"$PYTHON" analyze_response.py --h5-dir results/h5 --out-dir results/analysis
echo "=== plot ==="
"$PYTHON" plot_comparison.py --h5-dir results/h5 --out-dir results/figures --sobol-id 0
echo "Local analysis complete."
echo "  CSV: results/analysis/method_comparison_summary.csv"
echo "  TF:  results/figures/tf_methods_sobol00_M1.png"
echo "  AF:  results/figures/af_method_subplots_sobol00_M1.png"
echo "  Sa:  results/figures/method_subplots_sobol00_M1.png"
echo "  Profiles: results/figures/hallal_profiles_sobol00.png"
echo "  2D GRF:   results/figures/grf2d_explainability_Vs*_s1.png"
