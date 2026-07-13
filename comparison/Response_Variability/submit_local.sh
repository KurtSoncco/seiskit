#!/bin/bash
# Quick local validation before HPC submit.
# Default: 1D Hallal only — indices 0,10,20 @ Sobol #0 (no 2D).
# Full smoke (120 indices): ./submit_local.sh --all
# Include 2D: ./submit_local.sh --2d   or   ./submit_local.sh --all --2d
# Seed counts: ./submit_local.sh --all --2d --n-seeds 20 --hallal-seeds 50
# Local GIFNO only (Stampede does Hallal+Pretell): ./submit_local.sh --all --2d --grf-only --n-seeds 40
# Usage:
#   ./submit_local.sh
#   ./submit_local.sh --analyze-only
#   ./submit_local.sh --all --2d --n-seeds 20 --hallal-seeds 50
#   ./submit_local.sh --all --2d --grf-only --n-seeds 40

set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../.. && pwd)"
OPENSEES_PYTHON="${ROOT}/.venv/bin/python"
if [ ! -x "$OPENSEES_PYTHON" ]; then
  OPENSEES_PYTHON="${PYTHON:-python3}"
fi
PYTHON="$OPENSEES_PYTHON"
SURROGATE_PYTHON="${SURROGATE_PYTHON:-$HOME/surrogate-seismic-waves/.venv/bin/python}"

BOX_DATA="/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data"
DEFAULT_MODEL_DIR="${HOME}/surrogate-seismic-waves/checkpoints/xt_lat128_d128"
DEFAULT_SURROGATE_ROOT="${HOME}/surrogate-seismic-waves/experiments/GIFNO-FDO-XT"

export RV_SMOKE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p results/h5 logs

ANALYZE_ONLY=0
RUN_ALL=0
INCLUDE_2D=0
GRF_ONLY=0
N_SEEDS=""
HALLAL_SEEDS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --analyze-only) ANALYZE_ONLY=1 ;;
    --all) RUN_ALL=1 ;;
    --2d) INCLUDE_2D=1 ;;
    --grf-only) GRF_ONLY=1; INCLUDE_2D=1 ;;
    --n-seeds)
      shift
      N_SEEDS="${1:?--n-seeds requires a positive integer}"
      ;;
    --hallal-seeds)
      shift
      HALLAL_SEEDS="${1:?--hallal-seeds requires a positive integer}"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--all] [--2d] [--grf-only] [--n-seeds N] [--hallal-seeds N] [--analyze-only]" >&2
      exit 1
      ;;
  esac
  shift
done

if [ -n "$HALLAL_SEEDS" ]; then
  export RV_HALLAL_N_SEEDS="$HALLAL_SEEDS"
  echo "Hallal 1D seeds: 1..${HALLAL_SEEDS} (hallal_vs + hallal_tts)"
fi

if [ "$INCLUDE_2D" = "1" ]; then
  export RV_SMOKE_2D=1
  export RV_USE_SURROGATE_2D="${RV_USE_SURROGATE_2D:-1}"
  if [ -n "$N_SEEDS" ]; then
    export RV_RF_N_SEEDS="$N_SEEDS"
    echo "RF seeds: 1..${N_SEEDS} (grf_2d + pretell paired)"
  fi
  export GIFNO_DATA_ROOT="${GIFNO_DATA_ROOT:-${BOX_DATA}}"
  export GIFNO_H5_DIR="${GIFNO_H5_DIR:-${GIFNO_DATA_ROOT}/h5}"
  export GIFNO_TF_DIR="${GIFNO_TF_DIR:-${GIFNO_DATA_ROOT}/transfer_function}"
  export GIFNO_SURROGATE_ROOT="${GIFNO_SURROGATE_ROOT:-${DEFAULT_SURROGATE_ROOT}}"
  export GIFNO_MODEL_DIR="${GIFNO_MODEL_DIR:-${DEFAULT_MODEL_DIR}}"
  export GIFNO_LATENT_CHANNELS="${GIFNO_LATENT_CHANNELS:-128}"
  export GIFNO_DEEPONET_LATENT_DIM="${GIFNO_DEEPONET_LATENT_DIM:-128}"
  export GIFNO_NUM_FNO_LAYERS="${GIFNO_NUM_FNO_LAYERS:-5}"
  if [ "$RV_USE_SURROGATE_2D" = "1" ] && [ -x "$SURROGATE_PYTHON" ]; then
    echo "2D: grf_2d -> GIFNO-FDO-XT ($SURROGATE_PYTHON); pretell/Hallal -> OpenSees ($OPENSEES_PYTHON)"
    echo "    GIFNO_MODEL_DIR=${GIFNO_MODEL_DIR}"
    echo "    GIFNO_SURROGATE_ROOT=${GIFNO_SURROGATE_ROOT}"
  elif [ "$RV_USE_SURROGATE_2D" = "1" ]; then
    echo "WARNING: surrogate venv not found at $SURROGATE_PYTHON; grf_2d may fail without torch."
  fi
fi

python_for_index() {
  local idx="$1"
  if [ "${RV_USE_SURROGATE_2D:-0}" = "1" ] && [ -x "$SURROGATE_PYTHON" ]; then
    local method
    method=$("$OPENSEES_PYTHON" -c "from manifest import index_to_params; print(index_to_params($idx).method)")
    if [ "$method" = "grf_2d" ]; then
      echo "$SURROGATE_PYTHON"
      return
    fi
  fi
  echo "$OPENSEES_PYTHON"
}

run_idx() {
  local idx="$1"
  local run_python
  run_python="$(python_for_index "$idx")"
  echo "=== index ${idx} (python=$(basename "$(dirname "$run_python")")) ==="
  if [ "$run_python" = "$SURROGATE_PYTHON" ]; then
    PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
      "$run_python" -u run_experiment.py --index "$idx" 2>&1 | tee "logs/smoke_${idx}.log" | tail -5
  else
    "$run_python" -u run_experiment.py --index "$idx" 2>&1 | tee "logs/smoke_${idx}.log" | tail -5
  fi
}

rf_quick_indices() {
  "$PYTHON" -c "
from manifest import hallal_block_size, active_rf_seeds
hb = hallal_block_size()
n = len(active_rf_seeds())
# grf_2d seeds 0..n-1, then pretell seeds 0..n-1
print(' '.join(str(hb + s) for s in range(n)))
print(' '.join(str(hb + n + s) for s in range(n)))
"
}

if [ "$ANALYZE_ONLY" = "0" ]; then
  if [ "$GRF_ONLY" = "1" ]; then
    mapfile -t INDICES < <("$PYTHON" -c "from manifest import indices_for_methods; print('\n'.join(map(str, indices_for_methods(['grf_2d']))))")
    echo "Local GIFNO-only: ${#INDICES[@]} grf_2d indices (rf_seeds=${RV_RF_N_SEEDS:-default})."
    for idx in "${INDICES[@]}"; do
      run_idx "$idx"
    done
  elif [ "$RUN_ALL" = "1" ]; then
    TOTAL=$("$PYTHON" -c "from manifest import total_combinations; print(total_combinations())")
    echo "Full smoke run: ${TOTAL} indices (RV_SMOKE=1, 2D=${INCLUDE_2D}, hallal_seeds=${RV_HALLAL_N_SEEDS:-default}, rf_seeds=${RV_RF_N_SEEDS:-default})."
    for idx in $(seq 0 $((TOTAL - 1))); do
      run_idx "$idx"
    done
  else
    echo "Quick validation (3× 1D Hallal @ Sobol #0). Pass --2d for grf_2d + pretell."
    INDICES=(0 10 20)
    if [ "$INCLUDE_2D" = "1" ]; then
      mapfile -t RF_LINES < <(rf_quick_indices)
      GRF_INDICES=(${RF_LINES[0]})
      DEL_INDICES=(${RF_LINES[1]})
      INDICES+=( "${GRF_INDICES[@]}" "${DEL_INDICES[@]}" )
    fi
    for idx in "${INDICES[@]}"; do
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
echo "  Geomean CSV: results/analysis/method_comparison_geomean_summary.csv"
echo "  TF:  results/figures/tf_methods_sobol00_M1.png"
echo "  Geomean: results/figures/tf_grf2d_vs_pretell_geomean_sobol00_M1.png"
echo "  AF:  results/figures/af_method_subplots_sobol00_M1.png"
echo "  Sa:  results/figures/method_subplots_sobol00_M1.png"
echo "  Profiles: results/figures/hallal_profiles_sobol00.png"
if [ "$INCLUDE_2D" = "1" ]; then
  echo "  2D GRF:   results/figures/grf2d_explainability_Vs*_s1.png"
fi
