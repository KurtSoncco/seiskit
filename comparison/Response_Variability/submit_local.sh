#!/bin/bash
# Quick local validation before HPC submit.
# Default: 1D Hallal only — indices 0,10,20 @ Sobol #0 (no 2D).
# Full smoke (120 indices): ./submit_local.sh --all
# Include 2D: ./submit_local.sh --2d   or   ./submit_local.sh --all --2d
# Seed counts: ./submit_local.sh --all --2d --n-seeds 20 --hallal-seeds 50
# Local GIFNO production (Stampede does Hallal+Pretell):
#   ./submit_local.sh --full --grf-only --n-seeds 40 --jobs 4
# Usage:
#   ./submit_local.sh
#   ./submit_local.sh --analyze-only
#   ./submit_local.sh --all --2d --n-seeds 20 --hallal-seeds 50
#   ./submit_local.sh --full --grf-only --n-seeds 40 --jobs 4

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

# Smoke by default; --full or RV_SMOKE=0 for production (64 Sobol).
export RV_SMOKE="${RV_SMOKE:-1}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p results/h5 logs

ANALYZE_ONLY=0
RUN_ALL=0
INCLUDE_2D=0
GRF_ONLY=0
NO_ANALYZE=0
N_SEEDS=""
HALLAL_SEEDS=""
JOBS="${JOBS:-1}"

while [ $# -gt 0 ]; do
  case "$1" in
    --analyze-only) ANALYZE_ONLY=1 ;;
    --no-analyze) NO_ANALYZE=1 ;;
    --all) RUN_ALL=1 ;;
    --full) RUN_ALL=1; export RV_SMOKE=0 ;;
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
    --jobs)
      shift
      JOBS="${1:?--jobs requires a positive integer}"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--full] [--all] [--2d] [--grf-only] [--n-seeds N] [--hallal-seeds N] [--jobs N] [--no-analyze] [--analyze-only]" >&2
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
    echo "RF seeds: 1..${N_SEEDS} (grf_2d + pretell + opensees_2d paired)"
  fi
  export GIFNO_DATA_ROOT="${GIFNO_DATA_ROOT:-${BOX_DATA}}"
  export GIFNO_H5_DIR="${GIFNO_H5_DIR:-${GIFNO_DATA_ROOT}/h5}"
  export GIFNO_TF_DIR="${GIFNO_TF_DIR:-${GIFNO_DATA_ROOT}/transfer_function}"
  export GIFNO_SURROGATE_ROOT="${GIFNO_SURROGATE_ROOT:-${DEFAULT_SURROGATE_ROOT}}"
  export GIFNO_MODEL_DIR="${GIFNO_MODEL_DIR:-${DEFAULT_MODEL_DIR}}"
  export GIFNO_LATENT_CHANNELS="${GIFNO_LATENT_CHANNELS:-128}"
  export GIFNO_DEEPONET_LATENT_DIM="${GIFNO_DEEPONET_LATENT_DIM:-128}"
  export GIFNO_NUM_FNO_LAYERS="${GIFNO_NUM_FNO_LAYERS:-5}"
  # Prefer CUDA when available (override with GIFNO_DEVICE=cpu).
  if [ -z "${GIFNO_DEVICE:-}" ]; then
    if "$SURROGATE_PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      export GIFNO_DEVICE=cuda
    else
      export GIFNO_DEVICE=cpu
    fi
  else
    export GIFNO_DEVICE
  fi
  if [ "$RV_USE_SURROGATE_2D" = "1" ] && [ -x "$SURROGATE_PYTHON" ]; then
    echo "2D: grf_2d -> GIFNO; pretell/opensees_2d/Hallal -> OpenSees ($OPENSEES_PYTHON)"
    echo "    GIFNO_MODEL_DIR=${GIFNO_MODEL_DIR}"
    echo "    GIFNO_SURROGATE_ROOT=${GIFNO_SURROGATE_ROOT}"
    echo "    GIFNO_DEVICE=${GIFNO_DEVICE}"
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

# Fast grf_2d index list (avoids scanning the full Hallal block).
grf_indices() {
  "$PYTHON" -c "
from manifest import hallal_block_size, active_rf_seeds, active_sobol_count, RF_METHODS
hb = hallal_block_size()
n = len(active_rf_seeds())
n_methods = len(RF_METHODS)
grf_slot = RF_METHODS.index('grf_2d')
for s in range(active_sobol_count()):
    base = hb + s * (n_methods * n) + grf_slot * n
    for i in range(n):
        print(base + i)
"
}

rf_quick_indices() {
  "$PYTHON" -c "
from manifest import hallal_block_size, active_rf_seeds, RF_METHODS
hb = hallal_block_size()
n = len(active_rf_seeds())
n_methods = len(RF_METHODS)
# Print one line per RF method (grf_2d, pretell, opensees_2d, …)
for m_i, _method in enumerate(RF_METHODS):
    print(' '.join(str(hb + m_i * n + s) for s in range(n)))
"
}

run_indices_parallel() {
  local -a idxs=("$@")
  local n="${#idxs[@]}"
  echo "Running ${n} indices with --jobs ${JOBS} (RV_SMOKE=${RV_SMOKE})."
  if [ "$JOBS" -le 1 ]; then
    local idx
    for idx in "${idxs[@]}"; do
      run_idx "$idx"
    done
    return
  fi
  # Export helpers for subshells spawned by xargs.
  export ROOT SURROGATE_PYTHON OPENSEES_PYTHON
  export RV_SMOKE RV_SMOKE_2D RV_USE_SURROGATE_2D RV_RF_N_SEEDS RV_HALLAL_N_SEEDS
  export GIFNO_DATA_ROOT GIFNO_H5_DIR GIFNO_TF_DIR GIFNO_SURROGATE_ROOT GIFNO_MODEL_DIR
  export GIFNO_LATENT_CHANNELS GIFNO_DEEPONET_LATENT_DIM GIFNO_NUM_FNO_LAYERS GIFNO_DEVICE
  export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
  printf '%s\n' "${idxs[@]}" | xargs -P "$JOBS" -I{} bash -c '
    idx="$1"
    method=$("$OPENSEES_PYTHON" -c "from manifest import index_to_params; print(index_to_params(int(\"$idx\")).method)")
    if [ "${RV_USE_SURROGATE_2D:-0}" = "1" ] && [ -x "$SURROGATE_PYTHON" ] && [ "$method" = "grf_2d" ]; then
      py="$SURROGATE_PYTHON"
      export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    else
      py="$OPENSEES_PYTHON"
    fi
    echo "=== index ${idx} ==="
    "$py" -u run_experiment.py --index "$idx" > "logs/smoke_${idx}.log" 2>&1
    tail -3 "logs/smoke_${idx}.log"
  ' _ {}
}

if [ "$ANALYZE_ONLY" = "0" ]; then
  if [ "$GRF_ONLY" = "1" ]; then
    mapfile -t INDICES < <(grf_indices)
    echo "Local GIFNO-only: ${#INDICES[@]} grf_2d indices (RV_SMOKE=${RV_SMOKE}, rf_seeds=${RV_RF_N_SEEDS:-default})."
    run_indices_parallel "${INDICES[@]}"
  elif [ "$RUN_ALL" = "1" ]; then
    TOTAL=$("$PYTHON" -c "from manifest import total_combinations; print(total_combinations())")
    echo "Full run: ${TOTAL} indices (RV_SMOKE=${RV_SMOKE}, 2D=${INCLUDE_2D}, hallal_seeds=${RV_HALLAL_N_SEEDS:-default}, rf_seeds=${RV_RF_N_SEEDS:-default})."
    mapfile -t INDICES < <(seq 0 $((TOTAL - 1)))
    run_indices_parallel "${INDICES[@]}"
  else
    echo "Quick validation (3× 1D Hallal @ Sobol #0). Pass --2d for grf_2d + pretell."
    INDICES=(0 10 20)
    if [ "$INCLUDE_2D" = "1" ]; then
      mapfile -t RF_LINES < <(rf_quick_indices)
      for line in "${RF_LINES[@]}"; do
        # shellcheck disable=SC2206
        RF_IDX=($line)
        INDICES+=( "${RF_IDX[@]}" )
      done
    fi
    run_indices_parallel "${INDICES[@]}"
  fi
fi

if [ "$NO_ANALYZE" = "1" ]; then
  echo "Skipping analyze/plot (--no-analyze). H5s in results/h5/."
  exit 0
fi

echo "=== analyze ==="
"$PYTHON" analyze_response.py --h5-dir results/h5 --out-dir results/analysis
echo "=== plot ==="
"$PYTHON" plot_comparison.py --h5-dir results/h5 --out-dir results/figures \
  --analysis-dir results/analysis --sobol-ids 19,37,36,10,44
echo "Local analysis complete."
echo "  CSV: results/analysis/method_comparison_summary.csv"
echo "  Band misfit: results/analysis/tf_band_misfit_vs_opensees.csv"
echo "  Panels: results/figures/profile_tf_panel_sobol*_M1.png"
echo "  Peak bias: results/figures/tf_peak_*_all_sobol.png"
echo "  Band misfit fig: results/figures/tf_band_misfit_all_sobol.png"
echo "  Error vs Sobol: results/figures/tf_error_vs_sobol_params.png"
if [ "$INCLUDE_2D" = "1" ]; then
  echo "  (2D H5s used inside profile/TF panels)"
fi
