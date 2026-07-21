#!/bin/bash
# Run only missing grf_2d indices (skips existing H5s).
# Usage:
#   ./run_missing_grf.sh                  # refresh list + run, jobs=4
#   JOBS=8 ./run_missing_grf.sh
#   ./run_missing_grf.sh --list-only      # write missing list, do not run
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../.. && pwd)"
SURROGATE_PYTHON="${SURROGATE_PYTHON:-$HOME/surrogate-seismic-waves/.venv/bin/python}"
OPENSEES_PYTHON="${OPENSEES_PYTHON:-$ROOT/.venv/bin/python}"
LIST_ONLY=0
JOBS="${JOBS:-4}"
LIST_FILE="${LIST_FILE:-results/missing_grf_indices.txt}"

if [ "${1:-}" = "--list-only" ]; then
  LIST_ONLY=1
fi

export RV_SMOKE=0
export RV_SMOKE_2D=1
export RV_USE_SURROGATE_2D=1
export RV_RF_N_SEEDS="${RV_RF_N_SEEDS:-40}"
export GIFNO_DEVICE="${GIFNO_DEVICE:-cuda}"
export GIFNO_MODEL_DIR="${GIFNO_MODEL_DIR:-$HOME/surrogate-seismic-waves/checkpoints/xt_lat128_d128}"
export GIFNO_SURROGATE_ROOT="${GIFNO_SURROGATE_ROOT:-$HOME/surrogate-seismic-waves/experiments/GIFNO-FDO-XT}"
export GIFNO_LATENT_CHANNELS=128
export GIFNO_DEEPONET_LATENT_DIM=128
export GIFNO_NUM_FNO_LAYERS=5
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p results/h5 logs

"$OPENSEES_PYTHON" - <<'PY'
from pathlib import Path
from manifest import index_to_params, total_combinations
h5 = Path("results/h5")
miss = [
    i for i in range(total_combinations())
    if index_to_params(i).method == "grf_2d" and not (h5 / f"run_{i}.h5").exists()
]
Path("results/missing_grf_indices.txt").write_text("\n".join(map(str, miss)) + ("\n" if miss else ""))
print(f"missing grf_2d: {len(miss)}")
PY

N=$(grep -cve '^$' "$LIST_FILE" || true)
echo "GIFNO_DEVICE=${GIFNO_DEVICE}  MODEL=${GIFNO_MODEL_DIR}  JOBS=${JOBS}  N=${N}"
"$SURROGATE_PYTHON" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

if [ "$LIST_ONLY" = "1" ] || [ "$N" = "0" ]; then
  exit 0
fi

export ROOT SURROGATE_PYTHON
xargs -P "$JOBS" -a "$LIST_FILE" -I{} bash -c '
  idx="$1"
  echo "=== index ${idx} ==="
  "$SURROGATE_PYTHON" -u run_experiment.py --index "$idx" > "logs/smoke_${idx}.log" 2>&1
  tail -3 "logs/smoke_${idx}.log"
' _ {}

echo "DONE missing grf_2d"
