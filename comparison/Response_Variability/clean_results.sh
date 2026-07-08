#!/bin/bash
# Remove local run artifacts (old + new). Keeps .gitkeep placeholders.
set -euo pipefail
cd "$(dirname "$0")"

echo "Cleaning Response_Variability results and caches..."
rm -rf __pycache__ logs
rm -rf results/idx_*
find results/h5 -maxdepth 1 -name '*.h5' -delete 2>/dev/null || true
find results/figures -maxdepth 1 -name '*.png' -delete 2>/dev/null || true
find results/analysis -maxdepth 1 -name '*.csv' -delete 2>/dev/null || true

mkdir -p logs results/h5 results/figures results/analysis
touch results/h5/.gitkeep results/figures/.gitkeep results/analysis/.gitkeep
echo "Done."
