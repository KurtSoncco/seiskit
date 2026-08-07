#!/usr/bin/env bash
# Lean full-paper pipeline: skips OOM-prone / already-trained heavy steps by default.
#
# Env knobs:
#   FULL_PAPER_N_JOBS=4          # joblib workers (default: all cores)
#   RUN_ALL_SEEDS=1              # include qualitative all_seeds_all_nodes (heavy)
#   FORCE_RETRAIN=1              # re-run train_qbm / train_ngboost / train_sr
#   RUN_SR=1                     # include symbolic regression
#
# Log: statistical_analysis/full_paper/pipeline_run.log
set -uo pipefail

ROOT="/home/kurt-/seiskit/statistical_analysis/full_paper"
PY="/home/kurt-/seiskit/.venv/bin/python"
LOG="$ROOT/pipeline_run.log"
CODE="$ROOT/analysis/code"
BOX_FIG="/mnt/box/GIG Lab - UC Berkeley/Projects/Statistical Analysis/complete/full_paper/figures"
FAILED=0

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

run() {
  local label="$1"
  shift
  echo ""
  echo "======== $(date -Is) | START: $label ========"
  echo "+ $*"
  if (cd "$ROOT" && "$@"); then
    echo "======== $(date -Is) | DONE:  $label ========"
  else
    local rc=$?
    echo "======== $(date -Is) | FAIL($rc): $label ========"
    FAILED=$((FAILED + 1))
  fi
}

skip() {
  echo ""
  echo "======== $(date -Is) | SKIP: $1 ($2) ========"
}

have_models() {
  local dir="$1"
  local n_min="$2"
  local n
  n=$(find "$dir" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)
  [[ "$n" -ge "$n_min" ]]
}

echo "Pipeline start $(date -Is)"
echo "Log: $LOG"
echo "N_JOBS=${FULL_PAPER_N_JOBS:--1}  RUN_ALL_SEEDS=${RUN_ALL_SEEDS:-0}  FORCE_RETRAIN=${FORCE_RETRAIN:-0}  RUN_SR=${RUN_SR:-0}"

# 1. Appendix
run "appendix_im" "$PY" analysis/appendix_im/plot_peak_stability.py

# 2. Qualitative TF — skip all_seeds by default (was OOM-killing the previous run)
run "qualitative_center" "$PY" analysis/qualitative/run_tf.py --mode center_node_all_seeds
run "qualitative_one_seed" "$PY" analysis/qualitative/run_tf.py --mode one_seed_all_nodes
if [[ "${RUN_ALL_SEEDS:-0}" == "1" ]]; then
  run "qualitative_all_seeds" "$PY" analysis/qualitative/run_tf.py --mode all_seeds_all_nodes
else
  skip "qualitative_all_seeds" "set RUN_ALL_SEEDS=1 to enable; capped plot curves if enabled"
fi

# 3. chi_variables
run "central_variability" "$PY" "$CODE/chi_variables/central_variability.py"
run "mean_variance_adequacy" "$PY" "$CODE/chi_variables/mean_variance_adequacy.py"
run "metric_heteroscedasticity" "$PY" "$CODE/chi_variables/metric_heteroscedasticity.py"
run "node_seed_iid" "$PY" "$CODE/chi_variables/node_seed_iid.py"
run "node_ratio_normality" "$PY" "$CODE/chi_variables/node_ratio_normality.py"
run "distribution_histograms" "$PY" "$CODE/chi_variables/distribution_histograms.py"
run "factor_violins" "$PY" "$CODE/chi_variables/factor_violins.py"
run "central_profiles" "$PY" "$CODE/chi_variables/central_profiles.py"
run "geomean_factor_cross" "$PY" "$CODE/chi_variables/geomean_factor_cross.py"
run "variability_plots" "$PY" "$CODE/chi_variables/variability_plots.py"
run "variance_heatmaps" "$PY" "$CODE/chi_variables/variance_heatmaps.py"

# 4. Spatial
run "spatial_acf" "$PY" "$CODE/chi_spatial/spatial_acf.py"
run "plot_spatial_acf" "$PY" "$CODE/chi_spatial/plot_spatial_acf.py"
run "spatial_coherence" "$PY" "$CODE/chi_spatial/spatial_coherence.py"

# 5. OLS
run "stage1_mean_ols" "$PY" "$CODE/chi_ols/stage1_mean_ols.py"
run "r2_ceiling" "$PY" "$CODE/chi_ols/r2_ceiling.py"
run "naive_vs_hetero" "$PY" "$CODE/chi_ols/naive_vs_hetero.py"
run "spatial_in_ols" "$PY" "$CODE/chi_ols/spatial_in_ols.py"

# 6. QBM — skip retrain if pickles exist
if [[ "${FORCE_RETRAIN:-0}" == "1" ]] || ! have_models "$BOX_FIG/chi_qbm/models" 20; then
  run "train_qbm" "$PY" "$CODE/chi_qbm/train_qbm.py"
else
  skip "train_qbm" "models already on Box; FORCE_RETRAIN=1 to redo"
fi
run "spatial_ols_qbm" "$PY" "$CODE/chi_qbm/spatial_ols.py"
run "compare_models" "$PY" "$CODE/chi_qbm/compare_models.py"

# 7. NGBoost — skip retrain if pickles exist
if [[ "${FORCE_RETRAIN:-0}" == "1" ]] || ! have_models "$BOX_FIG/chi_ngboost/models" 5; then
  run "train_ngboost" "$PY" "$CODE/chi_ngboost/train_ngboost.py"
else
  skip "train_ngboost" "models already on Box; FORCE_RETRAIN=1 to redo"
fi
run "evaluate_ngboost" "$PY" "$CODE/chi_ngboost/evaluate_ngboost.py"
run "export_surfaces" "$PY" "$CODE/chi_ngboost/export_surfaces.py"
run "calibration_crps_pit" "$PY" "$CODE/chi_ngboost/calibration_crps_pit.py"
run "exceedance_friedman" "$PY" "$CODE/chi_ngboost/exceedance_friedman.py"

# 8. SHAP / ALE
run "shap_ngboost" "$PY" "$CODE/chi_shap/shap_ngboost.py"
run "shap_qbm" "$PY" "$CODE/chi_shap/shap_qbm.py"
run "shap_compare" "$PY" "$CODE/chi_shap/shap_compare.py"
run "ale_effects" "$PY" "$CODE/chi_shap/ale_effects.py"
run "shap_median_vs_tail" "$PY" "$CODE/chi_shap/shap_median_vs_tail.py"

# 9. SR optional
if [[ "${RUN_SR:-0}" == "1" ]]; then
  run "train_sr" "$PY" "$CODE/chi_sr/train_sr.py"
  run "evaluate_sr" "$PY" "$CODE/chi_sr/evaluate_sr.py"
else
  skip "chi_sr" "set RUN_SR=1 to enable"
fi

# 10. Manuscript figures
run "model_scheme" "$PY" figures/model_scheme.py
run "vs_rh_realizations" "$PY" figures/vs_rh_realizations.py
run "field_stat_recovery" "$PY" figures/field_stat_recovery.py
run "ricker_wave" "$PY" figures/descriptions/ricker_wave.py

echo ""
if [[ "$FAILED" -eq 0 ]]; then
  echo "Pipeline finished OK $(date -Is)"
else
  echo "Pipeline finished with $FAILED failure(s) $(date -Is)"
  exit 1
fi
