# Heteroscedastic Statistical Analysis of Seismic Transfer-Function Peak Ratios

Conference paper analysis: Monte Carlo random-field study of 1-D seismic
site-response simulations over stochastic, exponentially-correlated
shear-wave-velocity fields.

## Quick start

```bash
# From the seiskit workspace root:
uv sync --extra ml
uv run python statistical_analysis/conference_paper/run_all.py
```

## Data

Source: `tf_peak_ratios_mode0.h5` (group `master`) on the Box mount.
- 2,454,300 rows = 243 factorial cells x 101 channels x 100 seeds
- Design factors (3 levels each): Vs1, Height, CoV, rH, aHV
- Targets: `abs_TF_ratio`, `f_ratio`

All paths are configured in `config.py`.

## Layout

```
config.py              Centralized paths, constants, data loaders
run_all.py             Regenerate all results
models/                Saved LightGBM models (channel-50 and pooled 101-ch)
diagnostics/           EDA, normality, heteroscedasticity, baseline residuals
performance/           GBM performance, R2 ceiling, quantile 101-ch model
quantile/              Quantile regression coefficients, seed error
shap/                  SHAP summaries, interactions, tails, 101-ch
seed/                  Seed independence (ICC), seed adequacy
spatial/               101-channel spatial structure, variance decomposition
extrapolation/         Physics extrapolation, interp/extrap predictions
```

Results are written to `results/<topic>/{plots,data}/` under the Box mount.

## Dependencies

All scripts use `seiskit.plot_config` for publication-quality figures and
require the `[ml]` optional dependencies (`lightgbm`, `shap`, `scikit-learn`,
`statsmodels`).

## Script map

### Diagnostics
- `eda_channel50.py` — distributions, factor effects, seed structure
- `normality_assessment.py` — QQ plots: raw vs transformed targets
- `heteroscedasticity_diagnostics.py` — variance structure diagnosis
- `baseline_residual_diagnostics.py` — OLS baseline, variance partition

### Performance
- `gbm_performance.py` — LightGBM mean + quantile, pinball loss
- `model_r2_ceiling.py` — R2 ceiling decomposition, cell-mean predictability
- `r2_ceiling_diagnostics.py` — ceiling/efficiency CSV
- `quantile_101ch_performance.py` — pooled 101-channel quantile model

### Quantile
- `quantile_coefficients_abs_TF.py` — quantile regression for log(abs TF)
- `quantile_coefficients_f_ratio.py` — quantile regression for f_ratio
- `quantile_seed_error.py` — per-quantile Monte Carlo error

### SHAP
- `shap_summary_abs_TF.py` / `shap_summary_f_ratio.py` — SHAP summaries
- `shap_interactions.py` — interaction heatmap + dependence scatter
- `quantile_shap_101ch.py` — per-quantile SHAP across 101 channels
- `quantile_shap_interactions.py` / `quantile_shap_interactions_101ch.py`
- `quantile_shap_tails.py` — tail dependence plots

### Seed
- `seed_independence.py` — ICC, cell-cell correlation, design effect
- `seed_adequacy.py` — bootstrap convergence, required n

### Spatial
- `spatial_101ch_structure.py` — per-channel R2 ceiling, factor slope drift
- `spatial_variance_decomposition.py` — crossed variance components CSV

### Extrapolation
- `physics_extrapolation.py` — physics features, interp vs extrap R2
- `seed_variance_and_extrapolation.py` — split-half reproducibility
- `interp_extrap_predictions.py` — predicted-vs-actual scatter
