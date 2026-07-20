# Center-recorder analysis of seismic transfer-function peak ratios

Conference-paper workflow for a random-field study of 1-D seismic
site-response simulations.
**All analyses use the center recorder (channel 50) only.**

## Roles of the models (important)

These tools answer **different questions**. Do not rank them as “better / worse.”

| Tool | Estimates | Use for |
|------|-----------|---------|
| **OLS / MixedLM** | Linear conditional mean + SEs | Failure baseline: shows seed clustering, non-normal residuals, and that one slope cannot describe heteroscedastic effects |
| **Mean GBM** | Nonlinear conditional mean \(E[Y\mid X]\) | Predictive benchmarks: R², MSE, MAE, R² ceiling |
| **QBM** | Conditional quantiles \(q_\tau(Y\mid X)\) | Scientific explanation of spread/tails: pinball loss, calibration, SHAP/PDP by \(\tau\) |

**QBM is not “better than GBM.”** GBM is the mean-prediction benchmark.
QBM is the distributional model used with SHAP/PDP to show how inputs reshape
the full response, including effects that change across quantiles.

### R² ceiling (why test R² looks “low”)

Observation-level R² is limited by **seed-to-seed realization noise** that no
design-factor model can remove. The **R² ceiling** is the fraction of variance
that lives *between design cells* (deterministic signal). Efficiency is

\[
\text{efficiency} = \frac{\text{GBM test } R^2}{\text{R}^2\text{ ceiling}}.
\]

Example (`log_abs`): ceiling ≈ 0.40, GBM R² ≈ 0.35 → **~87% of the explainable
design signal** is captured even though absolute R² is only ~0.35.

### Raw vs natural-log amplitude

- **Native metrics** (fit on `abs_TF_ratio` vs fit on `log_abs`) are not
  comparable in MSE/MAE because units differ.
- **Fair comparison** scores both models on raw `|TF|_0^N`, back-transforming
  the log model with `exp` + Duan smearing.
- Logging mainly helps **symmetry / QBM calibration / interpretability**; it
  need not win every raw-scale mean metric.

### Friedman H² (interaction strength)

For features \(j,k\), H² measures how much of the joint PDP cannot be written
as an additive combination of the two main-effect PDPs. After centering,
\(H^2\in[0,1]\): near 0 ≈ additive; near 1 ≈ strong interaction.
(Previous uncentered code could exceed 1 and should not be published.)

## Narrative

1. Center-recorder peak ratios and factor structure.
2. Heteroscedasticity + seed clustering → mean-only OLS is insufficient.
3. Mean GBM + R² ceiling: how much design signal is predictable.
4. Raw vs ln amplitude: transform sensitivity on a fair raw scale.
5. **QBM + xAI** (primary scientific section): quantile effects, SHAP
   importance, signed directionality, PDPs, interactions.
6. Seed adequacy and extrapolation limits.

## Quick start

```bash
uv sync --extra ml
uv run python statistical_analysis/conference_paper/run_all.py
# reuse existing seed-split models/:
uv run python statistical_analysis/conference_paper/run_all.py --skip-train
# cell-split models for quantile_shape_cell.py:
uv run python statistical_analysis/conference_paper/quantile/quantile_channel_model.py --split cell
```

## Data

- HDF5 group `master` on Box (see `config.py`)
- Analysis table: 24,300 rows via `load_channel50()`
- Factors: `Vs1`, `Height`, `CoV`, `rH`, `aHV`
- Targets: `abs_TF_ratio`, `log_abs=ln(abs_TF_ratio)`, `f_ratio`

## Layout

```
config.py           Paths, loaders, Paul Tol colors, split-specific models
run_all.py          Dependency-aware regeneration
models/             lgbm_*_{seed|cell}.pkl
diagnostics/        EDA, normality, heteroscedasticity, OLS residuals
quantile/           QBM training, QR coefficients, seed error
performance/        Mean GBM metrics, R² ceiling, QBM calibration
shap/               Mean/quantile SHAP, PDP, H²
seed/               ICC / adequacy
extrapolation/      Physics features, interp/extrap limits
results/<topic>/    Local plots/ and data/
```

## Factor colors (Paul Tol Bright)

| Factor | Hex |
|--------|-----|
| Vs1 | `#4477AA` |
| Height | `#EE6677` |
| CoV | `#228833` |
| rH | `#CCBB44` |
| aHV | `#66CCEE` |

Reserved: purple `#AA3377`, gray `#BBBBBB`.

## Script map

### Diagnostics
- `eda_channel50.py`, `quantile_eda.py`, `normality_assessment.py`
- `heteroscedasticity_diagnostics.py`, `baseline_residual_diagnostics.py`

### Quantile / QBM
- `quantile_channel_model.py` — trains mean + quantile models
- `quantile_coefficients_abs_TF.py` / `_f_ratio.py` — QR curves vs OLS mean
- `quantile_seed_error.py` — Monte Carlo error at n=100

### Performance
- `gbm_performance.py` — mean GBM R²/MSE/MAE + QBM pinball/calibration +
  `model_metrics.csv` (native + fair raw-amplitude comparison)
- `model_r2_ceiling.py` — ceiling / efficiency / PI coverage

### SHAP / xAI
- `shap_seed_suite.py` — **QBM xAI suite** by default (plots for quantiles /
  PDP / H² only). Mean-GBM SHAP tables are still written to CSV; add
  ``--plot-mean`` if you want mean-GBM plots too.
- `shap_summary.py`, `shap_interactions.py`, `quantile_shap_tails.py`,
  `quantile_shap_interactions.py` — older single-purpose scripts (still usable)
- `quantile_shape_cell.py` — cell-split beeswarm / PDP / H²
  (requires `*_cell.pkl` models)

Re-run the suite without recomputing SHAP::

    uv run python statistical_analysis/conference_paper/shap/shap_seed_suite.py

Force SHAP recomputation::

    uv run python statistical_analysis/conference_paper/shap/shap_seed_suite.py --force-shap

### Seed & extrapolation
- `seed_independence.py`, `seed_adequacy.py`
- `physics_extrapolation.py`, `interp_extrap_predictions.py`,
  `seed_variance_and_extrapolation.py`

## Model artifacts (split-specific)

```
lgbm_mean_{target}_{seed|cell}.pkl
lgbm_q{05,10,25,50,75,90,95}_{target}_{seed|cell}.pkl
```

Default training/evaluation uses **`seed`**. Cell-split files are required only
for `quantile_shape_cell.py` and must be trained separately.

## Interpretation boundaries

- Scope is the **center recorder**.
- `log_abs` is natural log.
- Mean `|SHAP|` ranks importance; signed SHAP / PDPs show direction and shape.
- QBM answers distributional questions; mean GBM answers mean-prediction questions.
