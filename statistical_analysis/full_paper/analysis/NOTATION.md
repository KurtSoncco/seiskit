# χ analysis notation (central tendency & variability)

Canonical symbols for design-cell χ ratios in the full-paper analysis.
Equation generators that emit matching markdown:

- [`code/chi_variables/central_variability.py`](code/chi_variables/central_variability.py)
- [`code/chi_variables/mean_variance_adequacy.py`](code/chi_variables/mean_variance_adequacy.py)
- [`code/chi_qbm/`](code/chi_qbm/) (LightGBM QBM / mean GBM)
- [`code/chi_ngboost/`](code/chi_ngboost/) (Normal NGBoost predictive distribution)
- [`code/chi_shap/`](code/chi_shap/) (SHAP on NGBoost + QBM)

Do **not** use bare \(n\)/\(m\) or \(N\)/\(S\) for node/seed counts in new math or prose.

## Indices and counts

| Symbol | Meaning | Experiment value |
|--------|---------|------------------|
| \(i\) | node (spatial) index | \(i = 1,\ldots,N_x\) |
| \(j\) | seed (realization) index | \(j = 1,\ldots,N_s\) |
| \(N_x\) | number of nodes | 101 |
| \(N_s\) | number of seeds | 100 |

### Code mapping

| Math | Python / CSV |
|------|----------------|
| \(N_x\) | `N_NODES`, `n_nodes` |
| \(N_s\) | `N_SEEDS`, `n_seeds` |

CSV column names and Python identifiers keep historical forms (e.g. `S_req_*`, `adequate_mean_S100`); only **displayed** math/prose use \(N_x\), \(N_s\).

## Working scale

Fix one design cell and one metric \(\chi\).

\[
Y_{ij} = \ln \chi_{ij}.
\]

Unless noted, CT/variability sums use population (ddof = 0) means and variances over finite \(Y_{ij}\) only.

## Central tendency

| Symbol | Definition | CSV / code |
|--------|------------|------------|
| \(\mu_j\) | seed log-mean: \(N_x^{-1}\sum_{i=1}^{N_x} Y_{ij}\) | `mu_j` |
| \(G^{\mathrm{seed}}_j\) | \(e^{\mu_j}\) | `chi_geom_j` |
| \(\nu_i\) | node log-mean: \(N_s^{-1}\sum_{j=1}^{N_s} Y_{ij}\) | `nu_i` |
| \(G^{\mathrm{node}}_i\) | \(e^{\nu_i}\) | `chi_geom_i` |
| \(\bar Y\) | overall log-mean: \((N_x N_s)^{-1}\sum_{i,j} Y_{ij}\) | `ln_chi_bar` |
| \(G\) | overall geomean \(e^{\bar Y}\) | `chi_geom` |
| \(\overline{\chi}\) | arithmetic mean of \(\chi_{ij}\) | `chi_mean` |
| \(\mathrm{median}_{ij}(\chi)\) | median of \(\chi_{ij}\) | `chi_median` |
| \(G^{\mathrm{seed}}_{P10,P90}\), \(G^{\mathrm{node}}_{P10,P90}\) | percentiles of seed/node geomeans | `G_seed_p10`, … |

On a complete balanced grid,

\[
G =
\Bigl(\prod_{j=1}^{N_s} G^{\mathrm{seed}}_j\Bigr)^{1/N_s} =
\Bigl(\prod_{i=1}^{N_x} G^{\mathrm{node}}_i\Bigr)^{1/N_x}.
\]

## Variability (law of total variance)

| Symbol | Meaning | CSV (rms form where noted) |
|--------|---------|----------------------------|
| \(s^{2}_{W,j}\) | within-seed var: \(N_x^{-1}\sum_i (Y_{ij}-\mu_j)^2\) | → `s_W` |
| \(\overline{s^{2}_W}\) | average over seeds | `s_W_bar` \(= \sqrt{\overline{s^{2}_W}}\) |
| \(s^{2}_{B,i}\) | across-seed var: \(N_s^{-1}\sum_j (Y_{ij}-\nu_i)^2\) | → `s_B` |
| \(\overline{s^{2}_B}\) | average over nodes | `s_B_bar` |
| \(\sigma^{2}_{\mu}\) | var of seed means \(\{\mu_j\}\) | `s_mu` \(= \sqrt{\sigma^{2}_{\mu}}\) |
| \(\sigma^{2}_{\nu}\) | var of node means \(\{\nu_i\}\) | `s_nu` |
| \(\sigma^{2}_{\mathrm{total}}\) | total log-var of all \(Y_{ij}\) | `s_total` |

Primary splits:

\[
\sigma^{2}_{\mathrm{total}} = \overline{s^{2}_W} + \sigma^{2}_{\mu}
= \overline{s^{2}_B} + \sigma^{2}_{\nu}.
\]

Fractions: \(f_{W\mid\mathrm{seed}}\), \(f_{\mu}\), \(f_{B\mid\mathrm{node}}\), \(f_{\nu}\) (CSV `frac_W_seed`, `frac_mu`, `frac_B_node`, `frac_nu`).

Diagnostic only: \(s^{2}_{T,\mathrm{naive}} = \overline{s^{2}_W}+\overline{s^{2}_B}\), gap \(s_{T,\mathrm{gap}}\) (`s_T_naive`, `s_T_gap`).

Interaction residual: \(R_{ij} = Y_{ij} - \mu_j - \nu_i + \bar Y\), with \(\sigma^{2}_{R}\) / \(f_R\) (`s_interaction`, `frac_interaction`).

## Sample-size adequacy

Seed-blocked estimators (adequacy uses sample SD with ddof = 1 where stated):

| Symbol | Definition |
|--------|------------|
| \(\hat\mu_j\) | \(N_x^{-1}\sum_{i=1}^{N_x} Y_{ij}\) |
| \(\hat\mu\) | \(N_s^{-1}\sum_{j=1}^{N_s}\hat\mu_j\) |
| \(\hat G\) | \(e^{\hat\mu}\) |
| \(s_\mu\) | sample SD of \(\{\hat\mu_j\}\) |
| \(\mathrm{SE}(\hat\mu)\) | \(s_\mu / \sqrt{N_s}\) |
| \(\mathrm{RSE}(\hat G)\) | \(\mathrm{SE}(\hat\mu)\) (delta method) |
| \(N_{s,\mu}(\varepsilon)\) | \(\lceil(s_\mu/\varepsilon)^2\rceil\) seeds for \(\mathrm{RSE}(\hat G)\le\varepsilon\) |
| \(n_{\mathrm{eff}}\) | CosWM effective node count from ACF \(\rho(h)\) |
| \(s_j^2\) | per-seed sample var of \(\{Y_{ij}\}_i\) (ddof=1) |
| \(\hat\sigma^2\) | \(N_s^{-1}\sum_j s_j^2\) |
| \(s_{s^2}\) | sample SD of \(\{s_j^2\}\) |
| \(\mathrm{SE}(\hat\sigma^2)\) | \(s_{s^2}/\sqrt{N_s}\) |
| \(N_{s,\sigma^2}(\varepsilon)\) | seeds for target \(\mathrm{RSE}(\hat\sigma^2)\le\varepsilon\) |

Effective size:

\[
n_{\mathrm{eff}}
= \frac{N_x}{1+2\sum_{k=1}^{N_x-1}\bigl(1-k/N_x\bigr)\rho(k\,\Delta x)}.
\]

Do **not** treat \(N_x N_s\) samples as iid. Keep \(n_{\mathrm{eff}}\) as written (effective sample size, not a node count \(N_x\)).

## Reliability ceiling (from variability)

Across design cells \(k = 1,\ldots,K\) (\(K = 243\)), let \(\bar Y_k\) be the cell mean of \(Y\) and \(s^2_k\) the within-cell sample variance (node×seed draws in cell \(k\)). Observation-level \(R^2\) for any design-only predictor of a single draw is limited by irreducible within-cell noise — the same seed/spatial variability decomposed above.

| Symbol | Definition | CSV / code |
|--------|------------|------------|
| \(\mu_k\) | cell mean \(E[Y\mid k]\) | cell mean of \(Y\) |
| \(\sigma^{2}_{\mathrm{signal}}\) | \(\mathrm{Var}_k(\bar Y_k)\) | `sigma2_signal` |
| \(\sigma^{2}_{\mathrm{noise}}\) | \(\mathrm{mean}_k(s^2_k)\) | `sigma2_noise` |
| \(R^2_{\mathrm{ceiling}}\) | \(\sigma^{2}_{\mathrm{signal}}/(\sigma^{2}_{\mathrm{signal}}+\sigma^{2}_{\mathrm{noise}})\) | `reliability_ceiling` |
| efficiency | \(R^2_{\mathrm{model}} / R^2_{\mathrm{ceiling}}\) | `efficiency` |

Signal-to-total ceiling (population-information bound, not a fitted-model score):

\[
R^2_{\mathrm{ceiling}}
= \frac{\widehat{\mathrm{Var}}_k(\bar Y_k)}{\widehat{\mathrm{Var}}_k(\bar Y_k) + \overline{s^2_k}}.
\]

Interpretation: fraction of variance that lives *between* design cells. Within-cell \(s^2_k\) is the cell-level counterpart of \(\sigma^{2}_{\mathrm{total}}\) from the variability section — design factors cannot remove it. Efficiency compares a mean model’s hold-out \(R^2\) (mean GBM or NGBoost \(\mu\)) to this ceiling:

\[
\mathrm{efficiency}
= \frac{R^2_{\mathrm{model}}}{R^2_{\mathrm{ceiling}}}.
\]

Optional cross-checks (same as conference paper): noise-corrected signal \(\sigma^{2}_{\mathrm{signal,bc}}=\max\bigl(0,\sigma^{2}_{\mathrm{signal}}-\overline{s^2_k/n_k}\bigr)\) and between-cell SS fraction \(R^2_{\mathrm{ceiling,SS}}\).

## QBM (`chi_qbm`)

| Symbol | Definition | CSV / code |
|--------|------------|------------|
| \(\mathbf{x}_k\) | standardized main-effect design vector for cell \(k\) | `*_z` columns |
| \(q_\tau(Y\mid\mathbf{x})\) | conditional \(\tau\)-quantile (LightGBM QBM) | quantile models |
| \(\rho_\tau\) | pinball / check loss at level \(\tau\) | `pinball` columns |
| \(R^1(\tau)\) | Koenker–Machado pseudo-\(R^2\): \(1-\rho_\tau(\mathrm{model})/\rho_\tau(\mathrm{null})\) | `pseudo_r2` |
| \(x_{\mathrm{node}}\) | standardized node index (spatial feature for QBM) | `node_z` |

QBM models \(q_\tau(Y\mid \mathbf{x}_k, x_{\mathrm{node}})\) by gradient boosting; trees encode interactions. Prediction intervals use quantile pairs (e.g. \(\tau=0.05,0.95\)). Mean GBM \(E[Y\mid\mathbf{x}]\) is the mean-prediction benchmark scored against \(R^2_{\mathrm{ceiling}}\); QBM is the distributional model (pinball, calibration, quantile SHAP).

## NGBoost and SHAP

| Symbol | Definition | CSV / code |
|--------|------------|------------|
| \(p(Y\mid\mathbf{x})\) | parametric predictive density (Normal NGBoost on \(Y=\ln\chi\)) | `chi_ngboost` |
| \(\mu(\mathbf{x})\) | predictive mean (loc) | `mu` |
| \(\sigma(\mathbf{x})\) | predictive scale | `sigma` |
| \(\log\sigma(\mathbf{x})\) | log-scale (dispersion surface) | `log_sigma` |
| \(q_\tau(\mathbf{x})\) | Normal predictive quantile \(\mu+\sigma\,z_\tau\) | `q05` / `q50` / `q95` surfaces |
| \(\phi_j\) | SHAP attribution for feature \(j\) (model-tagged) | `shap_*` CSVs |
| \(\phi_{jk}\) | pairwise SHAP interaction | `shap_interactions_*` |

NGBoost learns \(Y\mid\mathbf{x}\sim\mathcal{N}(\mu(\mathbf{x}),\sigma^2(\mathbf{x}))\) by natural-gradient boosting. Mean \(R^2\) for \(\mu\) is likewise reported relative to \(R^2_{\mathrm{ceiling}}\). SHAP decomposes \(\mu\) / \(\log\sigma\) (and QBM quantiles) for design and spatial drivers.

## Intensity measures and design cells

| Symbol | Meaning |
|--------|---------|
| \(\boldsymbol{\theta}_k\) | design-factor vector for cell \(k\) (Vs1, Height, CoV, \(r_H\), \(a_{HV}\)) |
| \(\chi_{ij}(\boldsymbol{\theta}_k)\) | peak TF or IM at node \(i\), seed \(j\) in cell \(k\) |
| \(\chi^{\mathrm{1D}}\) | 1D reference (same \(H\), \(V_{s1}\)) |
| \(\chi^N_{ij}\) | \(\chi_{ij}/\chi^{\mathrm{1D}}\) (CSV columns `*_ratio`) |

PSA in this campaign is evaluated at the 1D fundamental (\(T_0=4H/V_{s1}\)). Peak picking and Appendix 2 live under [`appendix_im/`](appendix_im/).

## Between-seed spatial coherence (`chi_spatial/spatial_coherence.py`)

Distinct from lag-ACF in `spatial_acf.py` (within-seed spatial structure of one realization).

| Symbol | Definition |
|--------|------------|
| \(C_{ik}\) | \(\frac{1}{N_s}\sum_j (Y_{ij}-\nu_i)(Y_{kj}-\nu_k)\) |
| \(\rho_{ik}\) | \(C_{ik}/(s_{B,i}\,s_{B,k})\) |
| \(R\) | \([\rho_{ik}]\in\mathbb{R}^{N_x\times N_x}\) |

## Calibration and explainability

| Symbol / tool | Role | Code |
|---------------|------|------|
| CRPS | sharpness of predictive Normal | `chi_ngboost/calibration_crps_pit.py` |
| PIT | calibration histogram of \(\Phi((Y-\mu)/\sigma)\) | same |
| ALE | accumulated local effects (marginal) | `chi_shap/ale_effects.py` |
| Friedman \(H\) | pairwise interaction strength | `chi_ngboost/exceedance_friedman.py` |
| \(\Delta\)SHAP | median vs upper-tail attributions | `chi_shap/shap_median_vs_tail.py` |

Optional symbolic regression of NGBoost surfaces: `chi_sr/` (not required for the main outline).
