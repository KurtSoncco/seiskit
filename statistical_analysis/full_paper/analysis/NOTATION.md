# χ analysis notation (central tendency & variability)

Canonical symbols for design-cell χ ratios in the full-paper analysis.
Equation generators that emit matching markdown:

- [`code/chi_variables/central_variability.py`](code/chi_variables/central_variability.py)
- [`code/chi_variables/mean_variance_adequacy.py`](code/chi_variables/mean_variance_adequacy.py)
- [`code/chi_ols/`](code/chi_ols/) (Stage-1 OLS, R² ceiling, hetero contrast, spatial \(n_{\mathrm{eff}}\))
- [`code/chi_qbm/`](code/chi_qbm/) (spatial GLS OLS, LightGBM QBM, three-way comparison)
- [`code/chi_ngboost/`](code/chi_ngboost/) (Normal NGBoost predictive distribution)
- [`code/chi_shap/`](code/chi_shap/) (SHAP on NGBoost + QBM; shortlist for SR)
- [`code/chi_sr/`](code/chi_sr/) (symbolic regression of NGBoost \(\mu\) / \(\log\sigma\))

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

## Stage-1 OLS and reliability ceiling (`chi_ols`)

Across design cells \(k = 1,\ldots,K\) (\(K = 243\)), with observation \(Y_{kij}\) at cell \(k\), seed \(j\), node \(i\):

| Symbol | Definition | CSV / code |
|--------|------------|------------|
| \(\mathbf{x}_k\) | standardized main-effect design vector for cell \(k\) | `*_z` columns |
| \(\boldsymbol{\beta}\) | Stage-1 OLS mean coefficients | `coef` in `mean_effects.csv` |
| \(\mathrm{SE}_{\mathrm{naive}}\) | classical OLS standard error | `se_naive` |
| \(\mathrm{SE}_{\mathrm{seed}}\) | cluster-robust SE (cluster = seed) | `se_seed_cluster` |
| \(\mathrm{SE}_{\mathrm{cell}}\) | cluster-robust SE (cluster = cell) | `se_cell_cluster` |
| \(\mathrm{SE\,infl.}\) | \(\mathrm{SE}_{\mathrm{cluster}} / \mathrm{SE}_{\mathrm{naive}}\) | `se_infl_seed`, `se_infl_cell` |
| \(R^2\) | in-sample coefficient of determination | `r2_insample` |
| \(R^2_{\mathrm{CV}}\) | cell-grouped cross-validated \(R^2\) | `r2_cv_cell` |
| \(\mu_k\) | cell mean \(E[Y\mid k]\) (oracle / for ceiling) | cell mean of \(Y\) |
| \(\sigma^2_{\mathrm{signal}}\) | \(\mathrm{Var}_k(\bar Y_k)\) | `sigma2_signal` |
| \(\sigma^2_{\mathrm{noise}}\) | \(\mathrm{mean}_k(s^2_k)\) within-cell variance | `sigma2_noise` |
| \(R^2_{\mathrm{ceiling}}\) | \(\sigma^2_{\mathrm{signal}}/(\sigma^2_{\mathrm{signal}}+\sigma^2_{\mathrm{noise}})\) | `reliability_ceiling` |
| efficiency | \(R^2 / R^2_{\mathrm{ceiling}}\) | `efficiency` |
| \(\boldsymbol{\gamma}\) | coefficients in \(\log s^2_k = \mathbf{z}_k^\top\boldsymbol{\gamma}\) | `variance_effects.csv` |

Mean model:

\[
Y_{kij} = \mathbf{x}_k^\top\boldsymbol{\beta} + \varepsilon_{kij}.
\]

Reliability (signal-to-total) ceiling for predicting a single draw given design only:

\[
R^2_{\mathrm{ceiling}}
= \frac{\widehat{\mathrm{Var}}(\bar Y_k)}{\widehat{\mathrm{Var}}(\bar Y_k) + \overline{s^2_k}}.
\]

## Spatial GLS and QBM (`chi_qbm`)

| Symbol | Definition | CSV / code |
|--------|------------|------------|
| \(R_k\) | CosWM (or Exp/Gauss) correlation on the \(N_x\)-node array for cell \(k\) | from `acf_fit_params.csv` |
| \(\boldsymbol{\beta}_{\mathrm{GLS}}\) | feasible GLS mean coefs under block-diagonal \(R\) | `spatial_mean_effects.csv` |
| \(\tilde y\) | whitened response \(L^{-1} y\) with \(R = LL^\top\) | whitening QA |
| \(q_\tau(Y\mid\mathbf{x})\) | conditional \(\tau\)-quantile (LightGBM QBM) | quantile models |
| \(\rho_\tau\) | pinball / check loss at level \(\tau\) | `pinball` columns |
| \(R^1(\tau)\) | Koenker–Machado pseudo-\(R^2\): \(1-\rho_\tau(\mathrm{model})/\rho_\tau(\mathrm{null})\) | `pseudo_r2` |
| \(x_{\mathrm{node}}\) | standardized node index (spatial feature for QBM) | `node_z` |

Feasible GLS (correlation known up to scale) with cell-constant design rows \(\mathbf{x}_k\):

\[
\hat{\boldsymbol{\beta}}_{\mathrm{GLS}}
= \Bigl(\sum_{k,j} (\mathbf{1}^\top R_k^{-1}\mathbf{1})\,
\mathbf{x}_k\mathbf{x}_k^\top\Bigr)^{-1}
\sum_{k,j} (\mathbf{1}^\top R_k^{-1}\mathbf{y}_{kj})\,\mathbf{x}_k.
\]

QBM models \(q_\tau(Y\mid \mathbf{x}_k, x_{\mathrm{node}})\) by gradient boosting; trees encode interactions. Prediction intervals use quantile pairs (e.g. \(\tau=0.05,0.95\)).

## NGBoost, SHAP, and symbolic regression

| Symbol | Definition | CSV / code |
|--------|------------|------------|
| \(p(Y\mid\mathbf{x})\) | parametric predictive density (Normal NGBoost on \(Y=\ln\chi\)) | `chi_ngboost` |
| \(\mu(\mathbf{x})\) | predictive mean (loc) | `mu`, NGBoost / SR |
| \(\sigma(\mathbf{x})\) | predictive scale | `sigma`, NGBoost |
| \(\log\sigma(\mathbf{x})\) | log-scale (dispersion surface) | `log_sigma` |
| \(\phi_j\) | SHAP attribution for feature \(j\) (model-tagged) | `shap_*` CSVs |
| \(\phi_{jk}\) | pairwise SHAP interaction | `shap_interactions_*` |
| \(\hat\mu_{\mathrm{SR}}\) | symbolic approximation to NGBoost \(\mu\) | `sr_formulas.csv` |
| \(\widehat{\log\sigma}_{\mathrm{SR}}\) | symbolic approximation to NGBoost \(\log\sigma\) | `sr_formulas.csv` |

NGBoost learns \(Y\mid\mathbf{x}\sim\mathcal{N}(\mu(\mathbf{x}),\sigma^2(\mathbf{x}))\) by natural-gradient boosting. SHAP decomposes \(\mu\) / \(\log\sigma\) (and QBM quantiles) without whitening residual spatial dependence. Symbolic regression compresses SHAP-shortlisted features into explicit formulas.
