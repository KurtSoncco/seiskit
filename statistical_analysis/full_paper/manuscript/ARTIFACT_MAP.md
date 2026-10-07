# Paper artifact map

Base: `complete/full_paper/figures/`

| Paper item | Main-text path | Supplemental |
|------------|----------------|--------------|
| Fig1 | `model_scheme/model_scheme_array.pdf` | `model_scheme_center.pdf` |
| Fig2 | `descriptions/ricker_wave.pdf` | — |
| Fig3 | `vs_rh_realizations/vs_rh_realizations.pdf`, `vs_cov_realizations.pdf` | — |
| Fig4 | `qualitative/center_node_one_seed/3x3/tf_raw_3x3_h50_vs1_230.pdf` | other `h*_vs1_*` |
| Fig5 | `qualitative/one_seed_all_nodes/3x3/tf_raw_3x3_h50_vs1_230.pdf` | other 8 cases |
| Fig6 | `qualitative/center_node_all_seeds/3x3/tf_raw_3x3_h50_vs1_230.pdf` | other 8 cases |
| Fig7 | `qualitative/all_seeds_all_nodes/3x3/tf_raw_3x3_h50_vs1_230.pdf` | other 8 cases |
| Fig9 | `chi_variables/factor_violins/chi_violins_{one_seed_all_nodes,center_node_all_seeds}.pdf` | `distribution_histograms/hist_*.pdf` (same two samples) |
| Table2 | `manuscript/tables/tab2_r_ceiling.tex` ← `chi_ols/r2_ceiling/` | — |
| Fig10 | `chi_variables/central_profiles/seed_profiles/abs_TF_seed_profile_h50_vs1_230.pdf` | other cases/metrics |
| Fig11 | `chi_variables/central_profiles/node_profiles/abs_TF_node_profile_h50_vs1_230.pdf` | other cases |
| Fig12 | `chi_variables/variance_heatmaps/heatmap_frac_*.pdf` | per-metric panels |
| Fig13 | `chi_spatial/spatial_acf/spatial_acf_abs_TF_ratio.pdf` | ACF CSVs other metrics |
| Table3 | `manuscript/tables/tab3_acf.tex` | — |
| Fig14 | `chi_spatial/spatial_coherence/coherence_vs_lag_*.pdf` | — |
| Table4 | `manuscript/tables/tab4_literature_coherence.tex` ← `chi_spatial/literature_coherence/` | compare PDF |
| Table5 | `manuscript/tables/tab5_model_adequacy.tex` ← `chi_ngboost/{between,within}/`, `chi_ngboost/spread/{within,between}/` | PIT histograms per partition |
| Fig15 | `chi_shap/shap_beeswarm/between/shap_beeswarm.pdf` | `shap_beeswarm/within/` (seed 1, single realization) |
| Fig16 | `chi_shap/ale_effects/between/ale_<metric>.pdf` | `ale_effects/within/` (seed 1) |
| Fig17 | `chi_shap/shap_median_vs_tail/between/shap_median_vs_tail_delta_{abs,signed}.pdf` | `shap_median_vs_tail/within/` (seed 1) |
| Fig18 | `chi_shap/ale_dispersion/between/ale_{scale,q95}_<metric>.pdf` | `ale_dispersion/within/` (seed 1) |
| Fig19 | `chi_shap/interactions/between/interactions.pdf` | `interactions/within/` (seed 1) |
| Fig20 | `chi_shap/ale_2d/between/ale_2d_<metric>.pdf` | `ale_2d/within/` (seed 1) |
| Spread-a | `chi_ngboost/spread/figures/spread_by_factor_{within,between}.pdf` | — |
| Spread-b | `chi_shap/spread_effects/spread_beeswarm.pdf` | `spread_beeswarm_sigma.pdf` |
| Spread-c | `chi_shap/spread_effects/ale_spread_<metric>.pdf` | `f0_spread_hurdle.pdf` |
| Spread-supp | — | `chi_ngboost/spread/figures/{sB_node_profile,spread_calibration,representativeness}.pdf`, `chi_ngboost/node_robustness/node_robustness.pdf` |
| Table6 | `manuscript/tables/tab6_synthesis.tex` ← `chi_shap/spread_effects/` | `tab6_synthesis.csv` |
| App1 | `chi_variables/mean_variance_adequacy/mean_variance_adequacy.pdf` | — |
| App2 | `appendix_im/peak_found_rates.pdf` | — |
