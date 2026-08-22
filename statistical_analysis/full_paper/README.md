# Full-paper statistical analysis

χ-ratio analysis and Nature figures for the full paper. Outputs go to Box:

`/mnt/box/.../Statistical Analysis/complete/full_paper/figures/<topic>/`

## Layout

| Path | Role |
|------|------|
| [`config.py`](config.py) | Nature figure style, `FACTORS` / `METRICS`, `figure_dir` |
| [`analysis/NOTATION.md`](analysis/NOTATION.md) | Canonical math symbols |
| [`analysis/appendix_im/`](analysis/appendix_im/) | Appendix 2 / IM–peak upstream + appendix plots |
| [`analysis/qualitative/`](analysis/qualitative/) | \|TF\| **3×3** sensitivity (`run_tf.py`; cross layout removed) |
| [`analysis/code/_shared.py`](analysis/code/_shared.py) | Shared `load_ratios`, splits, metrics helpers |
| [`analysis/code/chi_*`](analysis/code/) | Quantitative χ pipelines |
| [`figures/`](figures/) | Manuscript graphics (model scheme, Vs realizations, …) |

## Upstream → χ contract

TF / PGA / SA(\(T_0\)) / \(I_a\) / peak picking / 1D normalization produce
`complete/peak_analysis/join_master.h5`. All `chi_*` scripts consume that table
as \(\chi^N\) ratios. See [`analysis/appendix_im/README.md`](analysis/appendix_im/README.md).

## Suggested run order

1. `appendix_im/plot_peak_stability.py`
2. `qualitative/run_tf.py --mode all` (includes `center_node_one_seed`)
3. `chi_variables/` (central_variability → violins / profiles / histograms / heatmaps)
4. `chi_spatial/spatial_acf.py` then `spatial_coherence.py` then `literature_coherence.py`
5. `chi_ols/` → `chi_qbm/` → `chi_ngboost/` → `chi_shap/` (beeswarm, ALE median/dispersion/2D, interactions)
6. Optional: `chi_sr/` (symbolic regression; not in main paper outline)
7. `figures/vs_rh_realizations.py`, `figures/field_stat_recovery.py`
8. `manuscript/export_tables.py` → LaTeX Tables 2–6 + `ARTIFACT_MAP.md`

Use the project venv: `.venv/bin/python`.

See [`manuscript/ARTIFACT_MAP.md`](manuscript/ARTIFACT_MAP.md) for paper Fig/Table → Box path mapping.

## Notes

- Working scale: \(Y_{ij}=\ln\chi^N_{ij}\).
- Seed holdout is shared via `chi_qbm/.../seed_split.json` (`load_or_make_split`).
- Spatial **ACF** (within-seed lag structure) ≠ between-seed **coherence** \(R=[\rho_{ik}]\).
