# Appendix 2 — intensity measures, peak detection, 1D normalization

This folder keeps upstream IM/peak material inside `full_paper/analysis` so it
is not lost relative to the χ modeling code under `analysis/code/`.

## Contract

| Stage | Where |
|-------|--------|
| TF computation | `seiskit/ttf/`, campaign Box `h=*/transfer_function_results/` |
| PGA, PSA, Arias \(I_a\) | `seiskit/intensity_measures.py`, Box `h=*/psa_results/`, `ia_results/` |
| Peak detection (Appendix 2) | Box `complete/peak_analysis/` (`window_max`, prominence pick) |
| 1D normalization \(\chi^N\) | Box `peak_normalized.npz` / `psa_ia_normalized.npz` → **`join_master.h5`** |
| χ stats / ML | `analysis/code/chi_*` consuming `join_master.h5` |

## Symbols

- \(\chi_{ij}(\boldsymbol{\theta}_k)\): IM or peak TF measure at node \(i\), seed \(j\), design cell \(\boldsymbol{\theta}_k\)
- \(\chi^N_{ij} = \chi_{ij}/\chi^{\mathrm{1D}}\)
- \(Y_{ij}=\ln\chi^N_{ij}\)

Peak algorithm parameters (from Box manifests): `peak_method=window_max`,
`window_policy=midpoint`, `window_frac_fixed=0.5`, `min_prominence=0.05`.

## Scripts

| Script | Role |
|--------|------|
| [`plot_peak_stability.py`](plot_peak_stability.py) | Nature PDFs for appendix peak QA (from Box stability metrics / PNGs) |

Run::

    python analysis/appendix_im/plot_peak_stability.py
