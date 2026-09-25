# Hallal Toro / Passeri / Dmult / Pretell vs neural-operator 2-layer 2D

Compares Hallal-style 1D randomization / damping proxies **and** Pretell-style
1D column sampling (Campbell \(Q\)–\(V_s\)) to the **two-layer** Sobol OpenSees
2D campaign on Box (not the 243-cell statistical-analysis factorial; not
three-layer / dipping).

## Reference

- Box: `Projects/Neural Operator/data/`
- 2D center |TF|: lateral index 10 of `transfer_function/tf_per_sample.npy`
  (21 laterals @ 15 m; base at y=2 m, surface at Lz)
- Design: 256 Sobol × 30 RF seeds = 7680 runs

## Methods

| Arm | Description |
|-----|-------------|
| Toro | Vs-only, fixed H, σ_ln = CoV, N=200 → geomean ± σ_ln |
| Passeri | tts-only, fixed H, σ_ln = CoV, N=200 → geomean ± σ_ln |
| Dmult | Base column; whole-profile \(0.25\,\xi_Q\) × clip(-1.3·Vs2/Vs1+13.90, 2, 10) |
| Pretell | 200 evenly spaced columns on the 500 m strip; Thomson–Haskell `AF_within` with **elemental** Campbell ξ_Q = 1/(2Q(Vs)); saves geomean, p16, p84, σ_ln |

Pearson: `corr(ln|TF|_method, ln|TF|_2D_center)` on 0.1–10 Hz (1000 pts).

## Run

```bash
# smoke
HALLAL_N_REAL=4 python run_comparison.py --smoke

# full Hallal arms (joblib; uses all cores or HALLAL_N_JOBS / SLURM_CPUS_PER_TASK)
python run_comparison.py

# Pretell (Campbell Q–Vs on strip columns; default N=200)
HALLAL_N_JOBS=16 PRETELL_N_SAMPLES=200 python add_pretell.py

# figures (4×3 panels + Pearson boxplot)
python plot_cases.py

# Savio (optional)
mkdir -p logs
sbatch submit_savio.sh
```

## Outputs (Box `hallal_vs_2d/`)

| File | Content |
|------|---------|
| `tf_2d_center.h5` | Cached 2D center |TF| (7680 × 1000) |
| `ensembles.h5` | Toro/Passeri geomean + σ_ln (256) and Dmult |TF| (256) |
| `pretell_ensembles.h5` | Pretell geomean / p16 / p84 / σ_ln per run index (7680) |
| `pearson_center.csv` / `.h5` | 7680 rows with `r_toro`, `r_passeri`, `r_dmult`, `r_pretell` |
| `figures/panels_4x3/` | One method per row (Toro / Passeri / Dmult / Pretell) |
| `figures/pearson_boxplot_methods.png` | Method agreement boxplot |

