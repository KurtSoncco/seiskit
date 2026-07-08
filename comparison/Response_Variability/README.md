# Response_Variability Comparison

Response-focused benchmark comparing five randomization / damping protocols on **64 Sobol base cases** (4D: Vs1, H, CoV, Vs2). Fixed geostatistics: `rH=10 m`, `aHV=50`, bedrock thickness `10 m`.

| Arm | Method | Description |
|-----|--------|-------------|
| `hallal_vs` | 1D Vs randomization | Toro (1995) full profile (Vs-only; fixed H) |
| `hallal_tts` | 1D travel-time randomization | Passeri full profile (Vs-only; fixed H) |
| `hallal_dmin` | Damping modification (Approach 5) | Base-case Vs + lab Q–Vs **Dmin ×3** on entire profile |
| `grf_2d` | Proposed 2D GRF (reference) | Full 2D field, `global_avg` Q–Vs |
| `delatorre` | de la Torre protocol | **1D OpenSees** with center column extracted from same 2D GRF |

## Comparison philosophy (TF-first)

Under linear viscoelastic site response, the transfer function

\[
AF(f) = |FAS_{surface}(f)| / |FAS_{base}(f)|
\]

is an intrinsic property of the profile (and 2D geometry). Primary metrics:

- median `|AF(f)|` and peak metrics (`f_peak`, `A_peak`)
- `σ_ln AF(f)` across seeds (aleatory TF variability)
- Anderson-style GOF on `ln AF` near `f_peak`

Sa / PGA are secondary sanity checks.

One broadband drive (`M1`, 3 Hz) is sufficient for TF comparison.

## Campaign size

| Mode | Sobol cases | Hallal seeds | RF seeds | Total runs |
|------|-------------|--------------|----------|------------|
| Full (`RV_SMOKE=0`) | 64 | 200 | 30 | **42,240** |
| Smoke (`RV_SMOKE=1`) | 4 | 10 | 5 | **160** |

Index layout: Hallal block (3 methods × seeds) → GRF block (`grf_2d`, `delatorre` × RF seeds).

Smoke grid: `dx=dz=1 m`, `Lx_var=100 m`, `BC=50 m` (full uses 0.5 m / 200 m / 100 m).

## Quick start (local smoke)

```bash
cd comparison/Response_Variability
chmod +x submit_local.sh
./submit_local.sh              # 5 runs (one seed per method @ Sobol #0)
./submit_local.sh --all         # full 160-index smoke
```

Generated artifacts (`results/h5`, `logs/`) are gitignored.

## Analysis

```bash
python analyze_response.py --h5-dir results/h5 --out-dir results/analysis
python plot_comparison.py --h5-dir results/h5 --out-dir results/figures --sobol-id 0
```

Primary smoke figure: `results/figures/tf_methods_sobol00_M1.png` (all methods: median AF ± 1σ_ln band).

Explainability:

- `hallal_profiles_sobol00.png` — Toro / Passeri realizations vs base profile
- `grf2d_explainability_*.png` — 2D GRF field + center-column extraction for de la Torre 1D

## HPC

**Recommended order:** submit 1D Hallal block first, then 2D GRF / de la Torre.

```bash
chmod +x submit_phase1_hallal.sh submit_phase1_rf.sh clean_results.sh

./clean_results.sh                 # optional: wipe local stale results

# Savio2 — phase 1a: Hallal 1D only
./submit_phase1_hallal.sh --smoke  # 120 indices, array 0-4
./submit_phase1_hallal.sh          # 38,400 indices, array 0-1599

# Savio2 — phase 1b: 2D GRF + de la Torre (after 1D completes)
./submit_phase1_rf.sh --smoke      # 40 indices, array 0-1
./submit_phase1_rf.sh              # 3,840 indices, array 0-159

# Or run everything in one array (mixed 1D + 2D):
./submit_phase1.sh --smoke
./submit_phase1.sh
```

**Stampede3 (TACC)** — set `RV_INDEX_OFFSET` / `RV_INDEX_MAX` for 1D-first staging:

```bash
# 1D block
RV_INDEX_OFFSET=0 RV_INDEX_MAX=38400 sbatch stampede3_full_run.slurm
# 2D block (after 1D)
RV_INDEX_OFFSET=38400 RV_INDEX_MAX=42240 sbatch stampede3_full_run.slurm
```

Scratch env vars: `RV_OUTDIR`, `RV_H5_DIR`.

**Hallal Dmin arm:** fixed base-case Vs; damping from Campbell (2009) Q–Vs with a constant multiplier on the whole Dmin profile (default **×3**, Hallal/Tao & Rathje TIDA; set `RV_DMIN_MULT=6` for the higher-damping sensitivity case).

## Modules

- `sobol_base_cases.py` — 4D Sobol CSV (`rv_sobol_base_cases.csv`)
- `manifest.py` — index → case mapping
- `run_experiment.py` — OpenSees driver
- `seiskit/profile_randomization.py` — Toro / Passeri
- `seiskit/intensity_measures.py` — PGA, Sa, σ_ln
- `seiskit/ttf/TTF.py` — Konno–Ohmachi AF(f)
