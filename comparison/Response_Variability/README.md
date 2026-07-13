# Response_Variability Comparison

Response-focused benchmark comparing five randomization / damping protocols on **64 Sobol base cases** (4D: Vs1, H, CoV, Vs2). Fixed geostatistics: `rH=10 m`, `aHV=50`, bedrock thickness `10 m`.

| Arm | Method | Description |
|-----|--------|-------------|
| `hallal_vs` | 1D Vs randomization | Toro (1995) simplified profile (σ_ln Vs only; fixed H) |
| `hallal_tts` | 1D travel-time randomization | Passeri simplified profile (σ_ln tts only; fixed H) |
| `hallal_dmin` | Damping modification (Approach 5) | Base-case Vs + lab Q–Vs **Dmin ×3…×6** (10 multipliers) |
| `grf_2d` | Proposed 2D GRF (reference) | **GIFNO surrogate** on neural-operator grid (default) |
| `pretell` | Pretell et al. (2022) | **1D OpenSees** geomean over profiles in central **100 m** |

## Neural-operator 2D grid (`grf_2d` & `pretell`)

Both RF arms build the **same 2D GRF** on the GIFNO training domain (no resampling for surrogate):

| Quantity | Value |
|----------|-------|
| Lateral total `Lx` | **1500 m** (1500 columns) |
| BC width (each side) | **500 m** |
| Variability strip | **500 m** (columns 500–999) |
| **GIFNO model input** | **500 × 128** strip only (`(4, 128, 500)`); BC columns excluded |
| `dx`, `dz` | **1.0 m** |
| Depth rows `nz` | `round(H + bedrock)` at 1 m (≤ 128 padded in surrogate) |
| Surrogate recorders | 21 laterals, 15 m spacing, centered in variability strip |
| TF frequencies | 1000 points, logspace **0.1–10 Hz** |

Hallal 1D arms remain on `dx=dz=0.5 m`.

### Pretell protocol

[Pretell et al. (2022)](https://journals.sagepub.com/doi/pdf/10.1177/87552930211069400): extract **1D Vs profiles** from the **central 100 m** of the 2D field (within the 500 m variability strip), run 1D site response on each, aggregate with **geometric mean** |TF|.

| Mode | Profiles per 2D realization |
|------|---------------------------|
| Smoke | **10** (minimum recommended) |
| Production | **50** |

Override: `RV_PRETELL_N_SAMPLES=N`.

## Comparison philosophy (TF-first)

Under linear viscoelastic site response, the transfer function

\[
AF(f) = |FAS_{surface}(f)| / |FAS_{base}(f)|
\]

is an intrinsic property of the profile (and 2D geometry). Primary metrics:

- median / geomean `|AF(f)|` and peak metrics (`f_peak`, `A_peak`)
- `σ_ln AF(f)` across seeds (aleatory TF variability)
- Anderson-style GOF on `ln AF` near `f_peak`

Sa / PGA are secondary sanity checks.

One broadband drive (`M1`, 3 Hz) is sufficient for TF comparison.

## Campaign size

| Mode | Sobol cases | Hallal Vs/Tts seeds | Dmin mults | RF seeds | Total runs |
|------|-------------|---------------------|------------|----------|------------|
| Full (`RV_SMOKE=0`) | 64 | 200 each | 10 | 30 | **30,080** |
| Smoke (`RV_SMOKE=1`, 1D only) | 4 | 10 each | 10 | — | **120** |
| Smoke + 2D (`RV_SMOKE_2D=1`) | 4 | 10 each | 10 | 5 | **160** |

Index layout: Hallal block (`hallal_vs` + `hallal_tts` seeds, then `hallal_dmin` ×10 multipliers) → GRF block (`grf_2d`, `pretell` × RF seeds).

### 2D surrogate (`grf_2d`)

By default, `grf_2d` uses **GIFNO-FDO-XT** checkpoint `xt_lat128_d128`
(`LATENT_CHANNELS=128`, `DEEPONET_LATENT_DIM=128`):

`~/surrogate-seismic-waves/checkpoints/xt_lat128_d128/best_model.pt`

The 2D field is generated **natively** on the training grid above (model input is the **500 m variability strip** only). `pretell` runs OpenSees 1D on the same field.

Center-recorder aleatory statistics across RF seeds use **geometric mean** with a **±1σ_ln** band:

```
AF_lo = geomean × exp(-σ_ln)     AF_hi = geomean × exp(+σ_ln)
```

```bash
# Uses ~/surrogate-seismic-waves/.venv (torch) when --2d is passed
./submit_local.sh --2d --n-seeds 10

# Defaults set automatically when --2d (override as needed)
export GIFNO_SURROGATE_ROOT=~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT
export GIFNO_MODEL_DIR=~/surrogate-seismic-waves/checkpoints/xt_lat128_d128
export GIFNO_LATENT_CHANNELS=128
export GIFNO_DEEPONET_LATENT_DIM=128
export GIFNO_DATA_ROOT="/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data"

RV_USE_SURROGATE_2D=0 ./submit_local.sh --2d   # OpenSees 2D fallback
```

## Quick start (local smoke)

```bash
cd comparison/Response_Variability
chmod +x submit_local.sh
./submit_local.sh              # 3× 1D Hallal @ Sobol #0 (no 2D)
./submit_local.sh --all         # full 120-index 1D smoke
./submit_local.sh --2d          # quick run incl. grf_2d + pretell
./submit_local.sh --all --2d    # full 160-index smoke with 2D
```

Generated artifacts (`results/h5`, `logs/`) are gitignored.

## Analysis

```bash
python analyze_response.py --h5-dir results/h5 --out-dir results/analysis
python plot_comparison.py --h5-dir results/h5 --out-dir results/figures --sobol-id 0
```

Primary smoke figure: `results/figures/tf_methods_sobol00_M1.png`.

**Surrogate vs Pretell (geomean):** `results/figures/tf_grf2d_vs_pretell_geomean_sobol00_M1.png`.

Explainability:

- `hallal_profiles_sobol00.png` — Toro / Passeri realizations vs base profile
- `grf2d_explainability_*.png` — NO-grid GRF + Pretell central 100 m extractions

## HPC

**Recommended order:** submit 1D Hallal block first, then 2D GRF / Pretell.

```bash
chmod +x submit_phase1_hallal.sh submit_phase1_rf.sh clean_results.sh

./submit_phase1_hallal.sh --smoke  # 120 indices
./submit_phase1_hallal.sh          # 26,240 indices

RV_SMOKE=1 RV_SMOKE_2D=1 ./submit_phase1_rf.sh --smoke   # 40 RF indices
./submit_phase1_rf.sh              # 3,840 RF indices
```

**Stampede3 (TACC):**

```bash
RV_INDEX_OFFSET=0 RV_INDEX_MAX=26240 sbatch stampede3_full_run.slurm
RV_INDEX_OFFSET=26240 RV_INDEX_MAX=30080 sbatch stampede3_full_run.slurm
```

**Hallal Dmin arm:** fixed base-case Vs; damping from Campbell (2009) Q–Vs with **10 Dmin multipliers** `linspace(3, 6)`.

## Modules

- `sobol_base_cases.py` — 4D Sobol CSV (`rv_sobol_base_cases.csv`)
- `manifest.py` — index → case mapping, NO grid constants, Pretell sampling
- `run_experiment.py` — OpenSees driver (+ GIFNO surrogate for `grf_2d`)
- `surrogate_2d.py` — GIFNO inference (native grid or legacy resample)
- `seiskit/profile_randomization.py` — Toro / Passeri
