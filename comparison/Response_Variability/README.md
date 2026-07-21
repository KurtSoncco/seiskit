# Response_Variability Comparison

Response-focused benchmark comparing randomization / damping protocols on **64 Sobol base cases** (4D: Vs1, H, CoV, Vs2). Fixed geostatistics: `rH=10 m`, `aHV=50`, bedrock thickness `10 m`.

| Arm | Method | Description |
|-----|--------|-------------|
| `hallal_vs` | 1D Vs randomization | Toro (1995) simplified profile (σ_ln Vs only; fixed H) |
| `hallal_tts` | 1D travel-time randomization | Passeri simplified profile (σ_ln tts only; fixed H) |
| `hallal_dmin` | Damping modification (Approach 5) | Base-case Vs + lab Q–Vs **Dmin ×3…×6** (10 multipliers) |
| `grf_2d` | 2D GRF (GIFNO) | GIFNO-FDO-XT surrogate on neural-operator grid |
| `pretell` | Pretell-style 1D ensemble | **1D OpenSees** geomean over **200** profiles across full **500 m** strip |
| `opensees_2d` | 2D GRF (**baseline**) | Full-mesh **OpenSees 2D** on the same GRF / seeds |

## Neural-operator 2D grid (RF arms)

RF methods build the **same 2D GRF** on the GIFNO training domain:

| Quantity | Value |
|----------|-------|
| Lateral total `Lx` | **1500 m** (1500 columns) |
| BC width (each side) | **500 m** |
| Variability strip | **500 m** (columns 500–999) |
| **GIFNO model input** | **500 × 128** strip only (`(4, 128, 500)`); BC columns excluded |
| `dx`, `dz` | **1.0 m** |
| Depth rows `nz` | `round(H + bedrock)` at 1 m (≤ 128 padded in surrogate) |
| Surrogate / 2D recorders | 21 laterals, 15 m spacing, centered in variability strip |
| TF frequencies | 1000 points, logspace **0.1–10 Hz** |

Hallal 1D arms remain on `dx=dz=0.5 m`.

### Pretell protocol

Extract **200** evenly spaced **1D Vs profiles** across the **full 500 m variability strip**, run 1D site response on each, aggregate with **geometric mean** |TF|.

| Mode | Profiles per 2D realization |
|------|---------------------------|
| Smoke | **10** |
| Production | **200** (full strip) |

Override: `RV_PRETELL_N_SAMPLES=N`.

## Comparison philosophy (TF-first)

Under linear viscoelastic site response, the transfer function

\[
TF(f) = |FAS_{surface}(f)| / |FAS_{base}(f)|
\]

is an intrinsic property of the profile (and 2D geometry). (HDF5 still stores this curve under the legacy dataset name `transfer_function/AF`.) Primary metrics:

- median / geomean `|TF(f)|` and peak metrics (`f_peak`, `A_peak` ≡ peak `|TF|`)
- `σ_ln TF(f)` across seeds (aleatory TF variability)
- Anderson-style GOF on `ln TF` near `f_peak`

**Reference method priority:** `opensees_2d` → `grf_2d` → `hallal_vs`.

Sa / PGA are secondary sanity checks. One broadband drive (`M1`, 3 Hz) is sufficient for TF comparison.

## Campaign size

Per Sobol point (production): **200** Hallal Vs + **200** Hallal Tts + **10** Dmin + **40** RF seeds × (`grf_2d` + `pretell` + `opensees_2d`) = **530** runs. Pretell draws **200** 1D profiles per RF realization.

| Mode | Sobol cases | Hallal Vs/Tts | Dmin | RF seeds | Total runs |
|------|-------------|---------------|------|----------|------------|
| Full (`RV_SMOKE=0`) | 64 | 200 each | 10 | 40 ×3 | **33,920** |
| Stampede OpenSees (skip `grf_2d`) | 64 | 200 each | 10 | 40 pretell + 40 ops2d | **31,360** |
| Smoke (`RV_SMOKE=1`, 1D only) | 4 | 10 each | 10 | — | **120** |
| Smoke + 2D (`RV_SMOKE_2D=1`) | 4 | 10 each | 10 | 5 ×3 | **180** |

Overrides: `RV_HALLAL_N_SEEDS`, `RV_RF_N_SEEDS`, `RV_PRETELL_N_SAMPLES`.

Index layout (append-only RF order keeps existing GIFNO/pretell slots stable):

Hallal block → `grf_2d` ×40 → `pretell` ×40 → `opensees_2d` ×40 (per Sobol).

### 2D surrogate (`grf_2d`)

By default, `grf_2d` uses **GIFNO-FDO-XT** checkpoint `xt_lat128_d128`
(`LATENT_CHANNELS=128`, `DEEPONET_LATENT_DIM=128`):

`~/surrogate-seismic-waves/checkpoints/xt_lat128_d128/best_model.pt`

`opensees_2d` never uses the surrogate (OpenSees baseline even if `RV_USE_SURROGATE_2D=1`).

```bash
./submit_local.sh --2d --n-seeds 10
export GIFNO_SURROGATE_ROOT=~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT
export GIFNO_MODEL_DIR=~/surrogate-seismic-waves/checkpoints/xt_lat128_d128
```

## Quick start (local smoke)

```bash
cd comparison/Response_Variability
chmod +x submit_local.sh
./submit_local.sh              # 3× 1D Hallal @ Sobol #0 (no 2D)
./submit_local.sh --all         # full 120-index 1D smoke
./submit_local.sh --2d          # quick run incl. RF arms
./submit_local.sh --all --2d    # full smoke with 2D
```

## Analysis

```bash
python analyze_response.py --h5-dir results/h5 --out-dir results/analysis
python plot_comparison.py --h5-dir results/h5 --out-dir results/figures \
  --analysis-dir results/analysis --sobol-ids 19,37,36,10,44
```

Per-Sobol figures: `profile_tf_panel_sobolNN_M1.png` (top: Vs profiles a–c; bottom: full-width TF).  
Cross-Sobol metrics: `tf_peak_*_all_sobol.png`, `tf_band_misfit_all_sobol.png`, `tf_error_vs_sobol_params.png` (ground truth: 2D OpenSees).
## HPC

**Recommended split:** Stampede3 runs **Hallal + Pretell + `opensees_2d`**. Run **`grf_2d` locally** with GIFNO (no weights in git).

After changing Pretell sampling / adding `opensees_2d`, **re-submit Stampede with `FORCE_RERUN=1`** for pretell (+ new ops2d). Hallal H5s remain index-compatible; local GIFNO `grf_2d` indices are unchanged.

```bash
chmod +x submit_stampede3_opensees.sh
FORCE_RERUN=1 ./submit_stampede3_opensees.sh -N 8 --ntasks-per-node=48 -t 48:00:00
```

Defaults: `RV_SKIP_METHODS=grf_2d`, `RV_USE_SURROGATE_2D=0`.

### Local — `grf_2d` surrogate

```bash
./submit_local.sh --full --grf-only --n-seeds 40 --jobs 6 --no-analyze
```

## Modules

- `sobol_base_cases.py` — 4D Sobol CSV (`rv_sobol_base_cases.csv`)
- `manifest.py` — index → case mapping, NO grid constants, Pretell sampling
- `run_experiment.py` — OpenSees driver (+ GIFNO for `grf_2d`)
- `surrogate_2d.py` — GIFNO inference
- `submit_stampede3_opensees.sh` — Stampede Hallal+Pretell+OPS2D launcher
- `seiskit/profile_randomization.py` — Toro / Passeri
