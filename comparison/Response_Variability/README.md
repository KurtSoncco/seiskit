# Response_Variability Comparison

Response-focused benchmark comparing:

| Arm | Method | Description |
|-----|--------|-------------|
| `grf_2d` | Proposed 2D GRF (reference) | Flat interface, `rV=0.6 m`, `rH=30 m`, `aHV=50` |
| `delatorre_2d` | de la Torre protocol | Same simulation; spatial averaging in post-processing |
| `hallal_vs` | 1D Vs randomization | Toro (1995) / Strata-style VsRand |
| `hallal_tts` | 1D travel-time randomization | `σ_ln(t_ts)=0.02` |
| `hallal_dmin` | Damping modification | Mean profile + elevated ζ |

## Comparison philosophy (TF-first)

Under **linear viscoelastic / small-strain** site response (as in Hallal et al. 2022 and de la Torre et al. 2022), the site transfer function

\[
AF(f) = |FAS_{surface}(f)| / |FAS_{base}(f)|
\]

is an **intrinsic property of the soil profile** (and of 2D scattering geometry). It does **not** depend on which ground motion you drive with — only on estimation quality of the FAS ratio.

So the primary scientific comparison is on **ensemble AF(f)**:

- median `|AF(f)|` and peak metrics (`f_peak`, `A_peak`)
- `σ_ln AF(f)` across seeds (aleatory TF variability)
- Anderson-style GOF on `ln AF` near `f_peak`

Sa / PGA remain secondary sanity checks (they fold AF with the input spectrum).

**Implication for the factorial design:** one broadband drive is enough for TF comparison; a five-motion suite is unnecessary for Phase 1.

## Primary cell (current)

- `H=15 m`, **`Vs1=230 m/s` only**, `CV=0.2`, `rH=30 m`, `rV=0.6 m` (`aHV=50`)
- `dx=dz=0.5 m`, `Lx_var=200 m`, `BC=100 m`
- Full: **5 methods × 1 motion (`M1`) × 200 seeds = 1,000**
- Smoke (`RV_SMOKE=1`): 5 × 1 × 10 = **50**

## Quick start (local smoke)

```bash
cd comparison/Response_Variability
chmod +x submit_local.sh
./submit_local.sh
```

Generated artifacts (`results/h5`, `results/idx_*`, `logs/`) are gitignored. Re-run smoke after code changes.

## Analysis

```bash
python analyze_response.py --h5-dir results/h5 --out-dir results/analysis
python plot_comparison.py --h5-dir results/h5 --out-dir results/figures
```

Primary figure: `results/figures/af_method_subplots_Vs1230_M1.png` (AF median + σ_ln / ratio per method).

Explainability figures:

- `hallal_profiles_Vs1230.png` — base Vs1 profile vs VsRand / ttsRand realizations (Vs vs depth)
- `grf2d_explainability_Vs1230_s1.png` — 2D GRF field with highlighted center-column extraction vs base template

## HPC (SLURM / Savio)

```bash
cd comparison/Response_Variability
chmod +x submit_phase1.sh submit_phase2.sh

./submit_phase1.sh --smoke    # 50 indices, array 0-2
./submit_phase1.sh            # 1,000 indices, array 0-41
./submit_phase2.sh --array=12,45,99
```

Scratch:

- `RV_OUTDIR` → `/global/scratch/users/$USER/rv_comparison/opensees_runs/...`
- `RV_H5_DIR` → `/global/scratch/users/$USER/rv_comparison/h5`

## Modules

- `seiskit/profile_randomization.py` — full Toro (NHPP + bedrock depth + Vs) and Passeri (NHPP + tts + bedrock merge)
- `seiskit/intensity_measures.py` — PGA, Sa(T), σ_ln
- `seiskit/gof.py` — Anderson GOF metrics
- `seiskit/ttf/TTF.py` — Konno–Ohmachi AF(f)
