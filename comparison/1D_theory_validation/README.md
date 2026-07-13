# 1D Theory Validation

Compare seiskit **1D** OpenSees site response
(`boundary_condition_type="1D"`) to closed-form multilayer transfer functions
(Kramer / Thomson–Haskell) in [`seiskit.theory`](../../seiskit/theory/).

This benchmark is for **1D only**. 2D free-field analyses
(`boundary_condition_type="2D"`) are unchanged and remain the path for
laterally heterogeneous / GRF domains.

## Setup

Each case is a single soil column (`Lx = hx`) with:

- equalDOF simple-shear kinematics + bottom ASDA absorbers
- recorders at the soil–bedrock interface and free surface
- Ricker motion injected as **velocity** on ASDA `-fx`

## Cases

| Case | Soil layers | Bedrock |
|------|-------------|---------|
| `2layer` | 30 m @ 200 m/s | 1000 m/s |
| `3layer` | 15 m @ 180 + 25 m @ 300 | 1000 m/s |
| `4layer` | 10@150 + 15@220 + 20@350 | 1200 m/s |

`*_xi02` uses ξ = 0.02 in soil (OpenSees: Rayleigh `uniform_soil_only` at travel-time `(f₀, 3f₀)`).

## Pass criteria

Primary: **AF_within** = |FAS_surface / FAS_interface| vs theory, over the
Ricker-energized band:

- median relative error < 5%
- p95 relative error < 15%
- first-mode peak frequency error < 2%

For undamped cases, **AF_outcrop** = |FAS_surface / (2·a_incident)| must also
meet median/p95 < 5%/15% and peak-amplitude error < 10%.

## Run

```bash
python comparison/1D_theory_validation/run_experiment.py
python comparison/1D_theory_validation/run_experiment.py --case=2layer
python comparison/1D_theory_validation/run_experiment.py --recompute-only
```

Outputs under `results/`: `af_comparison.png`, `metrics.json`, `summary.csv`.
