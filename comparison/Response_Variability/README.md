# Response_Variability Comparison

Response-focused benchmark comparing:

| Arm | Method | Description |
|-----|--------|-------------|
| `grf_2d` | Proposed 2D GRF (reference) | Flat interface, `rV=0.6 m`, `rH=30 m`, `aHV=50` |
| `delatorre_2d` | de la Torre protocol | Same simulation; spatial averaging in post-processing |
| `hallal_vs` | 1D Vs randomization | Passeri/Toro-style AR(1) profiles |
| `hallal_tts` | 1D travel-time randomization | `σ_ln(t_ts)=0.02` |
| `hallal_dmin` | Damping modification | Mean profile + elevated ζ |

## Primary cell

- `H=15 m`, `Vs1=230 m/s`, `CV=0.2`, `rH=30 m`, `rV=0.6 m` (`aHV=50`)
- `dx=dz=0.5 m`, `Lx_var=200 m`, `BC=100 m`
- 200 seeds × 5 motions × 5 methods (full); smoke = 10 seeds × 1 motion

## Quick start (local smoke)

```bash
cd comparison/Response_Variability
chmod +x submit_local.sh
./submit_local.sh 2
```

Smoke mode (`RV_SMOKE=1`): 50 cases (5 methods × 10 seeds).

## Single run

```bash
RV_SMOKE=1 python run_experiment.py --index 0
```

## Analysis

```bash
python analyze_response.py --h5-dir results/h5 --out-dir results/analysis
python plot_comparison.py --h5-dir results/h5 --out-dir results/figures
```

## HPC (SLURM)

```bash
mkdir -p logs
# Adjust --array for total_combinations() from manifest
sbatch job_experiment.sh
```

Set `RV_OUTDIR` and `RV_H5_DIR` for scratch output on clusters.

## Modules

- `seiskit/profile_randomization.py` — 1D profile generators + vertical ACF
- `seiskit/intensity_measures.py` — PGA, Sa(T), σ_ln
- `seiskit/gof.py` — Anderson GOF metrics
