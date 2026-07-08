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

## HPC (SLURM / Savio)

Production runs mirror `data_generation/emulator_first` (phase1 Savio2 + GNU Parallel) and `emulator_second` (phase2 HTC reruns).

```bash
cd comparison/Response_Variability
chmod +x submit_phase1.sh submit_phase2.sh submit_local.sh

# Smoke on cluster (50 indices, array 0-2, 24 sims/node)
./submit_phase1.sh --smoke

# Full campaign (15,000 indices, array 0-624)
./submit_phase1.sh

# Rerun failed indices on HTC (array task = experiment index)
./submit_phase2.sh --array=12,45,99
```

Scratch outputs (set automatically by the job scripts):

- `RV_OUTDIR` → `/global/scratch/users/$USER/rv_comparison/opensees_runs/...`
- `RV_H5_DIR` → `/global/scratch/users/$USER/rv_comparison/h5`

Single-index debug on Savio3 (Parametric_study pattern):

```bash
sbatch --array=0 job_experiment.sh
```

Per-array-task logs and failure summaries: `logs/per_idx/job_<JOB_ID>/task_<N>/joblog.tsv` and `logs/array_job_<JOB_ID>_summary.txt`.

## Modules

- `seiskit/profile_randomization.py` — 1D profile generators + vertical ACF
- `seiskit/intensity_measures.py` — PGA, Sa(T), σ_ln
- `seiskit/gof.py` — Anderson GOF metrics
