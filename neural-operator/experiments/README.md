# Experiments

OpenSees datasets that probe how the trained neural-operator surrogate (a
2-layer, random-field-Vs, 1D-soil-over-bedrock model) handles soil geometries
it wasn't trained on. Both experiments below use the same random-field Vs
generation approach as the main training pipeline (`neural-operator/data/`),
just with a different underlying layer geometry.

Each experiment directory is self-contained:

```
<experiment>/
  manifest.py              # parameter distributions + manifest generation
  run_experiment.py         # builds the model, runs OpenSees, writes one H5 per case
  plot_profiles.py          # visual QA: plots Vs realizations before running OpenSees
  stampede3_full_run.slurm  # TACC Stampede3 pylauncher production job
  stampede3_resume_run.slurm
  stampede3_single_index.sh
  submit_full.sh            # convenience wrapper (smoke | production)
  manifest.csv              # generated manifest (written on run)
  plots/*.png
  results/case_*/           # raw OpenSees recorder output per case
  h5/case_*.h5              # final packaged output per case
```

## 3-Vs-layer profile (`three_layer/`)

A 3-material profile — two independently-variable soil layers over fixed
bedrock — instead of the usual single soil layer over bedrock. Each soil layer
has its own random-field Vs variability; both layer/bedrock interfaces are
flat (no interlayer waviness).

**250 cases** (default): 10 Sobol topology points × 25 RF seed replicates.

| Parameter | Distribution | Notes |
|---|---|---|
| `Vs_mid` | lognormal, [450, 560] m/s | Sobol topology dim 0 |
| `H1`, `H2` | uniform, [5, 15] m each | topology dims 1–2; `H1+H2` ≤ 30 m |
| `Vs_contrast` | uniform log-ratio, [0.8, 1.6] | `log(Vs_mid/Vs1)`; topology dim 3 |
| `Vs1` | derived | `Vs_mid / exp(Vs_contrast)` |
| `Vs_bedrock` | fixed at 1500 m/s | |
| `CoV1/2`, `rH1/2`, `aHV1/2` | fixed at training midpoints | not sampled |

**Split tags** (in manifest + HDF5):
- `train` — in-distribution topology
- `extrap_test` — corner holdout: high `Vs_mid`, thin `H1+H2` (bottom quartile
  of [10, 30] m), high `Vs_contrast` (top quartile)

Run locally:
```bash
python three_layer/plot_profiles.py --overwrite-manifest
python three_layer/run_experiment.py [--index N] [--force]
```

## 2-Vs-layer profile with a dipping interface (`dipping/`)

The same background 2-layer profile used by the main training pipeline, but with
a straight interface dipping at a fixed angle instead of a randomly wavy one.
Background `(Vs1, Vs2, H)` is Sobol-sampled from the main training marginals;
dip angle is the generalization axis.

**150 cases** (default): 10 signed angles × 15 independent (Sobol background,
RF seed) pairs per angle.

| Parameter | Distribution | Notes |
|---|---|---|
| `dip_angle_deg` | fixed grid ±1…±5° | 10 signed angles |
| `Vs1`, `Vs2`, `H` | 3D Sobol, main pipeline marginals | varies per case |
| `CoV`, `rH`, `aHV` | fixed at training midpoints | |
| `dip_span` | 500 m | full `Lx_variability` width |
| `rf_seed` | 15 independent seeds per angle | |

**Split tags:**
| `split` | Angles | Count |
|---|---|---|
| `train` | ±1, ±3, ±4 | 90 |
| `interp_test` | ±2 | 30 |
| `extrap_test` | ±5 | 30 |

**H constraint for steep dips:** domain depth `Lz` is set from the deepest
interface over the 500 m dip span plus **≥20 m bedrock** below that interface at
every column (BC extensions copy edge columns). Recorders: shallow row at **y=2 m**
and surface at `y=Lz` (same as the main training pipeline).

Run locally:
```bash
python dipping/plot_profiles.py --overwrite-manifest
python dipping/run_experiment.py [--index N] [--force]
```

## TACC Stampede3 submission

Both experiments use the same pylauncher pattern as
`neural-operator/data/stampede3_full_run.slurm` (partition `skx`, account
`ECS24003`). Outputs default to `$SCRATCH/opensees_dipping` or
`$SCRATCH/opensees_three_layer`.

### Environment variables

| Variable | Dipping default | Three-layer default |
|---|---|---|
| `EXP_SAMPLES_PER_ANGLE` / `EXP_TOPOLOGY_COUNT` | 15 | 10 |
| `EXP_SEEDS_PER_ANGLE` / `EXP_RF_SEEDS` | 15 | 25 |
| `EXP_OVERWRITE_MANIFEST` | 0 | 0 |
| `FORCE_RERUN` | 0 | 0 |
| `RUN_BASE` | `$SCRATCH/opensees_dipping` | `$SCRATCH/opensees_three_layer` |
| `EXP_H5_DIR` | `$RUN_BASE/h5` | `$RUN_BASE/h5` |
| `EXP_OUTDIR` | `$RUN_BASE/raw_runs` | `$RUN_BASE/raw_runs` |

### Smoke tests

```bash
# Dipping (4 tasks)
EXP_SAMPLES_PER_ANGLE=1 EXP_SEEDS_PER_ANGLE=2 EXP_OVERWRITE_MANIFEST=1 FORCE_RERUN=1 \
  sbatch -N 1 --ntasks-per-node=4 -t 4:00:00 \
  neural-operator/experiments/dipping/stampede3_full_run.slurm

# Three-layer (4 tasks)
EXP_TOPOLOGY_COUNT=2 EXP_RF_SEEDS=2 EXP_OVERWRITE_MANIFEST=1 FORCE_RERUN=1 \
  sbatch -N 1 --ntasks-per-node=4 -t 4:00:00 \
  neural-operator/experiments/three_layer/stampede3_full_run.slurm
```

### Production runs

```bash
bash neural-operator/experiments/dipping/submit_full.sh production
bash neural-operator/experiments/three_layer/submit_full.sh production
```

Or directly:
```bash
sbatch -N 4 --ntasks-per-node=48 -t 48:00:00 \
  neural-operator/experiments/dipping/stampede3_full_run.slurm

sbatch -N 6 --ntasks-per-node=48 -t 48:00:00 \
  neural-operator/experiments/three_layer/stampede3_full_run.slurm
```

Resume incomplete indices:
```bash
sbatch neural-operator/experiments/dipping/stampede3_resume_run.slurm
sbatch neural-operator/experiments/three_layer/stampede3_resume_run.slurm
```

Single-index debug:
```bash
INDEX=0 sbatch neural-operator/experiments/dipping/stampede3_single_index.sh
```

## Design notes

- **No interlayer waviness for the 3-layer case.** Only intralayer random-field
  Vs variability; interfaces are flat.
- **H1 + H2 capped at 30 m** total to keep OpenSees runtime tractable (~1.5–2 h
  per case at moderate depth).
- **Steep dipping cases need deeper domains.** `Lz` grows with `H + 250·tan(|dip|)`
  plus 20 m minimum bedrock below the deepest interface so bedrock is always
  present in the variability window.
- **Deeper domains run slower.** Budget 48 h jobs and use the resume scripts
  for stragglers.

## Outputs

Each case produces one `h5/case_N.h5` with:
- `params` (group): physical parameters including `split` metadata.
- `Vs_realization_2D`, `Damping_zeta` (datasets).
- `grid` (group): domain/mesh metadata.
- `recorders/accel/{time,data}`: acceleration time histories.
