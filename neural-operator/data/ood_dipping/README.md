# Tier B OOD: soil–bedrock dipping

OpenSees 2D data campaign under `neural-operator/data/ood_dipping/`.
**960** runs = **32** Sobol physics × **30** Sobol RF seed levels.

## Physics axes (7D)

| Axis | Bounds |
|------|--------|
| `Vs1` | lognormal [100, 360] m/s |
| `H` | uniform **[25, 60] m** (min 25 m so center stays soil under dip) |
| `rH` | uniform [10, 100] m |
| `aHV` | lognormal [10, 50] |
| `dip_angle_deg` | uniform [-3, 3]° (soil–bedrock only) |
| `CoV` | uniform [0.1, 0.3] |
| `Vs2` | lognormal [760, 1500] m/s |

`dip_span = 500 m`, no wave. Full-span drop at 3° ≈ 26.2 m. `Lz` = deepest interface + ≥20 m bedrock.

## Local

```bash
python neural-operator/data/ood_dipping/plot_sobol_distribution.py
python neural-operator/data/ood_dipping/run_experiment.py --index 0 --force --overwrite-manifest
```

Outputs: `h5/run_{N}.h5`.

## Stampede3

```bash
# Smoke uses 12 h walltime (dipping OpenSees is slower than flat 3-layer).
bash neural-operator/data/ood_dipping/submit_full.sh smoke
bash neural-operator/data/ood_dipping/submit_full.sh production
sbatch neural-operator/data/ood_dipping/stampede3_resume_run.slurm
INDEX=0 sbatch neural-operator/data/ood_dipping/stampede3_single_index.sh
```

To finish only missing smoke indices without regenerating the manifest:
```bash
FORCE_RERUN=1 OOD_OVERWRITE_MANIFEST=0 \
  sbatch -N 1 --ntasks-per-node=2 -t 12:00:00 \
  neural-operator/data/ood_dipping/stampede3_resume_run.slurm
```

Scratch default: `$SCRATCH/opensees_ood_dipping/{h5,raw_runs}`.
Env: `OOD_PHYSICS_COUNT`, `OOD_SEED_LEVELS`, `OOD_OVERWRITE_MANIFEST`, `FORCE_RERUN`, `OOD_H5_DIR`, `OOD_OUTDIR`.
