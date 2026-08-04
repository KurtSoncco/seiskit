# Tier B OOD: flat 3-layer set

OpenSees 2D data campaign under `neural-operator/data/ood_three_layer/`.
**960** runs = **32** Sobol physics × **30** Sobol RF seed levels.

## Physics axes (8D)

| Axis | Bounds |
|------|--------|
| `Vs_mid` | lognormal [450, 560] m/s |
| `H1`, `H2` | uniform [5, 12] m; `H1+H2 ≤ 24` |
| `rH` | uniform [10, 100] m (shared layers) |
| `aHV` | lognormal [10, 50] (shared) |
| `Vs_contrast` | uniform [0.8, 1.6]; `Vs1 = Vs_mid/exp(contrast)` |
| `CoV` | uniform [0.1, 0.3] (shared) |
| `Vs_bedrock` | lognormal [760, 1500] m/s |

Flat interfaces, no interlayer wave. Domain: 500 m variability + 500 m BC each side.

**Total height:** `Lz = H1 + H2 + bedrock` with bedrock = 10 m and `H1+H2 ≤ 24` → **max `Lz ≈ 34 m`**.
Runtime guide (OpenSees 2D): ~50 m → 3–4 h; ~100 m → 8–9 h; this campaign (~≤34 m) → ~2–3 h/case.

## Local

```bash
# Sobol corner plot
python neural-operator/data/ood_three_layer/plot_sobol_distribution.py

# Build / refresh manifest
python -c "from pathlib import Path; import sys; sys.path.insert(0,'neural-operator/data/ood_three_layer'); from manifest import ensure_manifest; ensure_manifest(overwrite=True)"

# Single index (needs OpenSees)
python neural-operator/data/ood_three_layer/run_experiment.py --index 0 --force
```

Outputs: `h5/run_{N}.h5`, raw under `results/run_{N}/`.

## Stampede3

```bash
# Smoke uses 8 h walltime (max Lz≈34 m).
bash neural-operator/data/ood_three_layer/submit_full.sh smoke
bash neural-operator/data/ood_three_layer/submit_full.sh production
sbatch neural-operator/data/ood_three_layer/stampede3_resume_run.slurm
INDEX=0 sbatch neural-operator/data/ood_three_layer/stampede3_single_index.sh
```

Scratch default: `$SCRATCH/opensees_ood_three_layer/{h5,raw_runs}`.
Env: `OOD_PHYSICS_COUNT`, `OOD_SEED_LEVELS`, `OOD_OVERWRITE_MANIFEST`, `FORCE_RERUN`, `OOD_H5_DIR`, `OOD_OUTDIR`.
