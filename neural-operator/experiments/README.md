# Experiments

Small, hand-crafted OpenSees datasets that probe how the trained neural-operator
surrogate (a 2-layer, random-field-Vs, 1D-soil-over-bedrock model) handles soil
geometries it wasn't trained on. Both experiments below use the same
random-field Vs generation approach as the main training pipeline
(`neural-operator/data/`), just with a different underlying layer geometry.

Each experiment directory is self-contained:

```
<experiment>/
  manifest.py          # parameter distributions + manifest generation
  run_experiment.py     # builds the model, runs OpenSees, writes one H5 per case
  plot_profiles.py      # visual QA: plots each case's Vs realization before running OpenSees
  manifest.csv           # generated manifest (written on run)
  plots/*.png            # Vs realization plots (written by plot_profiles.py)
  results/case_*/        # raw OpenSees recorder output per case
  h5/case_*.h5            # final packaged output per case
```

## 3-Vs-layer profile (`three_layer/`)

A 3-material profile — two independently-variable soil layers over a fixed
bedrock — instead of the usual single soil layer over bedrock. Each soil layer
has its own random-field Vs variability (own CoV, correlation length `rH`,
anisotropy ratio `aHV`, and RNG seed); both layer/bedrock interfaces are flat
(no waviness — see "Design notes" below).

**3 cases**, each a Sobol-sampled draw from:

| Parameter | Distribution | Notes |
|---|---|---|
| `Vs1` (top layer) | lognormal, bounds [100, 230] m/s | |
| `Vs_mid` (middle layer) | lognormal, bounds [450, 560] m/s | |
| `Vs_bedrock` | fixed at 1500 m/s | not sampled, per spec |
| `H1`, `H2` (layer thicknesses) | uniform, bounds [5, 15] m each | capped so `H1 + H2 <= 30 m` total — see "Design notes" |
| `CoV1`, `CoV2` | uniform, bounds [0.1, 0.3] | same bounds as the main training pipeline, sampled independently per layer |
| `rH1`, `rH2` | uniform, bounds [10, 100] m | same bounds as the main training pipeline, sampled independently per layer |
| `aHV1`, `aHV2` | lognormal, bounds [10, 50] | same bounds as the main training pipeline, sampled independently per layer |

Run: `python three_layer/plot_profiles.py` to preview, then
`python three_layer/run_experiment.py [--index N] [--force]`.

## 2-Vs-layer profile with a dipping interface (`dipping/`)

The same background 2-layer profile used by the main training pipeline
(`Vs1`, `Vs2`, `H`, `CoV`, `rH`, `aHV` pinned at representative/median values),
but with a straight interface dipping at a fixed angle instead of the usual
randomly wavy one.

**2 cases only** — dip left-to-right (+2°) and right-to-left (−2°) — an
isolated comparison where the dip direction is the only thing that differs.

| Parameter | Value | Notes |
|---|---|---|
| `Vs1`, `Vs2` | medians of the main pipeline's lognormal distributions | fixed, not sampled |
| `H` | 15 m | fixed |
| `CoV`, `rH`, `aHV` | midpoints of the main pipeline's bounds | fixed |
| `dip_angle_deg` | ±2° | small angle — see "Design notes" |
| `dip_span` | 500 m (the full `Lx_variability` width) | |

Run: `python dipping/plot_profiles.py` to preview, then
`python dipping/run_experiment.py [--index N] [--force]`.

## Design notes (why these choices, not the literal first draft)

- **No interlayer waviness for the 3-layer case.** It makes no physical sense
  for a soil layer's thickness to vary randomly meter-to-meter the way the
  single-layer training data's wavy interface does; the 3-layer interfaces are
  flat, with only intralayer (lognormal random-field) Vs variability.
- **H1 + H2 capped at 30 m.** Letting each layer thickness range up to 100 m
  (matching the main pipeline's `H` bounds) produced domains up to ~180 m deep.
  At that size a single OpenSees run was projected to take **1-2 days**
  (estimated from calibrating against the dipping case's measured ~2 s/step at
  a 25 m domain). Capping total soil thickness at 30 m keeps runtime on the
  order of the dipping case (~1.5-2 hours per case).
- **Dip angle is small (2°), not 5°.** A dip angle is the true (vertical) angle
  of the interface, not a horizontal drift rate. At 5° across the full 500 m
  variability width the interface depth swings by about ±22 m — larger than
  the 15 m soil layer itself, which isn't physically plausible for a soil
  deposit (most of the domain degenerates to solid bedrock or solid soil,
  outcropping at the surface). At 2° the swing is a much more reasonable
  ~±8.7 m, comfortably inside the domain.
- **The "10 examples at H=15 m" idea from the original draft is out of
  scope.** The existing large-scale single-layer Sobol dataset
  (`neural-operator/data/`) already has ample coverage at low `H`, including
  the fastest-response regime — no new simulations are needed for that case.

## Outputs

Each case produces one `h5/case_N.h5` with:
- `params` (group): the sampled/fixed physical parameters for that case.
- `Vs_realization_2D`, `Damping_zeta` (datasets): the full 2D grids used in the analysis.
- `grid` (group): domain/mesh metadata (`Lx`, `Lz`, `dx`, `dz`, `dt`, ...).
- `recorders/accel/{time,data}`: recorded acceleration time histories.
