# Damping Study Experiment

This experiment compares three damping methods for 2D site response analysis:
- **Model A (global_avg)**: Global Average Damping - harmonic mean of Q values from soil layer only, applied to all soil elements
- **Model B (elemental_varying)**: Elemental Varying Damping - each element gets damping based on its Vs and Q
- **Model C (elemental_mass_only)**: Elemental Mass-Only Damping - mass-proportional only, per element
- **Model D (Normal)**: Uniform Rayleigh Damping - similar to cases proposed in box folder (0.75% at 0.75 and 8.25 Hz)
## Experiment Parameters

- **Discretization**: 2x2_4node only (2m × 2m elements with 4-node quadrilateral elements)
- **Height**: Fixed at 75m (Layer 1 = 75m, Layer 2 = 75m, Total depth = 150m)
- **rH/CV combinations**: 
  - (rH=10 m, CV=0.3)
  - (rH=50 m, CV=0.1)
- **Realizations**: 5 realizations (seeds: 10, 20, 30, 40, 50)
- **Frequency**: Fixed at 3.0 Hz

## Parameter Combinations

Total combinations: **3 damping methods × 2 rH/CV combinations × 5 seeds = 30 combinations**

- **Damping methods**: 3 values (global_avg, elemental_varying, elemental_mass_only)
- **rH/CV**: 2 combinations (rH=10/CV=0.3, rH=50/CV=0.1)
- **Seeds**: 5 values (10, 20, 30, 40, 50)

## Files

- `run_experiment.py`: Main experiment script (used by both local and SLURM runs)
- `run_local.py`: Local runner script for running experiments without SLURM
- `merge_timing_data.py`: Script to merge individual task timing files into a single CSV
- `job_experiment.sh`: SLURM batch script for running array jobs

## Usage

### Local Execution

The `run_local.py` script allows you to run experiments locally (without SLURM):

```bash
# Run a single case
python run_local.py --index 0

# Run a range of cases
python run_local.py --start 0 --end 9

# Run all 30 cases sequentially
python run_local.py --all

# Run all cases in parallel (4 workers)
python run_local.py --all --parallel --workers 4

# Run specific indices
python run_local.py --indices 0 5 10 15

# Run with custom number of workers
python run_local.py --all --parallel --workers 8

# Quiet mode (reduce output)
python run_local.py --all --quiet
```

**Local Runner Options:**
- `--index N`: Run a single case with index N
- `--start N --end M`: Run a range of cases from N to M (inclusive)
- `--all`: Run all 30 cases
- `--indices N M ...`: Run specific indices
- `--parallel`: Run cases in parallel (default: sequential)
- `--workers N`: Number of parallel workers (default: 4, only with --parallel)
- `--quiet`: Reduce output verbosity
- `--total N`: Total number of cases (default: 30)

The local runner provides progress tracking and a summary of results at the end.

### Direct Execution (Single Case)

You can also run a single case directly using the main script:

```bash
python run_experiment.py --index 0
```

### SLURM Array Job

For parallel execution on a cluster with SLURM:

```bash
# Run all 30 combinations
sbatch job_experiment.sh
```

The array job will run indices 0-29 (30 tasks total).

## Output Structure

Results are organized by damping method, rH, and CV:

```
results/
├── timing_data.csv                      # Merged timing data (created by merge_timing_data.py)
├── timing_data_task_0.csv               # Individual task timing files (deleted after merging)
├── timing_data_task_1.csv
├── ...
├── global_avg/
│   └── rH_10/
│       └── CV_0.3/
│           └── 2x2_4node_global_avg_rH10_CV0.3_s10/
│               ├── Vs_realization.png
│               └── ...
│   └── rH_50/
│       └── CV_0.1/
│           └── ...
├── elemental_varying/
│   └── ...
└── elemental_mass_only/
    └── ...
```

### Timing Data

To avoid race conditions when multiple tasks run simultaneously, each task writes its timing data to a separate file: `results/timing_data_task_{index}.csv`.

After all array tasks complete, merge the individual timing files into a single CSV:

```bash
python merge_timing_data.py
```

The merge script will:
- Detect and report any missing timing files (indicating failed tasks)
- Show which indices are missing
- Merge all available timing data into `results/timing_data.csv`
- Automatically delete individual timing files after successful merge

**Options:**
- `--strict`: Fail if any expected files are missing (useful for validation)
- `--keep-on-missing`: Keep individual files if any are missing (allows investigation)
- `--results-dir DIR`: Specify custom results directory
- `--output FILE`: Specify custom output file
- `--expected-total N`: Specify expected total number of tasks (default: 30)

**Example with missing files:**
```bash
# Merge and report missing files (default behavior)
python merge_timing_data.py

# Fail if any files are missing (strict validation)
python merge_timing_data.py --strict

# Keep individual files if any are missing (for debugging)
python merge_timing_data.py --keep-on-missing
```

The merged CSV contains the following columns:
- `case_type`: Discretization case (2x2_4node)
- `damping_method`: Damping method used (global_avg, elemental_varying, elemental_mass_only)
- `rH`: Horizontal correlation length in meters
- `CV`: Coefficient of variation
- `seed`: Random seed used
- `task_id`: Unique task identifier
- `total_time_sec`: Total wall time
- `field_generation_time_sec`: Time to generate VS field
- `model_build_time_sec`: Time to build model data
- `analysis_time_sec`: Time for OpenSees analysis
- `status`: Analysis status/result

## Index Mapping

The index maps to parameter combinations as follows:

```
index = damping_idx × 10 + combo_idx × 5 + seed_idx

Where:
- damping_idx: 0 (global_avg), 1 (elemental_varying), 2 (elemental_mass_only)
- combo_idx: 0 (rH=10, CV=0.3), 1 (rH=50, CV=0.1)
- seed_idx: 0 (seed=10), 1 (seed=20), 2 (seed=30), 3 (seed=40), 4 (seed=50)
```

Example:
- Index 0: global_avg, rH=10, CV=0.3, seed=10
- Index 1: global_avg, rH=10, CV=0.3, seed=20
- Index 5: global_avg, rH=50, CV=0.1, seed=10
- Index 10: elemental_varying, rH=10, CV=0.3, seed=10
- Index 29: elemental_mass_only, rH=50, CV=0.1, seed=50

## Technical Details

### Fixed Parameters

- **Domain size**: 1500m × 150m (Lx × Lz)
- **Variability zone**: 500m wide
- **Boundary condition zones**: 500m on each side
- **Element size**: 2m × 2m
- **Layer 1**: Vs = 100 m/s, 75m thick
- **Layer 2**: Vs = 1500 m/s, 75m thick
- **Density**: 2000 kg/m³
- **Poisson's ratio**: 0.3
- **Damping ratio**: 0.0075 (for elemental methods, computed from Q)
- **Damping frequencies**: (0.75, 2.25) Hz (for Rayleigh damping)
- **Time step**: 0.01 s
- **Duration**: 15.0 s
- **Motion time shift**: 1.4 s
- **Motion frequency**: 3.0 Hz (fixed)
- **Solver**: Mumps

### Variable Parameters

- **Damping method**: global_avg, elemental_varying, elemental_mass_only
- **rH**: Horizontal correlation length (10 m, 50 m)
- **CV**: Coefficient of variation (0.3, 0.1)
- **seed**: Random seed for spatial field generation (10, 20, 30, 40, 50)

### Damping Methods

1. **Global Average (global_avg)**:
   - Calculates harmonic mean Q from soil layer elements only (Vs < 500 m/s)
   - Excludes bedrock elements (Vs >= 500 m/s) from averaging
   - Applies average damping to all soil layer elements as one region
   - Applies bedrock-specific damping to bedrock elements separately

2. **Elemental Varying (elemental_varying)**:
   - Each element gets its own damping based on its Vs value
   - Q is computed from Vs using: Q = 7.17 + 0.00276*Vs (if Vs < 800 m/s) else Q = 50.0
   - Damping ratio: xi = 1 / (2 * Q)
   - Rayleigh coefficients computed using (f1=0.75 Hz, f2=2.25 Hz)

3. **Elemental Mass-Only (elemental_mass_only)**:
   - Each element gets mass-only damping based on its Vs value
   - Q and damping ratio computed same as elemental_varying
   - Mass-only coefficients: alphaM = 2 * w_target * xi, betaK = 0.0
   - Target frequency: 3.0 Hz (matches motion frequency)

