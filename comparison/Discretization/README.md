# Discretization Comparison Experiment

This experiment compares three different discretization approaches for 2D site response analysis:

1. **2x2_4node**: Baseline case with 2m × 2m elements using standard 4-node quadrilateral elements (`quad`)
2. **1x1_4node**: Fine discretization with 1m × 1m elements using 4-node quadrilateral elements (created by re-discretizing the 2×2 grid)
3. **2x2_8node**: Same 2m × 2m element size but using Enhanced Strain Quadrilateral elements (`enhancedQuad`) for higher-order accuracy

## Files

- `run_experiment.py`: Main experiment script that runs the three cases
- `re_discretization.py`: Contains the re-discretization function that expands 2×2 elements into 1×1 elements
- `merge_timing_data.py`: Script to merge individual task timing files into a single CSV (run after array job completes)
- `job_experiment.sh`: SLURM batch script for running array jobs

## Usage

### Single Case

Run a specific case with a parameter combination:

```bash
python run_experiment.py --case 2x2_4node --index 0
```

Available cases:
- `2x2_4node`: 2×2m elements, 4-node
- `1x1_4node`: 1×1m elements, 4-node (re-discretized)
- `2x2_8node`: 2×2m elements, 8-node

### Parameter Combinations

The experiment uses a simplified parameter space:
- **Combination 1**: rH = 10.0 m, CV = 0.3
- **Combination 2**: rH = 50.0 m, CV = 0.1
- **Seeds**: [10, 20, 30, 40, 50] (5 realizations per combination)

Total: 2 × 5 = 10 combinations per discretization case (indices 0-9)

### SLURM Array Job

For parallel execution with SLURM:

```bash
# Run all 10 combinations for case 2x2_4node
sbatch --array=0-9 job_experiment.sh 2x2_4node

# Run all cases (requires 3 separate array jobs)
sbatch --array=0-9 job_experiment.sh 2x2_4node
sbatch --array=0-9 job_experiment.sh 1x1_4node
sbatch --array=0-9 job_experiment.sh 2x2_8node
```

## Output Structure

Results are organized by case type:

```
results/
├── timing_data.csv                      # Merged timing data (created by merge_timing_data.py)
├── timing_data_task_2x2_4node_0.csv    # Individual task timing files (deleted after merging)
├── timing_data_task_2x2_4node_1.csv
├── timing_data_task_1x1_4node_0.csv
├── timing_data_task_1x1_4node_1.csv
├── ...                                  # (Individual files are cleaned up after merge)
├── 2x2_4node/
│   └── h5m/
│       └── rH_10/
│           └── CV_0.3/
│               └── 2x2_4node_h5m_rH10_CV0.3_s10/
│                   ├── Vs_realization.png
│                   └── ...
├── 1x1_4node/
│   └── ...
└── 2x2_8node/
    └── ...
```

### Timing Data

To avoid race conditions when multiple tasks run simultaneously, each task writes its timing data to a separate file: `results/timing_data_task_{case_type}_{index}.csv` (e.g., `timing_data_task_2x2_4node_0.csv`, `timing_data_task_1x1_4node_0.csv`). This ensures files from different case types don't overwrite each other.

After all array tasks complete, merge the individual timing files into a single CSV:

```bash
python merge_timing_data.py
```

This creates `results/timing_data.csv` and automatically deletes the individual timing files to keep the results directory clean. The merged CSV contains the following columns:
- `case_type`: Discretization case (2x2_4node, 1x1_4node, 2x2_8node)
- `base_case`: Base case name (h5m, h75m, h145m)
- `layer1_height_m`: Layer 1 height in meters
- `rH`, `CV`, `seed`: Parameter values
- `task_id`: Unique task identifier
- `total_time_sec`: Total wall time
- `field_generation_time_sec`: Time to generate VS field
- `rediscretization_time_sec`: Time for re-discretization (if applicable)
- `model_build_time_sec`: Time to build model data
- `analysis_time_sec`: Time for OpenSees analysis
- `status`: Analysis status/result

This CSV file can be used for comparing performance across different discretization approaches.

## Technical Details

### Re-discretization

The `re_discretization` function expands each 2×2 element into 4 smaller 1×1 elements by duplicating values (no averaging). This preserves the original field values while increasing resolution.

### Enhanced Strain Elements

The 2x2_8node case uses Enhanced Strain Quadrilateral elements (`enhancedQuad`), which provide higher-order accuracy through enhanced strain formulation. This improves numerical performance compared to standard `quad` elements while using the same 4-node geometry and mesh size.

## Notes

- The 1×1 case generates the field at 2×2 resolution first, then re-discretizes to 1×1
- All cases use the same spatial variability parameters for fair comparison
- Boundary conditions and analysis parameters are identical across cases

