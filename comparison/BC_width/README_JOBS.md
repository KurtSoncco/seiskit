# BC Width Experiment - Job Submission Guide

## Overview
This experiment tests the effect of boundary condition (BC) width on seismic analysis results. It has 14 total combinations:
- 2 Lx variability values: 800m, 100m
- 7 BC width values: 0, 100, 200, 300, 400, 500, 1000m
- Total = 2 × 7 = 14 combinations

## Job Scripts

### 1. `job_experiment.sh` - Array Job (Recommended for HPC)
- Runs 14 tasks in parallel using SLURM array job system
- Each task processes one combination of Lx_var and BC_width
- **Submission**: `sbatch job_experiment.sh`
- **Time**: ~2 hours per task (max 2h)
- **Output**: Results in `results/Lx_{Lx_var}/BC_width_{BC_width}/`

### 2. `job_plot.sh` - Plotting Only
- Generates comparison plots from completed analysis results
- Must be run AFTER all array tasks complete
- **Submission**: `sbatch job_plot.sh`
- **Time**: ~30 minutes
- **Output**:
  - `transfer_functions_comparison.html`
  - `acceleration_time_histories_comparison.html`
  - `*_surface_nodes_acceleration_stacked.png`

### 3. `submit_jobs.sh` - Full Workflow (Sequential)
- Runs all 14 tasks sequentially, then generates plots
- For testing/development on a single node
- **Note**: This is slower than using array jobs, but simpler
- **Submission**: `sbatch submit_jobs.sh`
- **Time**: ~3-4 hours total

## Recommended Workflow

### For HPC Cluster (Production):

#### Step 1: Submit Array Job
```bash
sbatch job_experiment.sh
```

This will submit job with ID, e.g., `12345678`

#### Step 2: Monitor Progress
```bash
# Check job status
squeue -u $USER

# Check output files
ls -lh array_job_*_task_*.out

# Check if results are being generated
find results -name "*.txt" | wc -l
```

#### Step 3: Check Timing Results (Optional)
```bash
# View timing summary for all tasks
./view_timing_results.sh
```

#### Step 4: After All Tasks Complete, Submit Plotting Job
```bash
sbatch job_plot.sh
```

### For Local Testing/Development:

#### Run Individual Task
```bash
# Activate environment first
source ../../.venv/bin/activate

# Run specific index
python run_experiment.py --index 0    # First task
python run_experiment.py --index 10    # Task 10

# After all tasks complete, generate plots
python run_experiment.py --plot
```

## Output Structure

```
results/
├── Lx_800/
│   ├── BC_width_0/
│   │   ├── 800_BC_width_0/
│   │   │   ├── soil_base_dof1_accel.txt
│   │   │   ├── soil_top_dof1_accel.txt
│   │   │   ├── surface_nodes_dof1_accel.txt
│   │   │   └── ...
│   │   └── Vs_realization.png
│   ├── BC_width_100/
│   └── ...
└── Lx_100/
    ├── BC_width_0/
    └── ...
```

## Index Mapping

| Index | Lx_var | BC_width | Total Width |
|-------|--------|----------|-------------|
| 0     | 800    | 0        | 800         |
| 1     | 800    | 100      | 1000        |
| 2     | 800    | 200      | 1200        |
| 3     | 800    | 300      | 1400        |
| 4     | 800    | 400      | 1600        |
| 5     | 800    | 500      | 1800        |
| 6     | 800    | 1000     | 2800        |
| 7     | 100    | 0        | 100         |
| 8     | 100    | 100      | 300         |
| 9     | 100    | 200      | 500         |
| 10    | 100    | 300      | 700         |
| 11    | 100    | 400      | 900         |
| 12    | 100    | 500      | 1100        |
| 13    | 100    | 1000     | 2100        |

## Monitoring and Timing

### View Timing Results
Use `view_timing_results.sh` to check the status and timing of all array job tasks:

```bash
./view_timing_results.sh
```

This script shows:
- Individual task status and timing
- Task parameters (Lx_var, BC_width, total width)
- Overall statistics (min, max, avg duration)
- Missing/incomplete tasks
- Quick commands for detailed inspection

#### Parallel vs Sequential Comparison
If you run both `job_experiment.sh` (array job) and `submit_jobs.sh` (sequential), the script will automatically detect both and provide a performance comparison:
- **Speedup**: How much faster parallel execution is
- **Efficiency**: How effectively parallel resources are used
- **Time saved**: Total time reduction with parallel execution

Example:
```bash
# Run parallel array job
sbatch job_experiment.sh

# Run sequential job
sbatch submit_jobs.sh

# Compare results
./view_timing_results.sh
```

Example output:
```
Task Breakdown:
Index | Lx_var | BC_width | Total Width | Status
------|--------|----------|-------------|--------
    0 |    800 |        0 |         800 | ✓ COMPLETED
    1 |    800 |      100 |        1000 | ✓ COMPLETED
    ...

============================================================================
Parallel vs Sequential Execution Comparison
============================================================================
Parallel execution (array job):
  Tasks completed: 14 / 14
  Longest task: 1845s (30m 45s)
  All tasks run simultaneously on different nodes

Sequential execution (submit_jobs.sh):
  Total time: 12847s (214m 7s)
  Tasks run one after another on same node

Performance Analysis:
  Speedup: 6.96x
  Efficiency: 49.7%
  Time saved: 11002s (183m 22s)

📊 Summary:
  ✓ Good parallelization with 6.96x speedup
```

## Important Notes

1. **Vs_min for Transfer Functions**: The plotting automatically extracts Vs_min from the 800m realization for proper transfer function calculation.

2. **Seed**: All tasks use the same random seed (42) to ensure consistency in the gaussian field generation.

3. **Array Job Dependencies**: To automatically run plotting after array job completes:
   ```bash
   sbatch --dependency=afterok:12345678 job_plot.sh
   ```
   (Replace 12345678 with your array job ID)

4. **Check Results**: 
   - Base and top accelerations are recorded at the soil layer boundaries
   - Surface accelerations are recorded for all surface nodes
   - Transfer functions compare Top/Base acceleration response

