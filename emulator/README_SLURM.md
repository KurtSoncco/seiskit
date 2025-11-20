# Data Generation Guide

This guide explains how to generate datasets for the transfer function emulator using either **local execution** or **SLURM cluster** execution.

## Quick Reference: SLURM Submission

### Easiest Method (Recommended)

```bash
# Generate 100 samples (materials + LF + HF)
./submit_data_pipeline.sh --n_test 100 --start_idx 1000

# That's it! The script handles everything automatically.
```

### Manual Method

```bash
# 1. Edit job_generate_data.sh: change --array=0-99 (for 100 samples)
# 2. Submit:
sbatch job_generate_data.sh
```

**See detailed instructions in the [SLURM Generation](#option-2-slurm-generation-large-datasets) section below.**

## Overview

The data generation pipeline creates:
1. **Materials**: Material grids (LF and HF resolution), parameters, and base motions
2. **LF Simulations**: Low-fidelity simulations (coarse grid, large time step)
3. **HF Simulations**: High-fidelity simulations (fine grid, small time step) - optional

## Files

### Core Scripts
- `generate_data.py`: Local data generation script (materials + LF + optionally HF)
- `generate_data_unified_SLURM.py`: Unified SLURM script (materials + LF + optionally HF)
- `main.py`: Main pipeline script for local execution (generate, train, evaluate)

### SLURM Job Scripts
- `job_generate_data.sh`: SLURM batch script for array jobs
- `job_compress_hf.sh`: SLURM batch script for data compression

### Helper Scripts
- `submit_data_pipeline.sh`: Unified pipeline submission script (recommended)
- `submit_compress_hf.sh`: Compression submission helper

### Compression
- `compress_data.py`: Compresses entire data folder into a zip archive

## Data Generation Workflows

### Option 1: Local Generation (Small Datasets)

Best for: Small datasets (< 50 samples), testing, or when you don't have cluster access.

#### Using `main.py` (Recommended)

```bash
# Generate complete dataset (train + val + test with LF and HF)
python main.py --mode generate --n_train 1000 --n_val 100 --n_test 100

# Generate only test set with HF
python main.py --mode generate --n_train 0 --n_val 0 --n_test 100

# Custom parameters
python main.py --mode generate \
    --n_train 1000 \
    --n_val 100 \
    --n_test 100 \
    --duration 25.0 \
    --dt_lf 0.2 \
    --dt_hf 0.01
```

#### Using `generate_data.py` Directly

```bash
# Generate training set (LF only)
python generate_data.py \
    --data_dir data \
    --n_simulations 1000 \
    --split train \
    --start_idx 0 \
    --run_hf False

# Generate test set (LF + HF)
python generate_data.py \
    --data_dir data \
    --n_simulations 100 \
    --split test \
    --start_idx 1000 \
    --run_hf True
```

**Output Structure:**
```
data/
├── materials/          # LF material grids (sim_XXXX.npy)
├── materials_hf/       # HF material grids (sim_XXXX.npy)
├── material_params/    # Material parameters (sim_XXXX.json)
├── base_motion/        # Base motion time-series (sim_XXXX.npy)
├── low_fidelity/       # LF simulation results
│   ├── output_accel/   # Acceleration outputs (sim_XXXX.npy)
│   └── pga/            # PGA values (sim_XXXX.npy)
└── high_fidelity/      # HF simulation results (if run_hf=True)
    ├── output_accel/   # Acceleration outputs (sim_XXXX.npy)
    └── pga/            # PGA values (sim_XXXX.npy)
```

### Option 2: SLURM Generation (Large Datasets)

Best for: Large datasets (100+ samples), production runs, or when you need parallel execution.

#### Prerequisites

1. **SLURM cluster access** with:
   - Python environment with seiskit and OpenSeesPy installed
   - Appropriate SLURM account and partition access
   - Virtual environment at `/global/home/users/kurtwal98/seiskit/.venv/`

2. **Data files are gitignored**: Since `.npy` and `.json` files are gitignored, generate data directly on the cluster (recommended) or transfer from local.

#### Method 1: Using Helper Script (Recommended - Easiest)

The helper script automatically configures and submits the job:

```bash
# Basic usage: Generate 100 samples (materials + LF + HF)
./submit_data_pipeline.sh --n_test 100 --start_idx 1000

# The script will:
# 1. Update the array range in job_generate_data.sh
# 2. Set all environment variables
# 3. Submit the job with sbatch
# 4. Show you the job ID and monitoring commands
```

**Example Output:**
```
============================================================================
Unified Data Generation Pipeline
============================================================================
Configuration:
  Data Directory:      data
  Start Index:         1000
  Number of Samples:   100
  Mode:                both
  ...
============================================================================

Step 1: Submitting data generation jobs...
✓ Data generation jobs submitted: 12345

Step 2: Submitting compression job (will run after generation completes)...
✓ Compression job submitted: 12346

============================================================================
Pipeline Summary
============================================================================
Generation Job ID: 12345
Compression Job ID: 12346

Monitor jobs with:
  squeue -u $USER
...
```

**More Examples:**

```bash
# Generate only materials + LF (no HF, no compression)
./submit_data_pipeline.sh --mode lf --n_test 100 --start_idx 1000 --no-compress

# Generate 200 samples with custom parameters
./submit_data_pipeline.sh \
    --n_test 200 \
    --start_idx 1000 \
    --hx 10.0 \
    --hx_hf 1.0 \
    --duration 25.0

# Generate and compress with removal of originals (saves space)
./submit_data_pipeline.sh --n_test 100 --remove_original
```

#### Method 2: Manual Submission with sbatch

For more control or custom configurations:

**Step-by-step:**

1. **Edit `job_generate_data.sh`** to set the array range:
   ```bash
   # Open the file
   nano job_generate_data.sh
   
   # Change this line (example: 100 samples = indices 0-99)
   #SBATCH --array=0-99
   ```

2. **Set environment variables** (optional - can also edit script directly):
   ```bash
   export MODE=both          # "lf" or "both"
   export START_IDX=1000     # Starting simulation index
   export DATA_DIR=data      # Data directory
   ```

3. **Submit the job:**
   ```bash
   sbatch job_generate_data.sh
   ```
   
   Output: `Submitted batch job 12345`

4. **Monitor the job:**
   ```bash
   # Check status
   squeue -u $USER
   
   # View logs
   tail -f logs/array_job_12345_task_0.out
   ```

**Alternative: Set variables inline (one-liner):**

```bash
MODE=both START_IDX=1000 DATA_DIR=data sbatch job_generate_data.sh
```

#### Method 3: Submit with Custom Array Range (Override)

You can override the array range in the script:

```bash
# Generate 100 samples (indices 0-99)
sbatch --array=0-99 job_generate_data.sh

# Generate 50 samples starting at index 2000
START_IDX=2000 sbatch --array=0-49 job_generate_data.sh

# Generate specific indices only (e.g., 0, 5, 10, 15)
sbatch --array=0,5,10,15 job_generate_data.sh

# Generate with step (e.g., every 5th index: 0, 5, 10, 15, ...)
sbatch --array=0-99:5 job_generate_data.sh
```

**Note:** When using `--array` override, make sure `START_IDX` is set correctly so that:
- `sim_id = START_IDX + array_index`
- For example: `START_IDX=1000` with `--array=0-99` generates `sim_1000` to `sim_1099`

#### Generation Modes

The unified script supports two modes:

- **`lf`**: Generates materials + runs LF simulation only
  - Faster, suitable for training data
  - Outputs: materials, LF simulation results
  
- **`both`** (default): Generates materials + runs LF + HF simulations
  - Complete pipeline, suitable for test/validation data
  - Outputs: materials, LF simulation results, HF simulation results

## Configuration Parameters

### Key Parameters

- `--data_dir`: Base data directory (default: `data`)
- `--start_idx`: Starting index for simulation IDs (default: `1000`)
- `--n_test`: Number of samples to generate (default: `100`)
- `--mode`: Generation mode - `"lf"` or `"both"` (default: `"both"`)
- `--hx`: LF element size in meters (default: `10.0`)
- `--hx_hf`: HF element size in meters (default: `1.0`)
- `--Lx`: Domain width in meters (default: `150.0`)
- `--Lz`: Domain height in meters (default: `150.0`)
- `--duration`: Simulation duration in seconds (default: `25.0`)
- `--dt_lf`: LF time step in seconds (default: `0.2`)
- `--dt_hf`: HF time step in seconds (default: `0.01`)

### SLURM Job Configuration

Edit `job_generate_data.sh` to adjust:
- `--account`: Your SLURM account (default: `fc_tfsurrogate`)
- `--partition`: Partition to use (default: `savio2`)
- `--time`: Time limit per task (default: `04:00:00` = 4 hours)
- `--cpus-per-task`: CPUs per task (default: `2`)
- `--array`: Array range (e.g., `0-99` for 100 samples)

## How It Works

### Local Generation

1. **Sequential execution**: Runs one simulation at a time
2. **Each simulation**:
   - Generates material grid and parameters
   - Runs LF simulation (coarse grid, large time step)
   - Optionally runs HF simulation (fine grid, small time step)
   - Saves all outputs to `data/` directory

### SLURM Generation

1. **Array Job**: SLURM creates an array of jobs, one for each sample
   - Array index 0 → sim_id = start_idx + 0
   - Array index 1 → sim_id = start_idx + 1
   - etc.

2. **Each Task** (for mode="both"):
   - **Step 1**: Generates material grid, parameters, and base motion
   - **Step 2**: Runs LF simulation (coarse grid, large time step)
   - **Step 3**: Runs HF simulation (fine grid, small time step)
   - Saves all outputs incrementally

3. **Parallel Execution**: All array tasks run in parallel on different nodes/CPUs

## Monitoring Jobs

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View specific job
squeue -j <JOB_ID>

# View job details
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

### View Logs

```bash
# View output from first task
tail -f logs/array_job_<JOB_ID>_task_0.out

# View errors
tail -f logs/array_job_<JOB_ID>_task_0.err

# View all task outputs
ls logs/array_job_<JOB_ID>_task_*.out
```

### Check Progress

```bash
# Count completed materials
ls data/materials/sim_*.npy 2>/dev/null | wc -l

# Count completed LF simulations
ls data/low_fidelity/pga/*.npy 2>/dev/null | wc -l

# Count completed HF simulations
ls data/high_fidelity/pga/*.npy 2>/dev/null | wc -l

# Watch progress in real-time
watch -n 30 'ls data/low_fidelity/pga/*.npy 2>/dev/null | wc -l'
```

## Data Compression

After data generation, compress the entire `data` folder for better storage and transfer:

### Using Helper Script

```bash
# Compress after generation job completes
GEN_JOB_ID=<your_job_id>
./submit_compress_hf.sh --after_job $GEN_JOB_ID

# Compress standalone (after jobs completed)
./submit_compress_hf.sh

# Compress and remove originals (saves space, use with caution!)
./submit_compress_hf.sh --remove_original

# Include temp_outputs directory
./submit_compress_hf.sh --include_temp

# Custom output location
./submit_compress_hf.sh --output_zip /path/to/data.zip
```

### Compression Benefits

- **Storage efficiency**: Typically 50-80% size reduction
- **Data management**: Single file is easier to transfer, backup, and archive
- **Faster I/O**: Reading from a single zip can be faster than many small files

### Extracting Compressed Data

```bash
# Extract all data
unzip data.zip -d .

# Extract specific directories
unzip data.zip "data/materials/*" -d .
```

## Complete Workflow Examples

### Example 1: Local Generation (Small Dataset)

```bash
# Generate 50 test samples locally with HF
python main.py --mode generate \
    --n_train 0 \
    --n_val 0 \
    --n_test 50 \
    --test_start_idx 0

# Verify
ls data/materials/sim_*.npy | wc -l      # Should show 50
ls data/low_fidelity/pga/*.npy | wc -l   # Should show 50
ls data/high_fidelity/pga/*.npy | wc -l  # Should show 50
```

### Example 2: SLURM Generation (Large Dataset)

```bash
# On cluster: Generate 200 test samples (materials + LF + HF)
./submit_data_pipeline.sh \
    --n_test 200 \
    --start_idx 1000 \
    --mode both

# Monitor
watch -n 30 'ls data/high_fidelity/pga/*.npy 2>/dev/null | wc -l'

# After completion, compress
./submit_compress_hf.sh --after_job <JOB_ID>
```

### Example 3: Generate Training Data on SLURM (LF Only)

```bash
# Generate 1000 training samples (materials + LF, no HF)
./submit_data_pipeline.sh \
    --n_test 1000 \
    --start_idx 0 \
    --mode lf \
    --no-compress

# Monitor LF progress
watch -n 30 'ls data/low_fidelity/pga/*.npy 2>/dev/null | wc -l'
```

## Troubleshooting

### Job fails immediately

- **Pre-flight check fails**: Check Python and OpenSees installation
  ```bash
  tail logs/array_job_<JOB_ID>_task_0.out | grep PREFLIGHT
  ```

- **Module errors**: Verify virtual environment path in job script
  ```bash
  # Check if venv exists
  ls /global/home/users/kurtwal98/seiskit/.venv/bin/python
  ```

### Simulation timeouts

- **Increase time limit**: Edit `--time` in `job_generate_data.sh` (e.g., `--time=08:00:00`)
- **Reduce simulation complexity**: 
  - Reduce `--duration` (e.g., 15.0s instead of 25.0s)
  - Increase `--dt_hf` (e.g., 0.02s instead of 0.01s)
- **Check specific failing tasks**: May indicate problematic parameters

### Missing files

- **Materials not found**: Ensure materials are generated first (mode always generates materials)
- **Verify paths**: Check that `data_dir` is correct and accessible
- **Check file permissions**: Ensure directories are writable

### OpenSees errors

- **Check individual task logs**: 
  ```bash
  cat logs/array_job_<JOB_ID>_task_<N>.err
  ```
- **Verify OpenSeesPy installation**: 
  ```bash
  python -c "import openseespy.opensees as ops; print('OK')"
  ```
- **Check LD_LIBRARY_PATH**: Verify it's set correctly in job script

### Performance Tips

- **For large datasets**: Use SLURM array jobs (parallel execution)
- **For testing**: Use local generation (faster iteration)
- **Storage**: Compress data after generation to save space
- **Partial completion**: Results are saved incrementally, so you can resume failed tasks

## Notes

- **LF simulations**: Faster, suitable for training (coarse grid: 10m, time step: 0.2s)
- **HF simulations**: Expensive, suitable for test/validation (fine grid: 1m, time step: 0.01s)
- **Each HF simulation**: May take 1-4 hours depending on domain size and duration
- **Array jobs**: Allow parallel execution of all simulations simultaneously
- **Failed tasks**: Can be re-run individually by adjusting the array range
- **Incremental saves**: Results are saved as they complete, so partial completion is recoverable
