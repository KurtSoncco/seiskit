# SLURM-based HF Data Generation

This directory contains scripts for generating High-Fidelity (HF) datasets on a SLURM supercomputer system.

## Overview

The HF data generation is split into two parts:
1. **LF Data Generation** (run locally or on SLURM): Generates material grids, base motions, and runs LF simulations
2. **HF Data Generation** (run on SLURM): Runs expensive HF OpenSees simulations in parallel using SLURM array jobs

## Files

### LF Material Generation
- `generate_lf_materials_SLURM.py`: Python script that generates material grids and parameters for a single simulation
- `job_generate_lf_materials.sh`: SLURM batch script for submitting array jobs to generate materials
- `submit_lf_materials.sh`: Helper script to easily submit material generation jobs with custom parameters

### HF Generation
- `generate_data_SLURM.py`: Python script that runs a single HF simulation based on SLURM array task ID
- `job_generate_hf.sh`: SLURM batch script for submitting array jobs
- `submit_hf_jobs.sh`: Helper script to easily submit jobs with custom parameters

### Data Compression
- `compress_hf_data.py`: Python script to compress HF data into a zip archive
- `job_compress_hf.sh`: SLURM batch script for compressing HF data
- `submit_compress_hf.sh`: Helper script to submit compression jobs

### Pipeline Scripts
- `submit_hf_pipeline.sh`: All-in-one script to submit both generation and compression jobs

## Prerequisites

1. **LF material data must be available**: The HF script requires LF material grids and parameters. You have two options:

   **Option A: Generate on the cluster (Recommended)**
   - Since data files (`.npy`, `.json`) are gitignored, it's easier to generate them directly on the cluster
   - Use `job_generate_lf_materials.sh` to generate materials in parallel
   - See "Step 1" below for details

   **Option B: Transfer from local machine**
   - Generate LF data locally using `generate_data.py` or `main.py`
   - Transfer to the cluster:
     ```bash
     # From your local machine, transfer the data directory to the cluster
     rsync -avz data/materials/ user@cluster:/path/to/seiskit/emulator/data/materials/
     rsync -avz data/material_params/ user@cluster:/path/to/seiskit/emulator/data/material_params/
     ```

2. **SLURM environment**: Access to a SLURM cluster with:
   - Python environment with seiskit and OpenSeesPy installed
   - Appropriate SLURM account and partition access

## Quick Start

### Step 1: Generate LF Material Data (on the cluster)

Since data files are gitignored, it's recommended to generate material data directly on the cluster:

```bash
# On the cluster, generate materials for 100 test samples
./submit_lf_materials.sh --n_test 100 --start_idx 1000

# Or generate for 200 samples
./submit_lf_materials.sh --n_test 200 --start_idx 1000

# Custom configuration
./submit_lf_materials.sh \
    --data_dir data \
    --n_test 100 \
    --start_idx 1000 \
    --hx 10.0 \
    --hx_hf 1.0 \
    --Lx 150.0 \
    --Lz 150.0
```

This will create:
- Material grids in `data/materials/` (LF resolution)
- Material grids in `data/materials_hf/` (HF resolution, for model input)
- Material parameters in `data/material_params/`

**Note**: This only generates material data, not simulation results. The HF generation script will use these materials to run simulations.

**Alternative**: If you already have LF data locally, you can transfer it (see Prerequisites above).

### Step 2: Submit HF Generation Jobs (after materials are ready)

#### Option A: All-in-one pipeline (recommended for first-time use)

```bash
# Generate HF datasets and automatically compress when done
./submit_hf_pipeline.sh

# Generate 200 datasets with compression
./submit_hf_pipeline.sh --n_test 200

# Generate without compression
./submit_hf_pipeline.sh --no-compress

# Generate and compress with removal of originals (saves space)
./submit_hf_pipeline.sh --remove_original
```

#### Option B: Use the helper script

```bash
# Generate 100 HF datasets (default)
./submit_hf_jobs.sh

# Generate 200 HF datasets
./submit_hf_jobs.sh --n_test 200

# Custom configuration
./submit_hf_jobs.sh \
    --data_dir data \
    --test_start_idx 1000 \
    --n_test 100 \
    --hx 10.0 \
    --hx_hf 1.0 \
    --duration 25.0 \
    --dt_hf 0.01
```

#### Option C: Submit directly with sbatch

```bash
# Edit job_generate_hf.sh to adjust parameters, then:
sbatch job_generate_hf.sh
```

## Configuration

### Key Parameters

- `--data_dir`: Base data directory (default: `data`)
- `--test_start_idx`: Starting index for test data (default: `1000`, assuming 1000 train + 100 val)
- `--n_test`: Number of test samples to generate (default: `100`)
- `--hx`: LF element size in meters (default: `10.0`)
- `--hx_hf`: HF element size in meters (default: `1.0`)
- `--Lx`: Domain width in meters (default: `150.0`)
- `--Lz`: Domain height in meters (default: `150.0`)
- `--duration`: Simulation duration in seconds (default: `25.0`)
- `--dt_hf`: HF time step in seconds (default: `0.01`)

### SLURM Job Configuration

Edit `job_generate_hf.sh` to adjust:
- `--account`: Your SLURM account
- `--partition`: Partition to use (e.g., `savio3`, `savio2`)
- `--time`: Time limit per task (default: `04:00:00` = 4 hours)
- `--cpus-per-task`: CPUs per task (default: `4`)
- `--array`: Array range (automatically set by `submit_hf_jobs.sh`)

## How It Works

1. **Array Job**: SLURM creates an array of jobs, one for each test sample
   - Array index 0 → sim_id = test_start_idx + 0
   - Array index 1 → sim_id = test_start_idx + 1
   - etc.

2. **Each Task**:
   - Loads existing LF material grid from `data/materials/sim_XXXX.npy`
   - Loads material parameters from `data/material_params/sim_XXXX.json`
   - Interpolates material grid to HF resolution (150×150 for 150m domain with 1m elements)
   - Runs OpenSees HF simulation
   - Saves results to `data/high_fidelity/output_accel/sim_XXXX.npy`
   - Computes and saves PGA to `data/high_fidelity/pga/sim_XXXX.npy`

3. **Parallel Execution**: All array tasks run in parallel on different nodes/CPUs

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View specific job
squeue -j <JOB_ID>

# View output from first task
tail -f logs/array_job_<JOB_ID>_task_0.out

# View errors
tail -f logs/array_job_<JOB_ID>_task_0.err

# Check job completion and timing
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

# Count completed tasks
ls data/high_fidelity/pga/*.npy | wc -l
```

## Troubleshooting

### Job fails immediately
- **Data not found on cluster**: The most common issue is that LF material data doesn't exist on the cluster. Solutions:
  - **Option 1 (Recommended)**: Generate materials on the cluster using `./submit_lf_materials.sh`
  - **Option 2**: Transfer materials from local machine (see Prerequisites)
  - The pre-flight check will show:
    - The expected file path
    - How many files exist (if any)
    - The file range if files exist
- Check that LF data exists on cluster: `ls data/materials/sim_*.npy` (run on the cluster)
- Verify test_start_idx matches your material generation start_idx
- Check preflight output in job logs for detailed diagnostics

### Simulation timeouts
- Increase `--time` in `job_generate_hf.sh`
- Reduce `--duration` or `--dt_hf` for faster simulations
- Check if specific tasks are failing (may indicate problematic parameters)

### Missing files
- Ensure LF data generation completed successfully
- Check that material files exist for all test indices
- Verify paths in the script match your data directory structure

### OpenSees errors
- Check individual task error logs: `cat logs/array_job_<JOB_ID>_task_<N>.err`
- Verify OpenSeesPy is properly installed in the virtual environment
- Check LD_LIBRARY_PATH is set correctly

## Example Workflow

### Complete Workflow (Generate Everything on Cluster)

```bash
# 1. Generate LF material data on cluster (100 test samples, starting at index 1000)
./submit_lf_materials.sh --n_test 100 --start_idx 1000

# 2. Monitor material generation
watch -n 10 'squeue -u $USER'
watch -n 30 'ls data/materials/sim_*.npy 2>/dev/null | wc -l'

# 3. After materials are ready, submit HF generation jobs
./submit_hf_jobs.sh --n_test 100 --test_start_idx 1000

# 4. Monitor HF generation jobs
watch -n 10 'squeue -u $USER'

# 5. Check HF progress
watch -n 30 'ls data/high_fidelity/pga/*.npy 2>/dev/null | wc -l'

# 6. After completion, verify all HF data
ls data/high_fidelity/pga/*.npy | wc -l  # Should show 100 files

# 7. Compress HF data for better data management
./submit_compress_hf.sh --after_job <HF_JOB_ID>
```

### Alternative Workflow (Generate Locally, Transfer, Run HF on Cluster)

```bash
# 1. Generate LF data locally (including test set)
python main.py --mode generate --n_train 1000 --n_val 100 --n_test 100

# 2. Verify LF data exists locally
ls data/materials/sim_*.npy | wc -l  # Should show 1200 files

# 3. Transfer materials to cluster
rsync -avz data/materials/ user@cluster:/path/to/seiskit/emulator/data/materials/
rsync -avz data/material_params/ user@cluster:/path/to/seiskit/emulator/data/material_params/

# 4. On cluster, submit HF generation jobs
./submit_hf_jobs.sh --n_test 100 --test_start_idx 1000

# 5. Monitor and verify (same as above)
```

## Step 3: Compress HF Data (Optional but Recommended)

After all HF generation jobs complete, compress the data into a zip archive for better data management and storage efficiency.

### Option A: Use the helper script (recommended)

```bash
# Compress after HF generation job completes
HF_JOB_ID=$(sbatch --parsable job_generate_hf.sh)
./submit_compress_hf.sh --after_job $HF_JOB_ID

# Or compress standalone (after jobs already completed)
./submit_compress_hf.sh

# Compress and remove originals to save space (use with caution!)
./submit_compress_hf.sh --remove_original

# Include HF material grids in the archive
./submit_compress_hf.sh --include_materials

# Custom output location
./submit_compress_hf.sh --output_zip /path/to/hf_data.zip
```

### Option B: Submit directly with sbatch

```bash
# Edit job_compress_hf.sh to adjust parameters, then:
sbatch job_compress_hf.sh
```

### Compression Options

- `--data_dir`: Data directory (default: `data`)
- `--output_zip`: Output zip file path (default: `data_dir/high_fidelity.zip`)
- `--remove_original`: Remove original files after compression (saves space, use with caution!)
- `--include_materials`: Also include HF material grids in the archive
- `--after_job`: Run after specified job completes (dependency)

### Compression Benefits

- **Storage efficiency**: Compressed archives typically reduce size by 50-80%
- **Data management**: Single file is easier to transfer, backup, and archive
- **Faster I/O**: Reading from a single zip can be faster than many small files
- **Backup friendly**: Easier to backup a single archive file

### Extracting Compressed Data

```bash
# Extract all HF data
unzip data/high_fidelity.zip -d data/

# Extract specific files
unzip data/high_fidelity.zip "high_fidelity/pga/sim_*.npy" -d data/
```

## Notes

- HF simulations are computationally expensive (fine grid, small time step)
- Each HF simulation may take 1-4 hours depending on domain size and duration
- Array jobs allow parallel execution of all 100 (or more) simulations
- Failed tasks can be re-run individually by adjusting the array range
- Results are saved incrementally, so partial completion is recoverable

