#!/bin/bash
# Helper script to submit HF generation jobs to SLURM
# This script makes it easy to customize and submit HF data generation jobs

set -euo pipefail

# Default configuration
DATA_DIR="${DATA_DIR:-data}"
TEST_START_IDX="${TEST_START_IDX:-1000}"
N_TEST="${N_TEST:-100}"
HX="${HX:-10.0}"
HX_HF="${HX_HF:-1.0}"
LX="${LX:-150.0}"
LZ="${LZ:-150.0}"
DURATION="${DURATION:-25.0}"
DT_HF="${DT_HF:-0.01}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --test_start_idx)
            TEST_START_IDX="$2"
            shift 2
            ;;
        --n_test)
            N_TEST="$2"
            shift 2
            ;;
        --hx)
            HX="$2"
            shift 2
            ;;
        --hx_hf)
            HX_HF="$2"
            shift 2
            ;;
        --Lx)
            LX="$2"
            shift 2
            ;;
        --Lz)
            LZ="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --dt_hf)
            DT_HF="$2"
            shift 2
            ;;
        --help)
            cat <<EOF
Usage: $0 [OPTIONS]

Submit HF data generation jobs to SLURM.

Options:
  --data_dir DIR          Data directory (default: data)
  --test_start_idx IDX    Starting index for test data (default: 1000)
  --n_test N              Number of test samples to generate (default: 100)
  --hx SIZE               LF element size in meters (default: 10.0)
  --hx_hf SIZE            HF element size in meters (default: 1.0)
  --Lx WIDTH              Domain width in meters (default: 150.0)
  --Lz HEIGHT              Domain height in meters (default: 150.0)
  --duration SECONDS      Simulation duration in seconds (default: 25.0)
  --dt_hf SECONDS         HF time step in seconds (default: 0.01)
  --help                  Show this help message

Examples:
  # Generate 100 HF datasets (default)
  $0

  # Generate 200 HF datasets
  $0 --n_test 200

  # Custom data directory and test start index
  $0 --data_dir /path/to/data --test_start_idx 2000 --n_test 150

Environment variables can also be used:
  DATA_DIR, TEST_START_IDX, N_TEST, HX, HX_HF, LX, LZ, DURATION, DT_HF

Note: This script creates a temporary job script with your parameters.
      The array range is automatically set to 0-(\$N_TEST-1).
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# Calculate array range (0 to N_TEST-1)
ARRAY_MAX=$((N_TEST - 1))

echo "============================================================================"
echo "HF Data Generation Job Submission"
echo "============================================================================"
echo "Configuration:"
echo "  Data Directory:      $DATA_DIR"
echo "  Test Start Index:    $TEST_START_IDX"
echo "  Number of Samples:   $N_TEST"
echo "  Array Range:         0-$ARRAY_MAX"
echo "  LF Element Size:     $HX m"
echo "  HF Element Size:     $HX_HF m"
echo "  Domain Size:         ${LX}m × ${LZ}m"
echo "  Duration:            $DURATION s"
echo "  HF Time Step:        $DT_HF s"
echo "============================================================================"
echo ""

# Check if job script exists
JOB_SCRIPT="job_generate_hf.sh"
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Error: Job script '$JOB_SCRIPT' not found in current directory" >&2
    exit 1
fi

# Create a temporary job script with customized parameters
TEMP_JOB_SCRIPT=$(mktemp)
trap "rm -f $TEMP_JOB_SCRIPT" EXIT

# Read the original job script and modify the array range and Python command
sed -e "s/--array=0-99/--array=0-${ARRAY_MAX}/" \
    -e "s|--data_dir data|--data_dir ${DATA_DIR}|" \
    -e "s|--test_start_idx 1000|--test_start_idx ${TEST_START_IDX}|" \
    -e "s|--hx 10.0|--hx ${HX}|" \
    -e "s|--hx_hf 1.0|--hx_hf ${HX_HF}|" \
    -e "s|--Lx 150.0|--Lx ${LX}|" \
    -e "s|--Lz 150.0|--Lz ${LZ}|" \
    -e "s|--duration 25.0|--duration ${DURATION}|" \
    -e "s|--dt_hf 0.01|--dt_hf ${DT_HF}|" \
    "$JOB_SCRIPT" > "$TEMP_JOB_SCRIPT"

# Make it executable
chmod +x "$TEMP_JOB_SCRIPT"

# Submit the job
echo "Submitting job to SLURM..."
JOB_ID=$(sbatch "$TEMP_JOB_SCRIPT" | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo ""
    echo "✓ Job submitted successfully!"
    echo "  Job ID: $JOB_ID"
    echo "  Array Range: 0-$ARRAY_MAX"
    echo ""
    echo "Monitor job status with:"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "View job output:"
    echo "  tail -f logs/array_job_${JOB_ID}_task_0.out"
    echo ""
    echo "Check job completion:"
    echo "  sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed"
else
    echo "Error: Failed to submit job" >&2
    exit 1
fi
