#!/bin/bash
# Unified pipeline script for complete data generation on SLURM
# Generates materials + HF data, then compresses the entire data folder
# This is the recommended way to generate all data on HPC

set -euo pipefail

# Default configuration
DATA_DIR="${DATA_DIR:-data}"
START_IDX="${START_IDX:-1000}"
N_TEST="${N_TEST:-100}"
MODE="${MODE:-both}"
HX="${HX:-10.0}"
HX_HF="${HX_HF:-1.0}"
LX="${LX:-150.0}"
LZ="${LZ:-150.0}"
DURATION="${DURATION:-25.0}"
DT_HF="${DT_HF:-0.01}"
COMPRESS="${COMPRESS:-true}"
REMOVE_ORIGINAL="${REMOVE_ORIGINAL:-false}"
INCLUDE_TEMP="${INCLUDE_TEMP:-false}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --start_idx)
            START_IDX="$2"
            shift 2
            ;;
        --n_test)
            N_TEST="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
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
        --no-compress)
            COMPRESS="false"
            shift
            ;;
        --remove_original)
            REMOVE_ORIGINAL="true"
            shift
            ;;
        --include_temp)
            INCLUDE_TEMP="true"
            shift
            ;;
        --help)
            cat <<EOF
Usage: $0 [OPTIONS]

Unified pipeline for data generation on SLURM:
1. Generate materials and/or HF data
2. Compress entire data folder (optional)

Options:
  --data_dir DIR          Data directory (default: data)
  --start_idx IDX         Starting index for simulation IDs (default: 1000)
  --n_test N              Number of test samples to generate (default: 100)
  --mode MODE             Generation mode: "lf" (materials+LF) or "both" (materials+LF+HF, default)
  --hx SIZE               LF element size in meters (default: 10.0)
  --hx_hf SIZE            HF element size in meters (default: 1.0)
  --Lx WIDTH              Domain width in meters (default: 150.0)
  --Lz HEIGHT             Domain height in meters (default: 150.0)
  --duration SECONDS      Simulation duration in seconds (default: 25.0)
  --dt_hf SECONDS         HF time step in seconds (default: 0.01)
  --no-compress           Skip compression job
  --remove_original       Remove original files after compression
  --include_temp          Include temp_outputs directory in compression
  --help                  Show this help message

Examples:
  # Full pipeline: generate both materials and HF, then compress
  $0

  # Generate only materials + LF (no HF)
  $0 --mode lf --no-compress

  # Generate 200 samples and compress with removal
  $0 --n_test 200 --remove_original
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

echo "============================================================================"
echo "Unified Data Generation Pipeline"
echo "============================================================================"
echo "Configuration:"
echo "  Data Directory:      $DATA_DIR"
echo "  Start Index:         $START_IDX"
echo "  Number of Samples:   $N_TEST"
echo "  Mode:                $MODE"
echo "  LF Element Size:     $HX m"
echo "  HF Element Size:     $HX_HF m"
echo "  Domain Size:         ${LX}m × ${LZ}m"
echo "  Duration:            $DURATION s"
echo "  HF Time Step:        $DT_HF s"
echo "  Compression:         $COMPRESS"
if [ "$COMPRESS" = "true" ]; then
    echo "  Remove Original:     $REMOVE_ORIGINAL"
    echo "  Include Temp:        $INCLUDE_TEMP"
fi
echo "============================================================================"
echo ""

# Calculate array range (0 to N_TEST-1)
ARRAY_MAX=$((N_TEST - 1))

# Step 1: Submit data generation jobs
echo "Step 1: Submitting data generation jobs..."
export DATA_DIR START_IDX MODE HX HX_HF LX LZ DURATION DT_HF

# Update the job script array range
JOB_SCRIPT="job_generate_data.sh"
TEMP_SCRIPT=$(mktemp)
sed "s/^#SBATCH --array=.*/#SBATCH --array=0-${ARRAY_MAX}/" "${JOB_SCRIPT}" > "${TEMP_SCRIPT}"

# Submit the job
GEN_JOB_ID=$(sbatch --parsable "${TEMP_SCRIPT}")
rm "${TEMP_SCRIPT}"

if [ -z "$GEN_JOB_ID" ]; then
    echo "Error: Failed to submit data generation jobs" >&2
    exit 1
fi

echo "✓ Data generation jobs submitted: $GEN_JOB_ID"
echo ""

# Step 2: Submit compression job (if requested)
if [ "$COMPRESS" = "true" ]; then
    echo "Step 2: Submitting compression job (will run after generation completes)..."
    COMPRESS_JOB_ID=$(./submit_compress_hf.sh \
        --data_dir "$DATA_DIR" \
        --after_job "$GEN_JOB_ID" \
        --remove_original "$REMOVE_ORIGINAL" \
        --include_temp "$INCLUDE_TEMP" 2>&1 | grep "Job ID:" | awk '{print $3}')

    if [ -n "$COMPRESS_JOB_ID" ]; then
        echo "✓ Compression job submitted: $COMPRESS_JOB_ID"
        echo ""
    else
        echo "Warning: Failed to submit compression job" >&2
        echo "You can submit it manually later with:" >&2
        echo "  ./submit_compress_hf.sh --after_job $GEN_JOB_ID" >&2
        echo ""
    fi
fi

# Summary
echo "============================================================================"
echo "Pipeline Summary"
echo "============================================================================"
echo "Generation Job ID: $GEN_JOB_ID"
if [ "$COMPRESS" = "true" ] && [ -n "${COMPRESS_JOB_ID:-}" ]; then
    echo "Compression Job ID:  $COMPRESS_JOB_ID"
fi
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check generation progress:"
echo "  watch -n 30 'ls $DATA_DIR/materials/sim_*.npy 2>/dev/null | wc -l'"
echo "  watch -n 30 'ls $DATA_DIR/low_fidelity/pga/*.npy 2>/dev/null | wc -l'"
if [ "$MODE" = "both" ]; then
    echo "  watch -n 30 'ls $DATA_DIR/high_fidelity/pga/*.npy 2>/dev/null | wc -l'"
fi
echo ""
echo "View job details:"
echo "  sacct -j $GEN_JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed"
if [ "$COMPRESS" = "true" ] && [ -n "${COMPRESS_JOB_ID:-}" ]; then
    echo "  sacct -j $COMPRESS_JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed"
fi
echo "============================================================================"

