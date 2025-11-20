#!/bin/bash
# All-in-one script to submit HF generation and compression jobs
# This script submits HF generation jobs and automatically sets up compression
# to run after generation completes

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
COMPRESS="${COMPRESS:-true}"
REMOVE_ORIGINAL="${REMOVE_ORIGINAL:-false}"
INCLUDE_MATERIALS="${INCLUDE_MATERIALS:-false}"

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
        --no-compress)
            COMPRESS="false"
            shift
            ;;
        --remove_original)
            REMOVE_ORIGINAL="true"
            shift
            ;;
        --include_materials)
            INCLUDE_MATERIALS="true"
            shift
            ;;
        --help)
            cat <<EOF
Usage: $0 [OPTIONS]

Submit HF generation jobs and optionally compression job.

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
  --no-compress           Skip compression job
  --remove_original       Remove original files after compression
  --include_materials     Include HF material grids in compression
  --help                  Show this help message

Examples:
  # Full pipeline: generate 100 HF datasets and compress
  $0

  # Generate without compression
  $0 --no-compress

  # Generate 200 datasets and compress with removal of originals
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
echo "HF Data Generation Pipeline"
echo "============================================================================"
echo "Configuration:"
echo "  Data Directory:      $DATA_DIR"
echo "  Test Start Index:    $TEST_START_IDX"
echo "  Number of Samples:   $N_TEST"
echo "  LF Element Size:     $HX m"
echo "  HF Element Size:     $HX_HF m"
echo "  Domain Size:         ${LX}m × ${LZ}m"
echo "  Duration:            $DURATION s"
echo "  HF Time Step:        $DT_HF s"
echo "  Compression:         $COMPRESS"
if [ "$COMPRESS" = "true" ]; then
    echo "  Remove Original:     $REMOVE_ORIGINAL"
    echo "  Include Materials:   $INCLUDE_MATERIALS"
fi
echo "============================================================================"
echo ""

# Step 1: Submit HF generation jobs
echo "Step 1: Submitting HF generation jobs..."
HF_JOB_ID=$(./submit_hf_jobs.sh \
    --data_dir "$DATA_DIR" \
    --test_start_idx "$TEST_START_IDX" \
    --n_test "$N_TEST" \
    --hx "$HX" \
    --hx_hf "$HX_HF" \
    --Lx "$LX" \
    --Lz "$LZ" \
    --duration "$DURATION" \
    --dt_hf "$DT_HF" 2>&1 | grep "Job ID:" | awk '{print $3}')

if [ -z "$HF_JOB_ID" ]; then
    echo "Error: Failed to submit HF generation jobs" >&2
    exit 1
fi

echo "✓ HF generation jobs submitted: $HF_JOB_ID"
echo ""

# Step 2: Submit compression job (if requested)
if [ "$COMPRESS" = "true" ]; then
    echo "Step 2: Submitting compression job (will run after HF generation)..."
    COMPRESS_JOB_ID=$(./submit_compress_hf.sh \
        --data_dir "$DATA_DIR" \
        --after_job "$HF_JOB_ID" \
        --remove_original "$REMOVE_ORIGINAL" \
        --include_materials "$INCLUDE_MATERIALS" 2>&1 | grep "Job ID:" | awk '{print $3}')

    if [ -n "$COMPRESS_JOB_ID" ]; then
        echo "✓ Compression job submitted: $COMPRESS_JOB_ID"
        echo ""
    else
        echo "Warning: Failed to submit compression job" >&2
        echo "You can submit it manually later with:" >&2
        echo "  ./submit_compress_hf.sh --after_job $HF_JOB_ID" >&2
        echo ""
    fi
fi

# Summary
echo "============================================================================"
echo "Pipeline Summary"
echo "============================================================================"
echo "HF Generation Job ID: $HF_JOB_ID"
if [ "$COMPRESS" = "true" ] && [ -n "${COMPRESS_JOB_ID:-}" ]; then
    echo "Compression Job ID:     $COMPRESS_JOB_ID"
fi
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check HF generation progress:"
echo "  watch -n 30 'ls $DATA_DIR/high_fidelity/pga/*.npy 2>/dev/null | wc -l'"
echo ""
echo "View job details:"
echo "  sacct -j $HF_JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed"
if [ "$COMPRESS" = "true" ] && [ -n "${COMPRESS_JOB_ID:-}" ]; then
    echo "  sacct -j $COMPRESS_JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed"
fi
echo "============================================================================"

