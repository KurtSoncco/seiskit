#!/bin/bash
# Helper script to submit HF data compression job to SLURM
# Can run standalone or as a dependency job after HF generation

set -euo pipefail

# Default configuration
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_ZIP="${OUTPUT_ZIP:-}"
REMOVE_ORIGINAL="${REMOVE_ORIGINAL:-false}"
INCLUDE_TEMP="${INCLUDE_TEMP:-false}"
AFTER_JOB="${AFTER_JOB:-}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_zip)
            OUTPUT_ZIP="$2"
            shift 2
            ;;
        --remove_original)
            REMOVE_ORIGINAL="true"
            shift
            ;;
        --include_temp)
            INCLUDE_TEMP="true"
            shift
            ;;
        --after_job)
            AFTER_JOB="$2"
            shift 2
            ;;
        --help)
            cat <<EOF
Usage: $0 [OPTIONS]

Submit HF data compression job to SLURM.

Options:
  --data_dir DIR          Data directory (default: data)
  --output_zip FILE       Output zip file path (default: data_dir/high_fidelity.zip)
  --remove_original       Remove original files after compression (use with caution!)
  --include_temp         Include temp_outputs directory in the archive
  --after_job JOB_ID      Run after specified job completes (dependency)
  --help                  Show this help message

Examples:
  # Compress HF data (standalone)
  $0

  # Compress and remove originals (saves space)
  $0 --remove_original

  # Include material grids
  $0 --include_temp

  # Run after HF generation job completes
  HF_JOB_ID=\$(sbatch --parsable job_generate_hf.sh)
  $0 --after_job \$HF_JOB_ID

  # Custom output location
  $0 --output_zip /path/to/hf_data.zip

Environment variables can also be used:
  DATA_DIR, OUTPUT_ZIP, REMOVE_ORIGINAL, INCLUDE_TEMP, AFTER_JOB
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
echo "HF Data Compression Job Submission"
echo "============================================================================"
echo "Configuration:"
echo "  Data Directory:      $DATA_DIR"
if [ -n "$OUTPUT_ZIP" ]; then
    echo "  Output ZIP:          $OUTPUT_ZIP"
else
    echo "  Output ZIP:          ${DATA_DIR}/high_fidelity.zip (default)"
fi
echo "  Remove Original:     $REMOVE_ORIGINAL"
echo "  Include Temp:        $INCLUDE_TEMP"
if [ -n "$AFTER_JOB" ]; then
    echo "  Dependency:          afterok:$AFTER_JOB"
fi
echo "============================================================================"
echo ""

# Check if compression script exists
COMPRESS_SCRIPT="compress_hf_data.py"
if [ ! -f "$COMPRESS_SCRIPT" ]; then
    echo "Error: Compression script '$COMPRESS_SCRIPT' not found in current directory" >&2
    exit 1
fi

# Check if job script exists
JOB_SCRIPT="job_compress_hf.sh"
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Error: Job script '$JOB_SCRIPT' not found in current directory" >&2
    exit 1
fi

# Create a temporary job script with customized parameters
TEMP_JOB_SCRIPT=$(mktemp)
trap "rm -f $TEMP_JOB_SCRIPT" EXIT

# Read the original job script
cat "$JOB_SCRIPT" > "$TEMP_JOB_SCRIPT"

# Update dependency if specified
if [ -n "$AFTER_JOB" ]; then
    # Replace or add dependency line
    if grep -q "JOB_ID_PLACEHOLDER" "$TEMP_JOB_SCRIPT"; then
        sed -i "s/JOB_ID_PLACEHOLDER/$AFTER_JOB/" "$TEMP_JOB_SCRIPT"
    else
        # Add dependency line if not present
        sed -i "/^#SBATCH --dependency=/d" "$TEMP_JOB_SCRIPT"
        sed -i "/^#SBATCH --error=/a #SBATCH --dependency=afterok:$AFTER_JOB" "$TEMP_JOB_SCRIPT"
    fi
else
    # Remove dependency line if no dependency specified
    sed -i "/^#SBATCH --dependency=/d" "$TEMP_JOB_SCRIPT"
fi

# Update environment variables in the script
sed -i "s|DATA_DIR=\"\${DATA_DIR:-data}\"|DATA_DIR=\"${DATA_DIR}\"|" "$TEMP_JOB_SCRIPT"
if [ -n "$OUTPUT_ZIP" ]; then
    sed -i "s|OUTPUT_ZIP=\"\${OUTPUT_ZIP:-}\"|OUTPUT_ZIP=\"${OUTPUT_ZIP}\"|" "$TEMP_JOB_SCRIPT"
fi
sed -i "s|REMOVE_ORIGINAL=\"\${REMOVE_ORIGINAL:-false}\"|REMOVE_ORIGINAL=\"${REMOVE_ORIGINAL}\"|" "$TEMP_JOB_SCRIPT"
sed -i "s|INCLUDE_TEMP=\"\${INCLUDE_TEMP:-false}\"|INCLUDE_TEMP=\"${INCLUDE_TEMP}\"|" "$TEMP_JOB_SCRIPT"

# Make it executable
chmod +x "$TEMP_JOB_SCRIPT"

# Submit the job
echo "Submitting compression job to SLURM..."
JOB_ID=$(sbatch "$TEMP_JOB_SCRIPT" | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo ""
    echo "✓ Compression job submitted successfully!"
    echo "  Job ID: $JOB_ID"
    if [ -n "$AFTER_JOB" ]; then
        echo "  Will run after job: $AFTER_JOB"
    fi
    echo ""
    echo "Monitor job status with:"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "View job output:"
    echo "  tail -f logs/compress_hf_${JOB_ID}.out"
    echo ""
    echo "Check job completion:"
    echo "  sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed"
else
    echo "Error: Failed to submit job" >&2
    exit 1
fi

