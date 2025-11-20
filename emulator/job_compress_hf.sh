#!/bin/bash
#SBATCH --job-name=compress_hf
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/compress_hf_%j.out
#SBATCH --error=logs/compress_hf_%j.err
#SBATCH --dependency=afterok:JOB_ID_PLACEHOLDER

# SLURM job script for compressing HF data into a zip archive
# This script should be run AFTER all HF generation jobs complete
#
# Usage:
#   # After submitting HF generation jobs, get the job ID and update dependency:
#   HF_JOB_ID=$(sbatch --parsable job_generate_hf.sh)
#   sed "s/JOB_ID_PLACEHOLDER/$HF_JOB_ID/" job_compress_hf.sh | sbatch
#
#   # Or use the helper script:
#   ./submit_compress_hf.sh --after_job <HF_JOB_ID>
#
#   # Or run standalone (no dependency):
#   sbatch job_compress_hf.sh  # (remove or comment out --dependency line)

set -euo pipefail

# Simple early logging
echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Host=$(hostname)" >&2

# Load modules
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "$(date -Is) | MODULE | Purging modules..." >&2
    module purge
    echo "$(date -Is) | MODULE | Loading gcc/13.2.0..." >&2
    module load gcc/13.2.0
    echo "$(date -Is) | MODULE | Module load complete." >&2
fi

# Activate your project venv (absolute path for HPC home)
source /global/home/users/kurtwal98/seiskit/.venv/bin/activate

# Prevent Python bytecode (.pyc) file writing
export PYTHONDONTWRITEBYTECODE=1

# Run from the directory you submitted the job
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Define key paths
PYTHON_BIN="/global/home/users/kurtwal98/seiskit/.venv/bin/python"
COMPRESS_SCRIPT="${SLURM_SUBMIT_DIR:-$PWD}/compress_data.py"

# Add fail-fast checks
command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python binary not found at ${PYTHON_BIN}" >&2; exit 2; }
test -r "${COMPRESS_SCRIPT}" >/dev/null || { echo "ERROR: Compression script not readable at ${COMPRESS_SCRIPT}" >&2; exit 2; }

# Record start time
START_TIME=$(date +%s)
START_DATE=$(date)

echo "============================================================================" >&2
echo "HF Data Compression Job" >&2
echo "Job ID: ${SLURM_JOB_ID:-<local>}" >&2
echo "Start: $START_DATE" >&2
echo "============================================================================" >&2

# Default parameters (adjust as needed)
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_ZIP="${OUTPUT_ZIP:-}"
REMOVE_ORIGINAL="${REMOVE_ORIGINAL:-false}"
INCLUDE_TEMP="${INCLUDE_TEMP:-false}"

# Build command
CMD="${PYTHON_BIN} -u ${COMPRESS_SCRIPT} --data_dir ${DATA_DIR}"

if [ -n "${OUTPUT_ZIP}" ]; then
    CMD="${CMD} --output_zip ${OUTPUT_ZIP}"
fi

if [ "${REMOVE_ORIGINAL}" = "true" ]; then
    CMD="${CMD} --remove_original"
fi

if [ "${INCLUDE_TEMP}" = "true" ]; then
    CMD="${CMD} --include_temp"
fi

# Execute compression
echo "$(date -Is) | RUN | Starting compression..." >&2
echo "Command: $CMD" >&2

eval $CMD
PYTHON_EXIT_CODE=$?

echo "$(date -Is) | RUN | Compression finished. Exit Code: ${PYTHON_EXIT_CODE}" >&2

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "============================================================================" >&2
echo "Compression Job Completed" >&2
echo "Start Time: $START_DATE" >&2
echo "End Time: $(date)" >&2
echo "Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s (${DURATION}s)" >&2
echo "Python Exit Code: $PYTHON_EXIT_CODE" >&2
echo "============================================================================" >&2

# Exit with Python's exit code
exit $PYTHON_EXIT_CODE

