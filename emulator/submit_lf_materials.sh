#!/bin/bash
# Helper script to submit LF material generation jobs
# Generates material grids and parameters needed for HF generation

set -euo pipefail

# Default values
DATA_DIR="${DATA_DIR:-data}"
N_TEST="${N_TEST:-100}"
START_IDX="${START_IDX:-1000}"
HX="${HX:-10.0}"
HX_HF="${HX_HF:-1.0}"
LX="${LX:-150.0}"
LZ="${LZ:-150.0}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --n_test)
            N_TEST="$2"
            shift 2
            ;;
        --start_idx)
            START_IDX="$2"
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
        --help)
            cat <<EOF
Usage: $0 [OPTIONS]

Submit SLURM array job to generate LF material grids and parameters.

Options:
    --data_dir DIR      Data directory (default: data)
    --n_test N          Number of test samples to generate (default: 100)
    --start_idx IDX      Starting index for simulation IDs (default: 1000)
    --hx SIZE           LF element size in meters (default: 10.0)
    --hx_hf SIZE        HF element size in meters (default: 1.0)
    --Lx WIDTH          Domain width in meters (default: 150.0)
    --Lz HEIGHT         Domain height in meters (default: 150.0)
    --help              Show this help message

Examples:
    # Generate materials for 100 test samples (default)
    $0

    # Generate materials for 200 test samples
    $0 --n_test 200

    # Generate materials starting from index 2000
    $0 --start_idx 2000 --n_test 100
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
echo "Submitting LF Material Generation Job"
echo "============================================================================"
echo "Data directory: ${DATA_DIR}"
echo "Number of samples: ${N_TEST}"
echo "Starting index: ${START_IDX}"
echo "Array range: 0-${ARRAY_MAX}"
echo "Parameters:"
echo "  LF element size (hx): ${HX} m"
echo "  HF element size (hx_hf): ${HX_HF} m"
echo "  Domain size: ${LX} x ${LZ} m"
echo "============================================================================"

# Export variables for the job script
export START_IDX
export DATA_DIR
export HX
export HX_HF
export LX
export LZ

# Update the job script array range
JOB_SCRIPT="job_generate_lf_materials.sh"
TEMP_SCRIPT=$(mktemp)
sed "s/^#SBATCH --array=.*/#SBATCH --array=0-${ARRAY_MAX}/" "${JOB_SCRIPT}" > "${TEMP_SCRIPT}"

# Submit the job
JOB_ID=$(sbatch --parsable "${TEMP_SCRIPT}")
rm "${TEMP_SCRIPT}"

echo ""
echo "✅ Job submitted successfully!"
echo "Job ID: ${JOB_ID}"
echo ""
echo "Monitor job status:"
echo "  squeue -j ${JOB_ID}"
echo ""
echo "View output:"
echo "  tail -f logs/lf_materials_${JOB_ID}_task_0.out"
echo ""
echo "Check progress:"
echo "  ls ${DATA_DIR}/materials/sim_*.npy | wc -l"
echo "  ls ${DATA_DIR}/material_params/sim_*.json | wc -l"
echo ""
echo "After completion, you can submit HF generation jobs:"
echo "  ./submit_hf_jobs.sh --test_start_idx ${START_IDX} --n_test ${N_TEST}"

