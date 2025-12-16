#!/bin/bash
# Script to check SLURM job timing information
#
# Usage:
#   ./check_job_time.sh [JOB_ID]
#   If JOB_ID is not provided, it will prompt for it or try to find recent jobs

set -euo pipefail

# Function to format seconds as HH:MM:SS
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$secs"
}

# Get job ID from argument or prompt
JOB_ID="${1:-}"

if [ -z "$JOB_ID" ]; then
    echo "No job ID provided. Checking recent jobs..."
    echo ""
    echo "Recent jobs:"
    sacct --format=JobID,JobName,State,Start,End,Elapsed,MaxRSS,TotalCPU -S $(date -d '7 days ago' +%Y-%m-%d) | head -20
    echo ""
    read -p "Enter job ID (or press Enter to exit): " JOB_ID
    if [ -z "$JOB_ID" ]; then
        echo "Exiting."
        exit 0
    fi
fi

echo "============================================================================"
echo "SLURM Job Timing Information for Job ID: $JOB_ID"
echo "============================================================================"
echo ""

# Get detailed job information
echo "Job Summary:"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Start,End,Elapsed,MaxRSS,TotalCPU,ReqCPUS,ReqMem,NodeList --parsable2 | head -2 | column -t -s'|'
echo ""

# Get array job details
echo "Array Job Details (all tasks):"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Start,End,Elapsed,MaxRSS,TotalCPU,NodeList --parsable2 | column -t -s'|'
echo ""

# Calculate statistics
echo "Timing Statistics:"
ELAPSED_TIMES=$(sacct -j "$JOB_ID" --format=Elapsed --noheader --parsable2 | grep -v "^$" | head -1)

if [ -n "$ELAPSED_TIMES" ]; then
    # Parse elapsed time (format: HH:MM:SS or D-HH:MM:SS)
    # Convert to seconds for calculation
    TOTAL_SECONDS=0
    
    # Get all elapsed times and calculate max (since tasks run in parallel)
    MAX_ELAPSED=$(sacct -j "$JOB_ID" --format=Elapsed --noheader --parsable2 | grep -v "^$" | head -1)
    
    echo "  Job Duration: $MAX_ELAPSED"
    echo ""
    echo "  Note: For array jobs, this is the duration of the longest-running task."
    echo "  Since tasks run in parallel, total job time is less than sum of individual task times."
fi

# Check if timing CSV files exist
if [ -d "results" ]; then
    TIMING_FILES=$(find results -name "timing_data_task_*.csv" 2>/dev/null | wc -l)
    if [ "$TIMING_FILES" -gt 0 ]; then
        echo ""
        echo "============================================================================"
        echo "Found $TIMING_FILES timing CSV files in results/"
        echo "Run 'python merge_timing_data.py' to merge them and get total statistics."
        echo "============================================================================"
    fi
fi

echo ""
echo "For more details, use:"
echo "  sacct -j $JOB_ID -l          # Detailed job information"
echo "  sacct -j $JOB_ID --format=ALL # All available fields"
echo "  seff $JOB_ID                  # Job efficiency report"
