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

# Get array job details (all tasks)
echo "Array Job Details (all tasks):"
# For array jobs, sacct shows:
# - JOBID - main job (summary)
# - JOBID_TASKID - array task allocation
# - JOBID_TASKID.batch - actual execution step
# We want to show the .batch steps (actual execution) or the array task IDs
# Get all lines and filter for array tasks
ALL_DATA=$(sacct -j "$JOB_ID" --format=JobID,JobName,State,Start,End,Elapsed,MaxRSS,TotalCPU,ReqCPUS,ReqMem,NodeList --parsable2 --noheader)

# Filter for array tasks: lines starting with JOBID_TASKID (with or without .batch)
TASK_DATA=$(echo "$ALL_DATA" | grep -E "^${JOB_ID}_[0-9]+" | grep -v "^$")

if [ -z "$TASK_DATA" ]; then
    # If no array tasks found, show all job steps (might be a single job)
    echo "  No array tasks found. Showing all job steps:"
    TASK_DATA=$(echo "$ALL_DATA" | grep -v "^$")
fi

if [ -n "$TASK_DATA" ]; then
    # Print header
    HEADER="JobID|JobName|State|Start|End|Elapsed|MaxRSS|TotalCPU|ReqCPUS|ReqMem|NodeList"
    echo "$HEADER"
    echo "$HEADER" | sed 's/[^|]/-/g'  # Separator line
    # Print all task data (limit to first 100 tasks to avoid overwhelming output)
    echo "$TASK_DATA" | head -100
    TASK_COUNT=$(echo "$TASK_DATA" | wc -l)
    echo ""
    if [ "$TASK_COUNT" -gt 100 ]; then
        echo "  (Showing first 100 of $TASK_COUNT tasks. Use 'sacct -j $JOB_ID' for full list)"
    else
        echo "Total tasks: $TASK_COUNT"
    fi
else
    echo "  No task data found for job $JOB_ID"
    echo "  Raw sacct output:"
    echo "$ALL_DATA" | head -5
fi
echo ""

# Calculate statistics
echo "Timing Statistics:"
if [ -n "$TASK_DATA" ]; then
    # Get all elapsed times and find max/min (since tasks run in parallel)
    MAX_ELAPSED=$(echo "$TASK_DATA" | cut -d'|' -f6 | grep -v "^$" | sort -r | head -1)
    MIN_ELAPSED=$(echo "$TASK_DATA" | cut -d'|' -f6 | grep -v "^$" | sort | head -1)
    
    echo "  Longest task duration: $MAX_ELAPSED"
    if [ "$MIN_ELAPSED" != "$MAX_ELAPSED" ]; then
        echo "  Shortest task duration: $MIN_ELAPSED"
    fi
    
    # Calculate total job time: earliest start to latest end
    # Extract Start (column 4) and End (column 5) times
    # Format: 2025-12-15T19:08:12
    EARLIEST_START=$(echo "$TASK_DATA" | cut -d'|' -f4 | grep -v "^$" | sort | head -1)
    LATEST_END=$(echo "$TASK_DATA" | cut -d'|' -f5 | grep -v "^$" | sort -r | head -1)
    
    if [ -n "$EARLIEST_START" ] && [ -n "$LATEST_END" ]; then
        echo ""
        echo "  Total Job Duration (earliest start to latest end):"
        echo "    Start: $EARLIEST_START"
        echo "    End:   $LATEST_END"
        
        # Calculate difference in seconds
        # Convert ISO format to epoch seconds (handle both GNU date and BSD date)
        if command -v date >/dev/null 2>&1; then
            # Try GNU date format first (Linux)
            START_EPOCH=$(date -d "$EARLIEST_START" +%s 2>/dev/null)
            END_EPOCH=$(date -d "$LATEST_END" +%s 2>/dev/null)
            
            # If that failed, try BSD date format (macOS)
            if [ -z "$START_EPOCH" ] || [ -z "$END_EPOCH" ]; then
                START_EPOCH=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$EARLIEST_START" +%s 2>/dev/null)
                END_EPOCH=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$LATEST_END" +%s 2>/dev/null)
            fi
        fi
        
        if [ -n "$START_EPOCH" ] && [ -n "$END_EPOCH" ] && [ "$START_EPOCH" -lt "$END_EPOCH" ]; then
            DIFF_SECONDS=$((END_EPOCH - START_EPOCH))
            TOTAL_DURATION=$(format_time $DIFF_SECONDS)
            # Calculate hours (use awk if bc is not available)
            if command -v bc >/dev/null 2>&1; then
                TOTAL_HOURS=$(echo "scale=2; $DIFF_SECONDS / 3600" | bc)
            else
                TOTAL_HOURS=$(awk "BEGIN {printf \"%.2f\", $DIFF_SECONDS / 3600}")
            fi
            echo "    Duration: $TOTAL_DURATION ($TOTAL_HOURS hours)"
        else
            # Fallback: manual calculation from time strings
            echo "    (Calculating duration manually...)"
            # Extract time components and calculate difference
            # This is a simple fallback - for exact calculation, use the epoch method above
        fi
    fi
    
    echo ""
    echo "  Note: Individual tasks run in parallel, so total job time is less than"
    echo "  the sum of all task times. Use 'python merge_timing_data.py' to see"
    echo "  the sum of all individual task execution times."
else
    echo "  Could not extract timing information"
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
