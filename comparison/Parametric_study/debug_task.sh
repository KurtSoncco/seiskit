#!/bin/bash
# Debug a specific array task

if [ $# -eq 0 ]; then
    echo "Usage: $0 <task_index> [job_id]"
    echo ""
    echo "Examples:"
    echo "  $0 0           # Check task 0 from most recent job"
    echo "  $0 15 123456   # Check task 15 from job 123456"
    echo ""
    echo "Task indices range from 0-44:"
    echo "  3 rH values (10, 30, 50) × 3 CV values (0.1, 0.2, 0.3) × 5 seeds = 45 tasks"
    exit 1
fi

TASK_ID=$1
JOB_ID=$2

echo "============================================================================"
echo "Debugging Task $TASK_ID"
echo "============================================================================"
echo ""

# Map task index to parameters
rH_values=(10.0 30.0 50.0)
CV_values=(0.1 0.2 0.3)
seed_values=(10 20 30 40 50)

rH_idx=$((TASK_ID / 15))
remaining=$((TASK_ID % 15))
CV_idx=$((remaining / 5))
seed_idx=$((remaining % 5))

rH=${rH_values[$rH_idx]}
CV=${CV_values[$CV_idx]}
seed=${seed_values[$seed_idx]}

echo "Task parameters:"
echo "  rH = $rH m"
echo "  CV = $CV"
echo "  seed = $seed"
echo "  task_id = rH${rH%.0f}_CV${CV}_s${seed}"
echo ""

# Find job ID if not provided
if [ -z "$JOB_ID" ]; then
    # Look for the most recent output file
    RECENT_JOB=$(ls -t array_job_*_task_${TASK_ID}.out 2>/dev/null | head -1)
    if [ -n "$RECENT_JOB" ]; then
        JOB_ID=$(echo "$RECENT_JOB" | sed -n 's/array_job_\([0-9]*\)_task_'"${TASK_ID}"'\.out/\1/p')
        echo "Using job ID from recent output: $JOB_ID"
    else
        echo "Error: Could not find output file for task $TASK_ID"
        echo "Please provide job ID: $0 $TASK_ID <job_id>"
        exit 1
    fi
fi

echo ""
echo "Job ID: $JOB_ID"
echo ""

OUTPUT_FILE="array_job_${JOB_ID}_task_${TASK_ID}.out"
ERROR_FILE="array_job_${JOB_ID}_task_${TASK_ID}.err"

# Check if files exist
if [ ! -f "$OUTPUT_FILE" ] && [ ! -f "$ERROR_FILE" ]; then
    echo "No log files found for this task."
    echo "  Expected: $OUTPUT_FILE"
    echo "  Expected: $ERROR_FILE"
    echo ""
    echo "Available task files:"
    ls -1 array_job_${JOB_ID}_task_*.out 2>/dev/null | head -10
    exit 1
fi

# Check completion status
echo "Status:"
if [ -f "$OUTPUT_FILE" ] && grep -q "Job Completed - Timing Summary" "$OUTPUT_FILE"; then
    echo "  ✓ Completed successfully"
    grep "Total Duration:" "$OUTPUT_FILE" | head -1
else
    echo "  ✗ Did not complete successfully"
fi
echo ""

# Show errors
if [ -f "$ERROR_FILE" ] && [ -s "$ERROR_FILE" ]; then
    echo "Error output:"
    echo "----------------------------------------"
    cat "$ERROR_FILE"
    echo "----------------------------------------"
    echo ""
fi

# Show final part of output
if [ -f "$OUTPUT_FILE" ]; then
    echo "Last 50 lines of output:"
    echo "----------------------------------------"
    tail -50 "$OUTPUT_FILE"
    echo "----------------------------------------"
fi

# Check result file
RESULT_FILE="results/rH_${rH%.0f}/CV_${CV}/rH${rH%.0f}_CV${CV}_s${seed}/rH${rH%.0f}_CV${CV}_s${seed}/surface_nodes_dof1_accel.txt"
echo ""
echo "Expected result file: $RESULT_FILE"
if [ -f "$RESULT_FILE" ]; then
    echo "  ✓ Result file exists"
    echo "  Size: $(stat -f%z "$RESULT_FILE" 2>/dev/null || stat -c%s "$RESULT_FILE" 2>/dev/null) bytes"
else
    echo "  ✗ Result file not found"
fi

echo ""
echo "============================================================================"
echo "To re-run this task, use:"
echo "  SLURM_ARRAY_TASK_ID=$TASK_ID python run_experiment.py"
echo "============================================================================"

