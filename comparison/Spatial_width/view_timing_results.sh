#!/bin/bash
# Helper script to view timing results from SLURM array jobs

echo "============================================================================"
echo "Timing Results Summary"
echo "============================================================================"
echo ""

# Find all output files
OUTPUT_FILES=(array_job_*_task_*.out)

if [ ${#OUTPUT_FILES[@]} -eq 0 ]; then
    echo "No output files found!"
    echo "Make sure you're in the directory where the jobs were submitted."
    exit 1
fi

# Array to store durations
declare -a DURATIONS=()

for file in "${OUTPUT_FILES[@]}"; do
    echo "----------------------------------------"
    echo "File: $file"
    
    # Extract task ID from filename
    TASK_ID=$(echo $file | sed -n 's/array_job_[0-9]*_task_\([0-9]*\)\.out/\1/p')
    
    # Check if file exists and is readable
    if [ -f "$file" ] && [ -r "$file" ]; then
        # Show last 20 lines for context
        if grep -q "Job Completed - Timing Summary" "$file"; then
            echo "Status: COMPLETED"
            # Extract and show timing info
            awk '/Job Completed - Timing Summary/,/^==/{print}' "$file"
            
            # Extract duration
            DURATION=$(grep "Total Duration (seconds):" "$file" | awk '{print $4}')
            if [ ! -z "$DURATION" ]; then
                DURATIONS+=("$DURATION")
            fi
        else
            echo "Status: RUNNING or INCOMPLETE"
            echo "Last few lines:"
            tail -n 5 "$file" 2>/dev/null || echo "  (file may be empty or still being written)"
        fi
    else
        echo "Status: FILE NOT FOUND or NOT READABLE"
    fi
done

echo ""
echo "============================================================================"
echo "Overall Statistics (for completed jobs)"
echo "============================================================================"

if [ ${#DURATIONS[@]} -gt 0 ]; then
    # Calculate statistics
    TOTAL=0
    MIN=${DURATIONS[0]}
    MAX=${DURATIONS[0]}
    
    for duration in "${DURATIONS[@]}"; do
        TOTAL=$((TOTAL + duration))
        if [ $duration -lt $MIN ]; then
            MIN=$duration
        fi
        if [ $duration -gt $MAX ]; then
            MAX=$duration
        fi
    done
    
    COUNT=${#DURATIONS[@]}
    AVG=$((TOTAL / COUNT))
    
    echo "Completed jobs: $COUNT"
    echo "Min duration: ${MIN}s ($(($MIN / 60))m $(($MIN % 60))s)"
    echo "Max duration: ${MAX}s ($(($MAX / 60))m $(($MAX % 60))s)"
    echo "Avg duration: ${AVG}s ($(($AVG / 60))m $(($AVG % 60))s)"
    echo "Total duration: ${TOTAL}s ($(($TOTAL / 60))m $(($TOTAL % 60))s)"
else
    echo "No completed jobs yet."
fi

echo ""
echo "============================================================================"
echo "Quick View Commands:"
echo "============================================================================"
echo "View a specific output file:  cat array_job_ID_task_TASK.out"
echo "View a specific error file:    cat array_job_ID_task_TASK.err"
echo "Follow job progress:          tail -f array_job_ID_task_TASK.out"
echo "Check job status:             squeue -u \$USER"
echo "Get job statistics:          sacct -j JOB_ID --format=JobID,Elapsed,State"
echo "============================================================================"

