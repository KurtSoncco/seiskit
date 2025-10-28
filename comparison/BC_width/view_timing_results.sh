#!/bin/bash
# Helper script to view timing results from BC_width SLURM array jobs

echo "============================================================================"
echo "BC Width Experiment - Timing Results Summary"
echo "============================================================================"
echo ""

# Find all output files
OUTPUT_FILES=(array_job_*_task_*.out)

if [ ${#OUTPUT_FILES[@]} -eq 0 ]; then
    echo "No output files found!"
    echo "Make sure you're in the directory where the jobs were submitted."
    exit 1
fi

# Function to get task parameters from index
get_task_params() {
    local index=$1
    local BC_WIDTHS=(0 100 200 300 400 500 1000)
    
    if [ $index -lt 7 ]; then
        LX_VAR=800
        BC_IDX=$index
    else
        LX_VAR=100
        BC_IDX=$((index - 7))
    fi
    
    BC_WIDTH=${BC_WIDTHS[$BC_IDX]}
    TOTAL_WIDTH=$((LX_VAR + 2 * BC_WIDTH))
    
    echo "$LX_VAR $BC_WIDTH $TOTAL_WIDTH"
}

# Array to store durations and task info
declare -a DURATIONS=()
declare -a TASK_INFO=()

echo "Task Breakdown:"
echo "Index | Lx_var | BC_width | Total Width | Status"
echo "------|--------|----------|-------------|--------"

for file in "${OUTPUT_FILES[@]}"; do
    # Extract task ID from filename
    TASK_ID=$(echo $file | sed -n 's/array_job_[0-9]*_task_\([0-9]*\)\.out/\1/p')
    
    if [ -z "$TASK_ID" ]; then
        continue
    fi
    
    # Get task parameters
    read LX_VAR BC_WIDTH TOTAL_WIDTH <<< $(get_task_params $TASK_ID)
    
    # Check if file exists and is readable
    if [ -f "$file" ] && [ -r "$file" ]; then
        # Show status
        if grep -q "Job Completed - Timing Summary" "$file"; then
            STATUS="✓ COMPLETED"
            
            # Extract duration
            DURATION=$(grep "Total Duration (seconds):" "$file" | awk '{print $4}')
            if [ ! -z "$DURATION" ]; then
                DURATIONS+=("$DURATION")
                TASK_INFO+=("$TASK_ID|$LX_VAR|$BC_WIDTH|$TOTAL_WIDTH|$DURATION")
            fi
        else
            STATUS="⏳ RUNNING"
        fi
    else
        STATUS="❌ NOT FOUND"
    fi
    
    printf "%5d | %6d | %8d | %11d | %s\n" "$TASK_ID" "$LX_VAR" "$BC_WIDTH" "$TOTAL_WIDTH" "$STATUS"
done

echo ""
echo "============================================================================"
echo "Detailed Status (Completed Jobs)"
echo "============================================================================"

for file in "${OUTPUT_FILES[@]}"; do
    TASK_ID=$(echo $file | sed -n 's/array_job_[0-9]*_task_\([0-9]*\)\.out/\1/p')
    
    if [ -z "$TASK_ID" ]; then
        continue
    fi
    
    if grep -q "Job Completed - Timing Summary" "$file" 2>/dev/null; then
        read LX_VAR BC_WIDTH TOTAL_WIDTH <<< $(get_task_params $TASK_ID)
        
        echo ""
        echo "----------------------------------------"
        echo "Task $TASK_ID: Lx_var=${LX_VAR}m, BC_width=${BC_WIDTH}m (Total: ${TOTAL_WIDTH}m)"
        echo "----------------------------------------"
        
        # Extract and show timing info
        awk '/Job Completed - Timing Summary/,/^==/{print}' "$file" 2>/dev/null | head -n 8
    fi
done

echo ""
echo "============================================================================"
echo "Overall Statistics"
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
    
    echo "Completed jobs: $COUNT / 14"
    echo "Min duration: ${MIN}s ($(($MIN / 60))m $(($MIN % 60))s)"
    echo "Max duration: ${MAX}s ($(($MAX / 60))m $(($MAX % 60))s)"
    echo "Avg duration: ${AVG}s ($(($AVG / 60))m $(($AVG % 60))s)"
    echo "Total duration: ${TOTAL}s ($(($TOTAL / 60))m $(($TOTAL % 60))s)"
    
    if [ $COUNT -lt 14 ]; then
        echo ""
        echo "⚠️  Warning: Only $COUNT of 14 tasks completed"
        echo "Missing tasks:"
        for i in {0..13}; do
            found=false
            for task_info in "${TASK_INFO[@]}"; do
                task_id=$(echo $task_info | cut -d'|' -f1)
                if [ "$task_id" == "$i" ]; then
                    found=true
                    break
                fi
            done
            if [ "$found" = false ]; then
                read LX_VAR BC_WIDTH TOTAL_WIDTH <<< $(get_task_params $i)
                echo "  - Index $i: Lx_var=${LX_VAR}m, BC_width=${BC_WIDTH}m"
            fi
        done
    else
        echo ""
        echo "✅ All 14 tasks completed successfully!"
    fi
else
    echo "No completed jobs yet."
fi

echo ""
echo "============================================================================"
echo "Quick View Commands"
echo "============================================================================"
echo "View specific output file:      cat array_job_ID_task_N.out"
echo "View specific error file:      cat array_job_ID_task_N.err"
echo "Follow job progress:            tail -f array_job_ID_task_N.out"
echo "Check job status:               squeue -u \$USER"
echo "Get detailed job statistics:     sacct -j JOB_ID --format=JobID,TaskID,Elapsed,State,ExitCode"
echo ""
echo "View results directory:         ls -lh results/"
echo "Check specific result:          ls -lh results/Lx_800/BC_width_0/"
echo "Check all BC widths for Lx=800: ls -lh results/Lx_800/BC_width_*/"
echo "============================================================================"

