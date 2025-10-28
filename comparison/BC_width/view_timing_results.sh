#!/bin/bash
# Helper script to view timing results from BC_width SLURM array jobs

echo "============================================================================"
echo "BC Width Experiment - Timing Results Summary"
echo "============================================================================"
echo ""

# Find array job output files
ARRAY_FILES=(array_job_*_task_*.out)
# Find sequential job output files
SEQUENTIAL_FILES=(full_workflow_*.out)

# Determine which type of job was run
if [ ${#ARRAY_FILES[@]} -gt 0 ] && [ ${#SEQUENTIAL_FILES[@]} -gt 0 ]; then
    echo "Found both ARRAY job and SEQUENTIAL job outputs!"
    echo "Comparing parallel vs sequential execution..."
    echo ""
    COMPARE_MODE=true
elif [ ${#ARRAY_FILES[@]} -gt 0 ]; then
    echo "Running in ARRAY job mode (parallel execution)"
    echo ""
    COMPARE_MODE=false
    OUTPUT_FILES=("${ARRAY_FILES[@]}")
elif [ ${#SEQUENTIAL_FILES[@]} -gt 0 ]; then
    echo "Running in SEQUENTIAL job mode"
    echo ""
    COMPARE_MODE=false
    # For sequential mode, we need to parse the single output file
    if [ ${#SEQUENTIAL_FILES[@]} -eq 0 ]; then
        echo "No output files found!"
        exit 1
    fi
    OUTPUT_FILES=("${SEQUENTIAL_FILES[@]}")
else
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

# If in comparison mode, show parallel vs sequential comparison
if [ "$COMPARE_MODE" = true ]; then
    echo ""
    echo "============================================================================"
    echo "Parallel vs Sequential Execution Comparison"
    echo "============================================================================"
    
    # Parse array job durations
    ARRAY_TOTAL=0
    ARRAY_COUNT=0
    ARRAY_DURATIONS=()
    
    for file in "${ARRAY_FILES[@]}"; do
        if grep -q "Total Duration (seconds):" "$file" 2>/dev/null; then
            DURATION=$(grep "Total Duration (seconds):" "$file" | awk '{print $4}')
            if [ ! -z "$DURATION" ]; then
                ARRAY_DURATIONS+=("$DURATION")
                ARRAY_TOTAL=$((ARRAY_TOTAL + DURATION))
                ARRAY_COUNT=$((ARRAY_COUNT + 1))
            fi
        fi
    done
    
    # Parse sequential job total time
    SEQUENTIAL_FILE="${SEQUENTIAL_FILES[0]}"
    SEQUENTIAL_DURATION=0
    
    if [ -f "$SEQUENTIAL_FILE" ]; then
        # Look for "program] Total wall time" in the output
        TOTAL_TIME=$(grep "program\] Total wall time:" "$SEQUENTIAL_FILE" | tail -n 1)
        if [ ! -z "$TOTAL_TIME" ]; then
            # Extract time in HH:MM:SS format and convert to seconds
            TIME_STR=$(echo "$TOTAL_TIME" | awk '{print $NF}')
            # Parse HH:MM:SS
            HOURS=$(echo "$TIME_STR" | cut -d':' -f1)
            MINUTES=$(echo "$TIME_STR" | cut -d':' -f2)
            SECONDS=$(echo "$TIME_STR" | cut -d':' -f3)
            SEQUENTIAL_DURATION=$((HOURS * 3600 + MINUTES * 60 + SECONDS))
        fi
    fi
    
    if [ $ARRAY_COUNT -gt 0 ] && [ $SEQUENTIAL_DURATION -gt 0 ]; then
        ARRAY_MAX=0
        for dur in "${ARRAY_DURATIONS[@]}"; do
            if [ $dur -gt $ARRAY_MAX ]; then
                ARRAY_MAX=$dur
            fi
        done
        
        # Array job time is the longest task (since tasks run in parallel)
        PARALLEL_TIME=$ARRAY_MAX
        
        echo "Parallel execution (array job):"
        echo "  Tasks completed: $ARRAY_COUNT / 14"
        echo "  Longest task: ${PARALLEL_TIME}s ($(($PARALLEL_TIME / 60))m $(($PARALLEL_TIME % 60))s)"
        echo "  All tasks run simultaneously on different nodes"
        echo ""
        
        echo "Sequential execution (submit_jobs.sh):"
        echo "  Total time: ${SEQUENTIAL_DURATION}s ($(($SEQUENTIAL_DURATION / 60))m $(($SEQUENTIAL_DURATION % 60))s)"
        echo "  Tasks run one after another on same node"
        echo ""
        
        if [ $SEQUENTIAL_DURATION -gt 0 ]; then
            SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $SEQUENTIAL_DURATION / $PARALLEL_TIME}")
            EFFICIENCY=$(awk "BEGIN {printf \"%.1f\", ($SPEEDUP / 14) * 100}")
            
            echo "Performance Analysis:"
            echo "  Speedup: ${SPEEDUP}x"
            echo "  Efficiency: ${EFFICIENCY}% (theoretical max: ${SPEEDUP}%)"
            echo "  Time saved: $((SEQUENTIAL_DURATION - PARALLEL_TIME))s ($(( (SEQUENTIAL_DURATION - PARALLEL_TIME) / 60 ))m $(( (SEQUENTIAL_DURATION - PARALLEL_TIME) % 60 ))s)"
            
            echo ""
            echo "📊 Summary:"
            if [ $(echo "$SPEEDUP > 8" | bc) -eq 1 ]; then
                echo "  ✅ Excellent parallelization! Speedup > 8x"
            elif [ $(echo "$SPEEDUP > 4" | bc) -eq 1 ]; then
                echo "  ✓ Good parallelization with ${SPEEDUP}x speedup"
            else
                echo "  ⚠️  Limited speedup (${SPEEDUP}x). Consider optimizing."
            fi
        fi
    else
        echo "Need both parallel and sequential results for comparison."
    fi
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

