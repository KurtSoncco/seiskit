#!/bin/bash
# Helper script to view timing results from Parametric Study SLURM array jobs

echo "============================================================================"
echo "Parametric Study Experiment - Timing Results Summary"
echo "============================================================================"
echo ""

# Find array job output files
ARRAY_FILES=(array_job_*_task_*.out)

# Check if output files exist
if [ ${#ARRAY_FILES[@]} -eq 0 ]; then
    echo "No output files found!"
    echo "Make sure you're in the directory where the jobs were submitted."
    exit 1
fi

echo "Running in ARRAY job mode (parallel execution)"
echo ""
OUTPUT_FILES=("${ARRAY_FILES[@]}")

# Function to get task parameters from index
get_task_params() {
    local index=$1
    local rH_values=(10 30 50)
    local CV_values=(0.1 0.2 0.3)
    local seed_values=(10 20 30 40 50)
    
    # Map index to parameter combination
    # index = rH_idx × (3×5) + CV_idx × 5 + seed_idx
    local rH_idx=$((index / 15))  # 15 = 3 × 5
    local remaining=$((index % 15))
    local CV_idx=$((remaining / 5))
    local seed_idx=$((remaining % 5))
    
    local rH=${rH_values[$rH_idx]}
    local CV=${CV_values[$CV_idx]}
    local seed=${seed_values[$seed_idx]}
    
    echo "$rH $CV $seed"
}

# Array to store durations and task info
declare -a DURATIONS=()
declare -a TASK_INFO=()

echo "Task Breakdown:"
echo "Index |   rH   |  CV  | Seed | Status"
echo "------|--------|------|------|--------"

for file in "${OUTPUT_FILES[@]}"; do
    # Extract task ID from filename
    TASK_ID=$(echo $file | sed -n 's/array_job_[0-9]*_task_\([0-9]*\)\.out/\1/p')
    
    if [ -z "$TASK_ID" ]; then
        continue
    fi
    
    # Get task parameters
    read rH CV SEED <<< $(get_task_params $TASK_ID)
    
    # Check if file exists and is readable
    if [ -f "$file" ] && [ -r "$file" ]; then
        # Show status
        if grep -q "Job Completed - Timing Summary" "$file"; then
            STATUS="✓ COMPLETED"
            
            # Extract duration
            DURATION=$(grep "Total Duration (seconds):" "$file" | awk '{print $4}')
            if [ ! -z "$DURATION" ]; then
                DURATIONS+=("$DURATION")
                TASK_INFO+=("$TASK_ID|$rH|$CV|$SEED|$DURATION")
            fi
        else
            STATUS="⏳ RUNNING"
        fi
    else
        STATUS="❌ NOT FOUND"
    fi
    
    printf "%5d | %6.0f | %4.1f | %4d | %s\n" "$TASK_ID" "$rH" "$CV" "$SEED" "$STATUS"
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
        read rH CV SEED <<< $(get_task_params $TASK_ID)
        
        echo ""
        echo "----------------------------------------"
        echo "Task $TASK_ID: rH=${rH}m, CV=${CV}, seed=${SEED}"
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
    
    echo "Completed jobs: $COUNT / 45"
    echo "Min duration: ${MIN}s ($(($MIN / 60))m $(($MIN % 60))s)"
    echo "Max duration: ${MAX}s ($(($MAX / 60))m $(($MAX % 60))s)"
    echo "Avg duration: ${AVG}s ($(($AVG / 60))m $(($AVG % 60))s)"
    echo "Total duration: ${TOTAL}s ($(($TOTAL / 3600))h $(($(($TOTAL % 3600)) / 60))m $(($TOTAL % 60))s)"
    
    if [ $COUNT -lt 45 ]; then
        echo ""
        echo "⚠️  Warning: Only $COUNT of 45 tasks completed"
        echo "Missing tasks:"
        for i in {0..44}; do
            found=false
            for task_info in "${TASK_INFO[@]}"; do
                task_id=$(echo $task_info | cut -d'|' -f1)
                if [ "$task_id" == "$i" ]; then
                    found=true
                    break
                fi
            done
            if [ "$found" = false ]; then
                read rH CV SEED <<< $(get_task_params $i)
                echo "  - Index $i: rH=${rH}m, CV=${CV}, seed=${SEED}"
            fi
        done
    else
        echo ""
        echo "✅ All 45 tasks completed successfully!"
    fi
    
    # Show statistics by parameter
    echo ""
    echo "============================================================================"
    echo "Statistics by Parameter"
    echo "============================================================================"
    
    # Group by rH
    echo ""
    echo "By rH (correlation length):"
    for rH in 10 30 50; do
        count=0
        total=0
        min_val=999999
        max_val=0
        
        for task_info in "${TASK_INFO[@]}"; do
            params=$(echo $task_info | cut -d'|' -f2-4)
            task_rH=$(echo $params | cut -d'|' -f1)
            duration=$(echo $task_info | cut -d'|' -f5)
            
            if [ "$task_rH" == "$rH" ]; then
                count=$((count + 1))
                total=$((total + duration))
                if [ $duration -lt $min_val ]; then
                    min_val=$duration
                fi
                if [ $duration -gt $max_val ]; then
                    max_val=$duration
                fi
            fi
        done
        
        if [ $count -gt 0 ]; then
            avg=$((total / count))
            echo "  rH=${rH}m: $count tasks, avg=${avg}s, min=${min_val}s, max=${max_val}s"
        fi
    done
    
    # Group by CV
    echo ""
    echo "By CV (coefficient of variation):"
    for CV in 0.1 0.2 0.3; do
        count=0
        total=0
        min_val=999999
        max_val=0
        
        for task_info in "${TASK_INFO[@]}"; do
            params=$(echo $task_info | cut -d'|' -f2-4)
            task_CV=$(echo $params | cut -d'|' -f2)
            duration=$(echo $task_info | cut -d'|' -f5)
            
            if [ "$task_CV" == "$CV" ]; then
                count=$((count + 1))
                total=$((total + duration))
                if [ $duration -lt $min_val ]; then
                    min_val=$duration
                fi
                if [ $duration -gt $max_val ]; then
                    max_val=$duration
                fi
            fi
        done
        
        if [ $count -gt 0 ]; then
            avg=$((total / count))
            echo "  CV=${CV}: $count tasks, avg=${avg}s, min=${min_val}s, max=${max_val}s"
        fi
    done
else
    echo "No completed jobs yet."
fi

echo ""
echo "============================================================================"
echo "Quick View Commands"
echo "============================================================================"
echo "View specific output file:      cat array_job_ID_task_N.out"
echo "View specific error file:       cat array_job_ID_task_N.err"
echo "Follow job progress:            tail -f array_job_ID_task_N.out"
echo "Check job status:               squeue -u \$USER"
echo "Get detailed job statistics:    sacct -j JOB_ID --format=JobID,TaskID,Elapsed,State,ExitCode"
echo ""
echo "View results directory:         ls -lh results/"
echo "Check specific result:          ls -lh results/rH_10/CV_0.1/"
echo "Check all CV for rH=10:         ls -lh results/rH_10/CV_*/"
echo "Check all rH for CV=0.2:        ls -lh results/rH_*/CV_0.2/"
echo "============================================================================"

