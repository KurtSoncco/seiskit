#!/bin/bash
# Diagnostic script to check array job status and identify issues

echo "============================================================================"
echo "Array Job Diagnostics"
echo "============================================================================"
echo ""

# Check if running on cluster or local
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Running on SLURM cluster"
else
    echo "Running locally"
fi

# Check for log files
echo ""
echo "1. Looking for log files..."
OUTPUT_FILES=$(ls -1 array_job_*.out 2>/dev/null | head -20)
ERROR_FILES=$(ls -1 array_job_*.err 2>/dev/null | head -20)

if [ -z "$OUTPUT_FILES" ]; then
    echo "   No output files found (array_job_*.out)"
else
    echo "   Found output files:"
    echo "$OUTPUT_FILES" | sed 's/^/     /'
fi

if [ -z "$ERROR_FILES" ]; then
    echo "   No error files found (array_job_*.err)"
else
    echo "   Found error files:"
    echo "$ERROR_FILES" | sed 's/^/     /'
fi

# Check output files for success
echo ""
echo "2. Checking completed tasks..."
if [ -n "$OUTPUT_FILES" ]; then
    echo "   Successfully completed tasks:"
    for file in $OUTPUT_FILES; do
        if grep -q "Job Completed - Timing Summary" "$file" 2>/dev/null; then
            echo "     ✓ $file"
        fi
    done
fi

# Check for errors in log files
echo ""
echo "3. Checking for errors..."
if [ -n "$OUTPUT_FILES" ]; then
    for file in $OUTPUT_FILES; do
        if grep -q "error\|Error\|ERROR\|failed\|Failed\|FAILED" "$file" 2>/dev/null; then
            echo "   Issues in $file:"
            grep -i "error\|failed" "$file" | head -5 | sed 's/^/     /'
        fi
    done
fi

# Check result directories
echo ""
echo "4. Checking result directories..."
RESULT_DIRS=$(find results -type d -name "rH_*" 2>/dev/null | sort)
if [ -z "$RESULT_DIRS" ]; then
    echo "   No result directories found"
else
    echo "   Found result directories:"
    echo "$RESULT_DIRS" | sed 's/^/     /'
    
    # Count completed results
    echo ""
    echo "   Completed results (with surface_nodes_dof1_accel.txt):"
    for dir in $RESULT_DIRS; do
        if [ -d "$dir" ]; then
            # Look for subdirectories like CV_0.1, CV_0.2, etc.
            for cv_dir in "$dir"/*/; do
                if [ -d "$cv_dir" ]; then
                    # Look for task directories like rH10_CV0.1_s10
                    for task_dir in "$cv_dir"*/; do
                        task_name=$(basename "$task_dir")
                        # Check if this is the nested task directory (e.g., rH10_CV0.1_s10/rH10_CV0.1_s10/)
                        if [ -d "$task_dir" ]; then
                            # Check for the file directly in this directory
                            if [ -f "${task_dir}surface_nodes_dof1_accel.txt" ]; then
                                echo "     ✓ $task_name"
                            fi
                        fi
                    done
                fi
            done
        fi
    done
fi

# Expected vs actual
echo ""
echo "5. Task Status Summary..."
echo "   Expected: 45 tasks (indices 0-44)"
echo "   Breakdown: 3 rH values × 3 CV values × 5 seeds = 45"

# Count actual results
COMPLETED_COUNT=0
if [ -n "$RESULT_DIRS" ]; then
    for dir in $RESULT_DIRS; do
        if [ -d "$dir" ]; then
            for cv_dir in "$dir"/*/; do
                if [ -d "$cv_dir" ]; then
                    for task_dir in "$cv_dir"*/; do
                        if [ -d "$task_dir" ] && [ -f "${task_dir}surface_nodes_dof1_accel.txt" ]; then
                            COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
                        fi
                    done
                fi
            done
        fi
    done
fi

echo "   Completed: $COMPLETED_COUNT / 45"
echo "   Missing: $((45 - COMPLETED_COUNT))"

# Check which tasks are missing
echo ""
echo "6. Missing tasks:"
rH_values=(10.0 30.0 50.0)
CV_values=(0.1 0.2 0.3)
seed_values=(10 20 30 40 50)
task_idx=0

for rH in "${rH_values[@]}"; do
    for CV in "${CV_values[@]}"; do
        for seed in "${seed_values[@]}"; do
            task_id="rH${rH%.0f}_CV${CV}_s${seed}"
            result_file="results/rH_${rH%.0f}/CV_${CV}/${task_id}/${task_id}/surface_nodes_dof1_accel.txt"
            
            if [ ! -f "$result_file" ]; then
                echo "   Task $task_idx: $task_id (rH=$rH, CV=$CV, seed=$seed)"
            fi
            task_idx=$((task_idx + 1))
        done
    done
done

# Check recent log files for runtime information
echo ""
echo "7. Recent log file analysis (last 5 jobs)..."
RECENT_OUTPUTS=$(ls -t array_job_*.out 2>/dev/null | head -5)
for file in $RECENT_OUTPUTS; do
    if [ -f "$file" ]; then
        echo "   --- $file ---"
        task_id=$(grep "Array Task ID" "$file" | head -1 | awk '{print $NF}')
        start_time=$(grep "Start Time:" "$file" | head -1 | cut -d: -f2-)
        
        if grep -q "Job Completed" "$file"; then
            duration=$(grep "Total Duration:" "$file" | head -1)
            echo "     Task: $task_id | $start_time"
            echo "     Status: Completed | $duration"
        else
            exit_code=$(grep -i "exit code" "$file" | tail -1)
            echo "     Task: $task_id | $start_time"
            echo "     Status: Did not complete | $exit_code"
            
            # Show last 10 lines if failed
            echo "     Last 10 lines:"
            tail -10 "$file" | sed 's/^/       /'
        fi
        echo ""
    fi
done

echo "============================================================================"
echo "To debug a specific task, check its log files:"
echo "  cat array_job_<job_id>_task_<task_id>.out"
echo "  cat array_job_<job_id>_task_<task_id>.err"
echo "============================================================================"

