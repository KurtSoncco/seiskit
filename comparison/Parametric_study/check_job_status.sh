#!/bin/bash
# Check SLURM job status for array jobs

echo "============================================================================"
echo "SLURM Job Status Check"
echo "============================================================================"
echo ""

# Get user's jobs
echo "1. Your recent job submissions:"
if command -v squeue &> /dev/null; then
    echo ""
    squeue -u $USER -o "%.10i %.20j %.10T %.10M %.9l %.6D %.20R %.8e" | head -20
else
    echo "   squeue command not available (not on cluster or not in PATH)"
fi

echo ""
echo "2. Recent job history:"
if command -v sacct &> /dev/null; then
    echo ""
    # Show last 10 jobs with more detail
    sacct -u $USER --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,CPUTime,MaxRSS -X | head -15
else
    echo "   sacct command not available (not on cluster)"
fi

echo ""
echo "3. For detailed task information, run:"
echo "   sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem"

echo ""
echo "============================================================================"

