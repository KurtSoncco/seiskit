# Fix for Hanging Jobs Issue

## Problem Identified

The parametric study jobs were hanging because:

1. **Single-block analysis call**: The dynamic analysis called `ops.analyze(nsteps, config.dt)` all at once (e.g., 1500 steps in one call). If OpenSees got stuck on a step, there was no way to detect which step or provide progress information.

2. **No timeout mechanism**: If a step failed to converge or hung, the analysis could run indefinitely until SLURM's 2-hour timeout killed it.

3. **No progress tracking**: No way to see which step the analysis was on, making it impossible to diagnose hangs.

## Solution Implemented

Modified `seiskit/isolated_runner.py` to:

1. **Batch processing**: Run the analysis in batches (typically ~100 steps per batch, ~10 batches total for 1500 steps) instead of all at once. This provides:
   - Progress tracking after each batch
   - Faster hang detection (within 10 minutes per batch instead of potentially 2 hours)
   - Better error messages showing exactly which batch failed

2. **Timeout detection**: Each batch has a 10-minute timeout. If a batch takes longer, it raises an error with detailed information about:
   - Which batch/step range failed
   - How many steps completed successfully before the hang

3. **Progress reporting**: Prints progress after each batch including:
   - Percentage complete
   - Elapsed time
   - Average time per step
   - Estimated time remaining

## Expected Output

You should now see output like:
```
[rH10_CV0.1_s10] Starting dynamic analysis: 1500 steps (dt=0.01s, duration=15.0s)
[rH10_CV0.1_s10] Progress: 100/1500 steps (6.7%) | Elapsed: 12.3s | Avg: 0.123s/step | Est. remaining: 172.2s
[rH10_CV0.1_s10] Progress: 200/1500 steps (13.3%) | Elapsed: 24.5s | Avg: 0.122s/step | Est. remaining: 158.6s
...
```

If a job hangs, you'll see an error like:
```
RuntimeError: Dynamic analysis batch (steps 501-600/1500) for run rH10_CV0.1_s10 took 601.2s (> 600s timeout). This suggests the analysis is hung. Last 500 steps completed successfully.
```

## Next Steps

1. **Monitor the jobs**: The progress output will show which cases are progressing and which might be stuck.

2. **Identify problematic cases**: If specific parameter combinations (e.g., low rH values with high CV) consistently hang, they may indicate:
   - Numerical instabilities in the model
   - Need for adjusted convergence tolerances or max iterations
   - Potentially unstable VS field realizations

3. **Fine-tune if needed**: If the 10-minute batch timeout is too strict or too lenient, adjust `max_time_per_batch` in `seiskit/isolated_runner.py` (line 214).

## Additional Recommendations

If certain cases continue to hang, consider:
- Increasing `max_dynamic_iter` for problematic cases
- Relaxing `dynamic_tolerance` slightly (e.g., from 1e-4 to 1e-3)
- Filtering out unstable VS field realizations before analysis
- Using adaptive time stepping for problematic cases

