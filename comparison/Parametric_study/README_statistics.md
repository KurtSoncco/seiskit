# Transfer Function Statistics Analysis

This document describes the `analyze_statistics.py` script for computing and visualizing statistics across multiple realizations of the parametric study.

## Overview

The script:
1. **Computes statistics** across 5 seed realizations for each rH-CV combination
2. **Plots mean lines** as solid lines
3. **Shows ±1 standard deviation** as transparent shaded regions
4. **Uses unique color/linestyle combinations** for each rH-CV combination (10 colors × 4 linestyles = 40 unique combinations)

## Parameters

The analysis groups data by:
- **rH**: 10, 30, 50 m (correlation length)
- **CV**: 0.1, 0.2, 0.3 (coefficient of variation)
- **Seed**: 10, 20, 30, 40, 50 (spatial field variability)
- **interlayer_seed**: 42 (fixed, for wavy boundary)

## Running the Analysis

```bash
cd comparison/Parametric_study
python analyze_statistics.py
```

### Requirements

- All 45 OpenSees simulations must be completed
- Results should be in `./results/` directory
- The script will load all data and compute transfer functions

## Output

The script generates two interactive plots:
1. **transfer_functions_statistics.html**: Transfer function plot with:
   - Mean transfer function for each rH-CV combination
   - ±1 standard deviation shaded regions
   - Unique color and linestyle for each combination

2. **time_histories_statistics.html**: Time history plot with subplots:
   - **Top subplot**: Surface acceleration at central point
   - **Bottom subplot**: Base acceleration
   - Each subplot shows mean ±1 standard deviation
   - Same color/linestyle assignments as transfer functions

## Features

### Statistics Computation

**Transfer Functions:**
- Aligns all transfer functions to a common frequency grid
- Uses finest frequency resolution from all realizations
- Handles interpolation carefully with NaN values outside frequency ranges
- Computes mean and standard deviation while ignoring NaN values

**Time Histories:**
- Extracts central point from surface node array (middle column)
- Aligns all time histories to a common time grid (dt=0.01s)
- Computes mean and standard deviation for both surface and base accelerations
- Handles both surface and base acceleration time series

### Visualization
- **10 distinct colors** (blue, orange, green, red, purple, brown, pink, gray, olive, cyan)
- **4 linestyles** (solid, dash, dot, dashdot)
- **40 unique combinations** (sufficient for 9 rH-CV combinations)
- Semi-transparent shaded regions (10% opacity) showing ±1 standard deviation
- **Transfer functions**: Log-log axes for frequency and magnitude
- **Time histories**: Linear axes, two subplots (surface on top, base on bottom)

## Data Structure

The script expects results organized as:
```
results/
├── rH_10/
│   ├── CV_0.1/
│   │   ├── rH10_CV0.1_s10/
│   │   ├── rH10_CV0.1_s20/
│   │   ├── rH10_CV0.1_s30/
│   │   ├── rH10_CV0.1_s40/
│   │   └── rH10_CV0.1_s50/
│   ├── CV_0.2/
│   └── CV_0.3/
├── rH_30/
└── rH_50/
```

## Usage Example

```python
from analyze_statistics import (
    compute_tf_statistics,
    plot_tf_statistics,
    compute_time_history_statistics,
    plot_time_history_statistics
)
from seiskit.plot_results import load_datasets
from pathlib import Path

# Load data (same as in run_experiment.py)
data = load_datasets(DATA_CONFIG)

# Compute and plot transfer function statistics
stats = compute_tf_statistics(data, dz=2.5, Vs_min=Vs_min)
plot_tf_statistics(stats, Path("my_tf_plot.html"))

# Compute and plot time history statistics
time_stats = compute_time_history_statistics(data)
plot_time_history_statistics(time_stats, Path("my_time_plot.html"))
```

## Notes

- The interlayer variability (wavy boundary) uses a fixed seed (42) across all runs
- Each spatial field has its own seed for variability
- Statistics are computed across the 5 seed realizations for each rH-CV combination
- The script automatically handles varying frequency ranges across realizations

