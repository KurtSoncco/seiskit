Central folder to account for all analysis related to emulator_first, 
emulator_second, emulator_third. 

- It goes from raw data selection, 
- Computation of transfer functions,
- Computation of intensity measures like PGA, PSA(at f0 of 1D), Arias Intensity. 
- COmputation of 1D cases for each subfolder.
- Windows peak algorithm. 

Then, Plots.
- Plots of transfer functions. 
-- Comparison of transfer functions. 
-- Comparison using normalized frequency transfer functions. 

- Plots of statistics on peak values. 
-- Box Plots. 
-- Violin Plots. 
-- 


- Analysis of results on peak values and intensity measures. 
-- Comparison of meadin response for each metric, distributions. 
-- Distribution checks. 


The results are saved on the box mounted folder. /mnt/box/GIG Lab - UC Berkeley/Projects/Statistical Analysis/complete$ ls
figures  mixed_model  peak_analysis        stats_IM
mHVSR    ngboost      spatial_correlation  tf_results