# OS_experiment

32 OpenSees simulations: 10× height 15 m, 10× height 100 m, 12× height 57 m. Only the RF seed varies within each height group. Other parameters match `rf_seed_experiment` (Vs1=230, Vs2=1000, rH=30, CV=0.20, 1×1 m, damping_method=global_avg).

## Savio

1. Edit `job_run.sh`: set `#SBATCH --account=fc_XXXX` to your account.
2. From this directory: `sbatch job_run.sh`

The job runs 32 tasks in parallel on one node via gnu-parallel, with `--reservation=maint`. After the run, RSS and CPU efficiency are printed via `sacct` (see job stderr).

## Local / single task

```bash
python run_experiment.py --index 0   # height 15 m, seed 1
python run_experiment.py --index 31 # height 57 m, seed 12
```

## Index layout

| Indices | Height (m) | Seeds |
|---------|------------|-------|
| 0–9     | 15         | 1–10  |
| 10–19   | 100        | 1–10  |
| 20–31   | 57         | 1–12  |

Results go to `results/Vs1_230_thickness_{15|100|57}_rH_30_CV_0.200_s{seed}/`.
