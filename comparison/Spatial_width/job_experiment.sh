#!/bin/bash
#SBATCH --job-name=spatial-width
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --array=0-7
#SBATCH --output=array_job_%A_task_%a.out
#SBATCH --error=array_job_%A_task_%a.err

set -euo pipefail

# Ensure we can use 'module' in non-interactive shells
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

# Clean module env and load required toolchain
module purge
module load gcc/13.2.0
module load openblas/0.3.24
module load openmpi/4.1.6 

# Match thread counts to allocated CPUs (helps OpenBLAS and any OpenMP usage)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Activate your project venv (absolute path for HPC home)
source /global/home/users/kurtwal98/seiskit/.venv/bin/activate

# Make OpenSeesPy's native libs visible
export LD_LIBRARY_PATH=/global/home/users/kurtwal98/seiskit/.venv/lib/python3.11/site-packages/openseespylinux/lib:${LD_LIBRARY_PATH:-}

# Run from the directory you submitted the job (keeps relative paths sane)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Execute the array task; the script reads SLURM_ARRAY_TASK_ID automatically
python run_experiment_SLURM.py
