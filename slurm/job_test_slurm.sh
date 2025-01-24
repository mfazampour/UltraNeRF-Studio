#!/bin/sh
 
#SBATCH --job-name=sanity_check
#SBATCH --output=slurm/logs/sanity_check_output.log
#SBATCH --error=slurm/logs/sanity_check_error.log
#SBATCH --account=phds
#SBATCH --partition=phds
#SBATCH --time=0-00:05:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Adjust based on your needs, up to 4 GPUs available
#SBATCH --cpus-per-task=4  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=8G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultrabarf
python slurm/slurm_tester.py