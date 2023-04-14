#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH -A lleuven_phm
#SBATCH --partition=gpu
#SBATCH --ntasks=18 --gpus-per-node=1
#SBATCH --job-name=test5hr
#SBATCH -t 05:00:00
#SBATCH --mail-user=brechtdecock1@gmail.com
#SBATCH --mail-type=FAIL,BEGIN,END
# The script directory

module use /apps/leuven/icelake/2021a/modules/all

cd /vsc-hard-mounts/leuven-user/350/vsc35045/Documents/PYTHON

# Activate the virtulenv
conda activate transformer2

# Run job
python transformercombined.py
