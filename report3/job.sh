#!/bin/bash

#######################################
# H100 1/4GPU Job Script only for i112
# 2026/05/31 K. Hongo
#######################################

#SBATCH -J gpu_vs_cpu
#SBATCH -p i112
#SBATCH -A i112
#SBATCH --gpus=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

module purge
module load cuda/12.8u1
module load singularity

cd ${SLURM_SUBMIT_DIR}

export PYTHONNOUSERSITE=1

singularity exec --nv \
    /home/lecture/is/i112/2026/pytorch.sif \
    python3 fashion_mnist.py
