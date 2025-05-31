#!/bin/bash

#################################################################
# CPU Job Script only for i112
#                                       2025/05/30 K. Hongo
#################################################################

#PBS -N cpu_fashion_mnist
#PBS -j oe
#PBS -q DEFAULT
#PBS -l select=1

source /etc/profile.d/modules.sh
module purge
module load singularity

cd ${PBS_O_WORKDIR}

PYTHONNOUSERSITE=1 singularity exec ./pytorch_cpu.sif python3 fashion_mnist.py


