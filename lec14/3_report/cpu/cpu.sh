#!/bin/sh

#################################################################
# CPU Job Script only for i112
#                                       2025/05/30 K. Hongo
#################################################################

#PBS -N cpu
#PBS -j oe
#PBS -q DEFAULT
#PBS -l select=1:ngpus=1

source /etc/profile.d/modules.sh
module purge
module load cuda
module load singularity

cd ${PBS_O_WORKDIR}

singularity exec ./pytorch.sif python fashion_mnist.py

