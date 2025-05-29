#!/bin/sh

#################################################################
# H100 1/4GPU Job Script only for i112
#                                       2025/05/30 K. Hongo
#################################################################

#PBS -N gpu
#PBS -j oe
#PBS -q i112@kvm-pbs
#PBS -l select=1:ngpus=1

source /etc/profile.d/modules.sh
module purge
module load cuda
module load singularity

cd ${PBS_O_WORKDIR}

singularity exec --nv ./pyorch.sif python fashion_mnist.py

