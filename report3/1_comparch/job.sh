#!/bin/bash

#################################################################
# H100 1/4GPU Job Script only for i112
#                                       2025/05/30 K. Hongo
#################################################################

#PBS -N gpu_vs_cpu
#PBS -j oe
#PBS -q i112@kvm-pbs
#PBS -l select=1:ngpus=1

source /etc/profile.d/modules.sh
module purge
module load 12.8u1
module load singularity

cd ${PBS_O_WORKDIR}

PYTHONNOUSERSITE=1 singularity exec --nv ./pytorch.sif python3 comparch.py

