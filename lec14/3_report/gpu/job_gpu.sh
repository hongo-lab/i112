#!/bin/bash

#################################################################
# H100 1/4GPU Job Script only for i112
#                                       2025/05/30 K. Hongo
#################################################################

#PBS -N gpu_fashion_mnist
#PBS -j oe
#PBS -q i112@kvm-pbs
#PBS -l select=1:ngpus=1

source /etc/profile.d/modules.sh
module purge
module load 12.8u1
module load singularity

#module list

#echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
#nvidia-smi

cd ${PBS_O_WORKDIR}

PYTHONNOUSERSITE=1 singularity exec --nv ./pytorch_h100.sif python3 fashion_mnist.py

