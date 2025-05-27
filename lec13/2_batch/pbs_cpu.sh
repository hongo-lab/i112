#!/bin/sh
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -j oe
#PBS -N mnist
#PBS -q DEFAULT

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

source /etc/profile.d/modules.sh
module purge
module load singularity

cd ${PBS_O_WORKDIR}

singularity exec /app/container_images/tensorflow_2.6.0.sif python mnist.py

