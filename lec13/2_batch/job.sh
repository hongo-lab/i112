#!/bin/sh
#SBATCH -J mnist
#SBATCH -o mnist.log
#SBATCH -e mnist.err
#SBATCH -p DEF
#SBATCH -n 16

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

cd ${SLURM_SUBMIT_DIR}

singularity exec /app/kagayaki/container_images/tensorflow_2.6.0.sif python mnist.py

