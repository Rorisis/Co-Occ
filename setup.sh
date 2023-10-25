#!/bin/bash
#SBATCH -o files/job.%j.out
#SBATCH -e files/job.%j.err
#SBATCH --partition=i64m1tga40u
#SBATCH -J pytorch
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:8
#SBATCH --qos=low

# module load vasp/6.3.2
# MPIEXEC=`which mpirun`
# $MPIEXEC -np 128  vasp_std

module load cuda/11.3
# module show cuda/11.3
# export CUDA_HOME=/hpc2ssd/softwares/cuda/cuda-11.3
source /hpc2hdd/home/jpan305/miniconda3/bin/activate openocc

cd mmdetection3d
python setup.py install