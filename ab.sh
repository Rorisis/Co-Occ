#!/bin/bash
#SBATCH -o files/job.%j.out
#SBATCH -e files/job.%j.err
#SBATCH --partition=i64m1tga40u
#SBATCH -J pytorch
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:1
#SBATCH --qos=low
#SBATCH -w gpu3-12

# module load vasp/6.3.2
# MPIEXEC=`which mpirun`
# $MPIEXEC -np 128  vasp_std

module load cuda/11.3
# module show cuda/11.3
# export CUDA_HOME=/hpc2ssd/softwares/cuda/cuda-11.3
source /hpc2hdd/home/jpan305/miniconda3/bin/activate openocc
nvcc -V 
nvidia-smi
# bash tools/dist_train.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/ablation/baseline.py 8
# bash tools/dist_train.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/ablation/baseline_bifuser.py 8
# bash tools/dist_train.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/uniocc_nusc/cascade_voxel_nusc_multi_dense.py 8 
# bash tools/dist_test.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/uniocc_nusc/cascade_voxel_nusc_multi.py /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/work_dirs/cascade_voxel_nusc_multi/epoch_1.pth 8
bash tools/dist_train.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/uniocc_nusc/cascade_voxel_nusc_lidar_d.py 1