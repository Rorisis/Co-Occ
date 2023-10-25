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
nvcc -V 
nvidia-smi
# bash tools/dist_train.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/ablation/baseline.py 8
# bash tools/dist_train.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/uniocc_nusc/cascade_voxel_nusc_multi_d.py 8 
# bash tools/dist_train.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/uniocc_nusc/cascade_voxel_nusc_multi_d.py 8 
# bash tools/dist_train.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/uniocc_nusc/cascade_voxel_nusc_multi_swin.py 8 
# bash tools/dist_train.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/uniocc_nusc/multi_randomray.py 8 
# bash tools/dist_train.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/uniocc_nusc/d_ray64downsample8.py 8
# bash tools/dist_test.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/uniocc_nusc/cascade_voxel_nusc_multi.py /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/work_dirs/cascade_voxel_nusc_multi/epoch_1.pth 8
bash tools/dist_train.sh /hpc2hdd/home/jpan305/pytorch/MoEOccupancy/projects/configs/uniocc_kitti/scale_multi_kitti.py 8
