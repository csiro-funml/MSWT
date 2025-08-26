#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)


# module load numpy/2.0.0-py312
module load pytorch/2.5.1-py312-cu122-mpi
# module load pytorch/2.5.1-py312-cu124-mpi-sota
# module load pytorch/2.1.0-py312-cu122-mpi
module load tensorflow/2.16.1-pip-py312-cuda122
module load parallel python
# source /scratch3/wan410/venvs/testing/bin/activate
source $HOME/.venvs/pytorch/bin/activate



## Train the model
# CUDA_VISIBLE_DEVICES=0 python3 train_AR_NO.py --dataset='ns2d_pda' --use_writer --model='FNO' --lr_method='cossin' --T_in=7 --epochs=10000
# CUDA_VISIBLE_DEVICES=0 python3 train_AR_NO.py --dataset='ns2d_pda' --model='FNO' --lr_method='cossin' --T_in=7 --epochs=2000

## resume training
# CUDA_VISIBLE_DEVICES=0 python3 train_AR_NO.py  --use_writer --dataset='ns2d_pda' --model='wavelet_transformer' --lr_method='cossin' --resume_path=True

## Train the HFS model
# CUDA_VISIBLE_DEVICES=0 python3 train_HFS_NO.py --dataset='ns2d_pda' --model='HFS' --use_writer --T_in=7 --use_writer


## Test the model
# CUDA_VISIBLE_DEVICES=0 python3 test_AR_NO.py --dataset='sw2d_pda' --model='FNO'
# CUDA_VISIBLE_DEVICES=0 python3 test_AR_NO.py --dataset='ns2d_pda' --model='FNO' --lr_method='cossin' --T_in=7


# Train/Test the diffusion model
# CUDA_VISIBLE_DEVICES=0 python3 train_diffusion_NO.py --dataset='ns2d_pda' --model='FNO' --use_writer --batch_size=128
# CUDA_VISIBLE_DEVICES=0 python3 test_diffusion_NO.py --dataset='ns2d_pda' --model='FNO' --batch_size=128

# Test the HFS model
CUDA_VISIBLE_DEVICES=0 python3 test_HFS_NO.py --dataset='ns2d_pda' --model='HFS' --T_in=7


# Utile plotting in the server
# CUDA_VISIBLE_DEVICES=0 python3 utils_plot.py --dataset='ns2d_pda' --model='wavelet_transformer' 