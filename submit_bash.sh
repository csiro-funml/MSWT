#!/bin/bash


#SBATCH --time=20:00:00

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=9        # cpu-cores per task (>1 if multi-threaded tasks)


# module load numpy/2.0.0-py312
module load pytorch/2.5.1-py312-cu122-mpi
# module load pytorch/2.5.1-py312-cu124-mpi-sota
# module load pytorch/2.1.0-py312-cu122-mpi
module load tensorflow/2.16.1-pip-py312-cuda122
module load parallel python
# source /scratch3/wan410/venvs/testing/bin/activate
source $HOME/.venvs/pytorch/bin/activate


################################################################ NSE #################################################################
## Train the model
CUDA_VISIBLE_DEVICES=0 python3 NSE/train_AR_NO.py --dataset='ns2d_pda' --model='FFormer' --use_writer --lr_method='cossin' --T_in=7 --epochs=3000 --resume_path=True
# CUDA_VISIBLE_DEVICES=0 python3 train_AR_NO.py --dataset='ns2d_pda' --model='HFS' --use_write --lr_method='cossin' --T_in=7 --epochs=3000
# CUDA_VISIBLE_DEVICES=0 python3 train_AR_NO.py --dataset='ns2d_pda' --model='FNO' --use_write --lr_method='cossin' --T_in=7 --epochs=3000
# CUDA_VISIBLE_DEVICES=0 python3 train_AR_NO.py --dataset='ns2d_pda' --model='wavelet_transformer' --use_write --lr_method='cossin' --T_in=7 --epochs=3000
# CUDA_VISIBLE_DEVICES=0 python3 train_AR_NO.py --dataset='ns2d_pda' --model='UNet' --use_write --lr_method='cossin' --T_in=7 --epochs=3000
# CUDA_VISIBLE_DEVICES=0 python3 train_AR_NO.py --dataset='ns2d_pda' --model='HANO' --use_write --lr_method='cossin' --T_in=7 --epochs=3000
# CUDA_VISIBLE_DEVICES=0 python3 train_AR_NO.py --dataset='ns2d_pda' --model='UNet_withoutHFS' --use_write --lr_method='cossin' --T_in=7 --epochs=3000


## Test the model
# CUDA_VISIBLE_DEVICES=0 python3 test_AR_NO.py --dataset='sw2d_pda' --model='FNO' 
# CUDA_VISIBLE_DEVICES=0 python3 test_AR_NO.py --dataset='ns2d_pda' --model='FNO' --lr_method='cossin' --T_in=7 --epochs=3000
# CUDA_VISIBLE_DEVICES=0 python3 test_AR_NO.py --dataset='ns2d_pda' --model='UNet_withoutHFS'
# CUDA_VISIBLE_DEVICES=0 python3 test_AR_NO.py --dataset='ns2d_pda' --model='UNet'
# CUDA_VISIBLE_DEVICES=0 python3 NSE/test_AR_NO.py --dataset='ns2d_pda' --model='HFS' --lr_method='cossin' --T_in=7 --epochs=3000

# Train/Test the diffusion model
# CUDA_VISIBLE_DEVICES=0 python3 train_diffusion_NO.py --dataset='ns2d_pda' --model='FNO' --use_writer --batch_size=128 --epochs=3000
# CUDA_VISIBLE_DEVICES=0 python3 test_diffusion_NO.py --dataset='ns2d_pda' --model='FNO' --batch_size=128

# Train the PDERefiner model
# CUDA_VISIBLE_DEVICES=0 python3 train_pderefiner.py --dataset='ns2d_pda' --model='PDERefiner' --T_in=7 --T_ar=1 --epochs=3000  --use_writer
# CUDA_VISIBLE_DEVICES=0 python3 test_pderefiner.py --dataset='ns2d_pda' --model='PDERefiner' --T_in=7 --T_ar=1 --batch_size=128


# Utile plotting in the server
# CUDA_VISIBLE_DEVICES=0 python3 NSE/utils_plot.py --dataset='ns2d_pda' --model='HFS' 
# CUDA_VISIBLE_DEVICES=0 python3 utils_plot.py --dataset='ns2d_pda' --model='FNO' 
# CUDA_VISIBLE_DEVICES=0 python3 utils_plot.py --dataset='ns2d_pda' --model='wavelet_transformer' 
# CUDA_VISIBLE_DEVICES=0 python3 utils_plot.py --dataset='ns2d_pda' --model='PDERefiner' 


# Frequency filter testing
# CUDA_VISIBLE_DEVICES=0 python3 SWE/frequency_filter_testing.py --dataset='ns2d_pda' --model='wavelet_transformer' --n_layers=5 --n_autorepressive_steps=100 

#################################################################################################################################



################################################################ SWE #################################################################
# Train the model (including resume training)
# CUDA_VISIBLE_DEVICES=0 python3 SWE/train_AR_NO.py --dataset='sw2d_pda' --model='FNO' --lr_method='cossin' --T_in=7 --use_writer --resume_path=True --epochs=2000 --batch_size=128
# CUDA_VISIBLE_DEVICES=0 python3 SWE/train_AR_NO.py --dataset='sw2d_pda' --model='wavelet_transformer' --lr_method='cossin' --T_in=7 --use_writer --resume_path=True --epochs=2000 --batch_size=128
# CUDA_VISIBLE_DEVICES=0 python3 SWE/train_AR_NO.py --dataset='sw2d_pda' --model='HFS' --lr_method='cossin' --T_in=7 --epochs=2000 --use_writer --resume_path=True --batch_size=128

# Test the model
# CUDA_VISIBLE_DEVICES=0 python3 SWE/test_AR_NO.py --dataset='sw2d_pda' --model='FNO' --T_in=7 --epochs=2000
# CUDA_VISIBLE_DEVICES=0 python3 SWE/test_AR_NO.py --dataset='sw2d_pda' --model='wavelet_transformer' --T_in=7 --epochs=2000
# CUDA_VISIBLE_DEVICES=0 python3 SWE/test_AR_NO.py --dataset='sw2d_pda' --model='HFS' --T_in=7 --epochs=2000

# CUDA_VISIBLE_DEVICES=0 python3 SWE/utils_plot.py --dataset='sw2d_pda' --model='HFS' 
################################################################ SWE #################################################################



