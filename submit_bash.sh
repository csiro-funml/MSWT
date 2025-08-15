#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)



module load pytorch/2.5.1-py312-cu122-mpi
# module load pytorch/2.5.1-py312-cu124-mpi-sota
# module load pytorch/2.1.0-py312-cu122-mpi
module load tensorflow/2.16.1-pip-py312-cuda122
module load parallel python
source /scratch3/wan410/venvs/testing/bin/activate



## Train the model
# (CUDA_VISIBLE_DEVICES=0 python3 train_customized.py --dataset='ns2d_pda' --use_writer --model='wavelet_transformer' --lr_method='cossin' --T_in=7)
(CUDA_VISIBLE_DEVICES=0 python3 train_customized.py --dataset='ns2d_pda' --model='FNO' --lr_method='cossin' --T_in=7)

## resume training
# CUDA_VISIBLE_DEVICES=0 python3 train_customized.py  --use_writer --dataset='ns2d_pda' --model='wavelet_transformer' --lr_method='cossin' --resume_path=True


## Test the model
# CUDA_VISIBLE_DEVICES=0 python3 test_customized.py --dataset='sw2d_pda' --model='FNO'
# CUDA_VISIBLE_DEVICES=0 python3 test_customized.py --dataset='ns2d_pda' --model='FNO' --lr_method='cossin' --T_in=7