#!/bin/bash


#SBATCH --time=00:05:00           # Increased time for longer training with larger batches

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32        # Increased CPUs for DataLoader workers (H100 can handle more)
#SBATCH --output=slurm-%j.out     # Explicit output file (job ID will be inserted)
#SBATCH --error=slurm-%j.err      # Explicit error file

module load pytorch/2.5.1-py312-cu122-mpi
module load ffmpeg
source $HOME/.venvs/pytorch/bin/activate


# Print job info immediately (helps verify job started)
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "SLURM allocated $SLURM_CPUS_PER_TASK CPUs for this job"
echo "=========================================="


###############################################################NSE TORCHCFD ############################################################
# Training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python3 train_operator_AR_rell2_2d.py --config_path configs/linear/FNO.yaml --test_ratio 0.25 --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/linear/HFS.yaml --test_ratio 0.25 --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/linear/WNO.yaml --test_ratio 0.25 --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/linear/SAOT.yaml --test_ratio 0.25 --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/linear/PDERefinerUNet.yaml --test_ratio 0.25 --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/linear/MSWT_patching.yaml --test_ratio 0.25 --test_seed 42

#################################################################################################################################


# Testing
# python3 test_operator_AR_rell2_2d.py --config_path configs/linear/FNO.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/linear/HFS.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/linear/WNO.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/linear/SAOT.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/linear/PDERefinerUNet.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/linear/MSWT_patching.yaml --test_seed 42
#################################################################################################################################



# Ablations
# python3 train_operator_AR_rell2_2d.py --config_path configs/ablations/rebutall/MSWT_no_attention.yaml --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/ablations/rebutall/MSWT_no_tokenizer.yaml --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/ablations/rebutall/MSWT_strided_up_downsampling.yaml --test_seed 42


python3 test_operator_AR_rell2_2d.py --config_path configs/ablations/rebutall/MSWT_no_attention.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/ablations/rebutall/MSWT_no_tokenizer.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/ablations/rebutall/MSWT_strided_up_downsampling.yaml --test_seed 42