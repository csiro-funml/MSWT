#!/bin/bash


#SBATCH --time=00:10:00           # Increased time for longer training with larger batches

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32        # Increased CPUs for DataLoader workers (H100 can handle more)
#SBATCH --output=slurm-%j.out     # Explicit output file (job ID will be inserted)
#SBATCH --error=slurm-%j.err      # Explicit error file

module load pytorch/2.5.1-py312-cu122-mpi
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

# python3 train_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/FNO_periodic.yaml --test_ratio 0.25 --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/HFS_periodic.yaml --test_ratio 0.25 --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/WNO_periodic.yaml --test_ratio 0.25 --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/SAOT_periodic.yaml --test_ratio 0.25 --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/PDERefiner_periodic.yaml --test_ratio 0.25 --test_seed 42
# python3 train_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/MSWT_patching_periodic.yaml --test_ratio 0.25 --test_seed 42
#################################################################################################################################


# Testing
# python3 test_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/FNO_periodic.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/HFS_periodic.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/WNO_periodic.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/SAOT_periodic.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/PDERefiner_periodic.yaml --test_seed 42
python3 test_operator_AR_rell2_2d.py --config_path configs/periodict_used_in_paper/MSWT_patching_periodic.yaml --test_seed 42

#################################################################################################################################