#!/bin/bash


#SBATCH --time=5:20:00           # Increased time for longer training with larger batches

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

# python3 data_utils/datasets.py
# python3 train_operator_AR_rell2_2d.py --config_path configs/spherical/FNO_spherical.yaml
# python3 train_operator_AR_rell2_2d.py --config_path configs/spherical/HFS_spherical.yaml
# python3 train_operator_AR_rell2_2d.py --config_path configs/spherical/MSWT_patching_spherical.yaml
python3 train_operator_AR_rell2_2d.py --config_path configs/shared_folder_config/SFNO.yaml --test_seed 42
#################################################################################################################################


# Testing
python3 test_operator_AR_rell2_2d.py --config_path configs/shared_folder_config/SFNO.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/spherical/FNO_spherical.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/spherical/HFS_spherical.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/spherical/MSWT_patching_spherical.yaml --test_seed 42
# python3 test_operator_AR_rell2_2d.py --config_path configs/spherical/SFNO.yaml --test_seed 42
#################################################################################################################################