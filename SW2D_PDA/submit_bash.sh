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

# python3 train_operator_AR_rell2_2d.py --config_path configs/FNO_periodic.yaml --test_ratio 0.25
# python3 train_operator_AR_rell2_2d.py --config_path configs/HFS.yaml --test_ratio 0.25
# python3 train_operator_AR_rell2_2d.py --config_path configs/WNO.yaml --test_ratio 0.25 --resume_training
# python3 train_operator_AR_rell2_2d.py --config_path configs/SAOT.yaml --test_ratio 0.25 --resume_training
# python3 train_operator_AR_rell2_2d.py --config_path configs/PDERefiner.yaml --test_ratio 0.25 
# python3 train_operator_AR_rell2_2d.py --config_path configs/MSWT.yaml --test_ratio 0.25 
# python3 train_operator_AR_rell2_2d.py --config_path configs/PDERefinerUNet.yaml --test_ratio 0.25 --resume_training
# python3 train_operator_AR_rell2_2d.py --config_path configs/ablations/MSWT_NodecoderAttn.yaml --test_ratio 0.25
# python3 train_operator_AR_rell2_2d.py --config_path configs/ablations/MSWT_double_attn.yaml --test_ratio 0.25
# python3 train_operator_AR_rell2_2d.py --config_path configs/ablations/MSWT_NodecoderAttn_Group4.yaml --test_ratio 0.25
# python3 train_operator_AR_rell2_2d.py --config_path configs/ablations/MSWT_DeNoAttn_StackLayers.yaml --test_ratio 0.25
# python3 train_operator_AR_rell2_2d.py --config_path configs/MSWT_periodic.yaml --test_ratio 0.25 --resume_training
# python3 train_operator_AR_rell2_2d.py --config_path configs/MSWT_periodic_nlayers4.yaml --test_ratio 0.25
# python3 train_operator_AR_rell2_2d.py --config_path configs/MSWT_periodic_wave_db2.yaml --test_ratio 0.25
# python3 train_operator_AR_rell2_2d.py --config_path configs/MSWT_periodic_patching.yaml --test_ratio 0.25
#################################################################################################################################


# Testing
# python3 test_operator_AR_rell2_2d.py --config_path configs/FNO.yaml
# python3 test_operator_AR_rell2_2d.py --config_path configs/HFS.yaml
# python3 test_operator_AR_rell2_2d.py --config_path configs/WNO.yaml
# python3 test_operator_AR_rell2_2d.py --config_path configs/SAOT.yaml
# python3 test_operator_AR_rell2_2d.py --config_path configs/PDERefiner.yaml
# python3 test_operator_AR_rell2_2d.py --config_path configs/PDERefinerUNet.yaml
# python3 test_operator_AR_rell2_2d.py --config_path configs/MSWT.yaml
# python3 test_operator_AR_rell2_2d.py --config_path configs/MSWT_NodecoderAttn.yaml
# python3 test_operator_AR_rell2_2d.py --config_path configs/MSWT_double_attn.yaml
# python3 test_operator_AR_rell2_2d.py --config_path configs/MSWT_periodic.yaml
# python3 test_operator_AR_rell2_2d.py --config_path configs/MSWT_periodic_nlayers4.yaml

python3 test_operator_AR_rell2_2d.py --config_path configs/periodic/FNO_periodic.yaml --test_seed 42
python3 test_operator_AR_rell2_2d.py --config_path configs/periodic/MSWT_patching_periodic.yaml --test_seed 42
#################################################################################################################################