#!/bin/bash


#SBATCH --time=00:20:00           # Increased time for longer training with larger batches

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

#################################################################################################################################
# ABLATION STUDIES - ORGANIZED STRUCTURE
# 
# All ablation studies have been organized into separate files in the 'ablations/' directory:
#   - ablations/model_size_ablation.sh      : Model size experiments (modes: 32, 64, 128)
#   - ablations/loss_function_ablation.sh   : Loss function experiments (fourier, fourier2d, logscale variants)
#   - ablations/steps_ahead_ablation.sh     : Steps ahead prediction (T_out: 1, 5)
#   - ablations/run_ablation.sh             : Master script to view/run all ablations
#
# To view all ablation commands:
#   bash ablations/run_ablation.sh all show
#
# To view specific ablation:
#   bash ablations/run_ablation.sh model_size show
#   bash ablations/run_ablation.sh loss_function show
#   bash ablations/run_ablation.sh steps_ahead show
#
# See ablations/README.md for more details
#################################################################################################################################

# Current active command (uncomment and modify as needed)
CUDA_VISIBLE_DEVICES=0 python3 -u NSE/train_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO --modes 32 --width 32 --n_layers 4 --T_in 1 --T_out 2 --normalize 1 --normalize_strategy zscore --form velocity --batch_size 64 --gradient_accumulation_steps 4 --epochs 1000 --num_workers 8 --pin_memory --prefetch_factor 1 --warmup_epochs 500 --use_writer

# Method 1 - Use the helper script (recommended):
#   bash ablations/run_specific.sh steps_ahead 5 show    # Show T_out=5 commands
#   bash ablations/run_specific.sh steps_ahead 5 train   # Show only training command
#   bash ablations/run_specific.sh steps_ahead 5 test    # Show only testing command
#
# Method 2 - Filter output directly:
#   bash ablations/steps_ahead_ablation.sh | grep -A 1 "T_out=5" | grep "^CUDA_VISIBLE_DEVICES"
# 
# Then copy the output command here and uncomment it

# CUDA_VISIBLE_DEVICES=0 python3 NSE/utils_plot.py --dataset='ns2d_dedalus' --model='FNO' --normalize_strategy='zscore'

#################################################################################################################################