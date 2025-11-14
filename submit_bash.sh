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

################################################################ NSE #################################################################
## Train the model
# CUDA_VISIBLE_DEVICES=0 python3 NSE/train_AR_NO.py --dataset='ns2d_pda' --model='wavelet_transformer_skip' --use_write --lr_method='cossin' --T_in=7 --epochs=3000
# CUDA_VISIBLE_DEVICES=0 python3 NSE/train_AR_NO.py --dataset='ns2d_pda' --model='WaveletTransV2' --use_write --lr_method='cossin' --T_in=7 --epochs=3000 --resume_path=True
# CUDA_VISIBLE_DEVICES=0 python3 NSE/train_AR_NO.py --dataset='ns2d_fno_1e-3' --model='HFS' --lr_method='cossin' --T_in=7 --epochs=3000


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
# CUDA_VISIBLE_DEVICES=0 python3 NSE/test_AR_NO.py --dataset='ns2d_pda' --model='FNO' --lr_method='cossin' --T_in=7 --epochs=3000

# Train/Test the diffusion model
# CUDA_VISIBLE_DEVICES=0 python3 train_diffusion_NO.py --dataset='ns2d_pda' --model='FNO' --use_writer --batch_size=128 --epochs=3000
# CUDA_VISIBLE_DEVICES=0 python3 test_diffusion_NO.py --dataset='ns2d_pda' --model='FNO' --batch_size=128

# Train the PDERefiner model
# CUDA_VISIBLE_DEVICES=0 python3 train_pderefiner.py --dataset='ns2d_pda' --model='PDERefiner' --T_in=7 --T_ar=1 --epochs=3000  --use_writer
# CUDA_VISIBLE_DEVICES=0 python3 test_pderefiner.py --dataset='ns2d_pda' --model='PDERefiner' --T_in=7 --T_ar=1 --batch_size=128


# Utile plotting in the server
# CUDA_VISIBLE_DEVICES=0 python3 NSE/utils_plot.py --dataset='ns2d_pda' --model='FNO' 
# CUDA_VISIBLE_DEVICES=0 python3 utils_plot.py --dataset='ns2d_pda' --model='FNO' 
# CUDA_VISIBLE_DEVICES=0 python3 utils_plot.py --dataset='ns2d_pda' --model='wavelet_transformer' 
# CUDA_VISIBLE_DEVICES=0 python3 utils_plot.py --dataset='ns2d_pda' --model='PDERefiner' 


# Frequency filter testing
# CUDA_VISIBLE_DEVICES=0 python3 NSE/frequency_filter_testing.py --dataset='ns2d_pda' --model='wavelet_transformer' --n_layers=5 --n_autorepressive_steps=100 
# CUDA_VISIBLE_DEVICES=0 python3 NSE/frequency_filter_testing.py --dataset='ns2d_pda' --model='HFS' --n_layers=5 --n_autorepressive_steps=100 

#################################################################################################################################



###############################################################NSE TORCHCFD ############################################################
# Training
# CUDA_VISIBLE_DEVICES=0 python3 NSE/train_AR_NO.py --dataset='ns2d_dedalus' --model='FNO' --use_writer --lr_method='cossin' --T_in=7 --epochs=3000 --normalize_strategy='minmax'
# CUDA_VISIBLE_DEVICES=0 python3 NSE/train_AR_NO.py --dataset='ns2d_dedalus' --model='FNO' --use_writer --lr_method='cossin' --T_in=7 --epochs=3000 --normalize_strategy='zscore'
# CUDA_VISIBLE_DEVICES=0 python3 NSE/train_AR_NO.py --dataset='ns2d_dedalus' --model='FNO' --use_writer --modes=32 --width=32 --n_layers=4  --lr_method='cossin' --T_in=7 --epochs=3000 --normalize_strategy='zscore'
# CUDA_VISIBLE_DEVICES=0 python3 NSE/train_AR_NO.py --dataset='ns2d_dedalus' --model='FNO' --use_writer --modes=16 --width=32 --n_layers=4 --lr=3e-4 --lr_method='cossin' --T_in=7 --epochs=5000 --normalize_strategy='zscore'
# CUDA_VISIBLE_DEVICES=0 python3 NSE/train_AR_NO.py --dataset='ns2d_dedalus' --model='HFS' --use_writer --lr_method='cossin' --T_in=7 --epochs=3000
# Optimized training command for H100 GPU (FP32, no mixed precision)
# Reduced batch size to avoid OOM, using gradient accumulation to maintain effective batch size
# Set PYTORCH_CUDA_ALLOC_CONF to reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Training command (use_writer disabled, epochs=1000)
# CUDA_VISIBLE_DEVICES=0 python3 -u NSE/train_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO \
#     --modes 32 --width 32 --n_layers 4 --T_in 1 --T_ar 1  --normalize 1 --normalize_strategy zscore \
#     --form velocity --batch_size 64 --gradient_accumulation_steps 4 --epochs 1000 --num_workers 8 --pin_memory --prefetch_factor 1 --use_writer  # Disabled for now


# CUDA_VISIBLE_DEVICES=0 python3 -u NSE/train_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO \
#     --modes 64 --width 32 --n_layers 4 --T_in 1 --T_ar 1  --normalize 1 --normalize_strategy zscore \
#     --form velocity --batch_size 64 --gradient_accumulation_steps 4 --epochs 1000 --num_workers 8 --pin_memory --prefetch_factor 1 --use_writer # Disabled for now


# CUDA_VISIBLE_DEVICES=0 python3 -u NSE/train_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO \
#     --modes 128 --width 32 --n_layers 4 --T_in 1 --T_ar 1  --normalize 1 --normalize_strategy zscore \
#     --form velocity --batch_size 128 --gradient_accumulation_steps 4 --epochs 1000 --num_workers 8 --pin_memory --prefetch_factor 1 --use_writer --resume_path=True # Disabled for now



# # Testing
# CUDA_VISIBLE_DEVICES=0 python3 -u NSE/test_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO \
#     --modes 32 --width 32 --n_layers 4 --T_in 1 --T_ar 1  --normalize 1 --normalize_strategy zscore --epochs 1000 \
#     --form velocity --num_steps 30 --dataset_type long # Disabled for now


 python3 NSE/analyze_predictions_collaborator.py --mode explain \
  --input_file /scratch3/wan410/operator_learning_model/ns2d_dedalus_big_FNO_mod32_wid32_lay4_ntrain32006_formvelocity_lossfourier_logscaleTrue_warmup0/test_data_prediction_long.npz

# CUDA_VISIBLE_DEVICES=0 python3 -u NSE/test_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO \
#     --modes 64 --width 32 --n_layers 4 --T_in 1 --T_ar 1  --normalize 1 --normalize_strategy zscore --epochs 1000 \
#     --form velocity --num_steps 30 # Disabled for now

# CUDA_VISIBLE_DEVICES=0 python3 -u NSE/test_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO \
#     --modes 128 --width 32 --n_layers 4 --T_in 1 --T_ar 1  --normalize 1 --normalize_strategy zscore --epochs 1000 \
#     --form velocity --num_steps 30 # Disabled for now


## Ablation on Loss Regularization 
# CUDA_VISIBLE_DEVICES=0 python3 -u NSE/train_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO \
#     --modes 32 --width 32 --n_layers 4 --T_in 1 --T_ar 1  --normalize 1 --normalize_strategy zscore \
#     --form velocity --batch_size 64 --gradient_accumulation_steps 4 --epochs 1000 --num_workers 8 --pin_memory --prefetch_factor 1 \
#      --warmup_epochs 0 --loss_type fourier --fourier_logscale False \
#     --use_writer  # Disabled for now

# CUDA_VISIBLE_DEVICES=0 python3 -u NSE/train_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO \
#     --modes 32 --width 32 --n_layers 4 --T_in 1 --T_ar 1  --normalize 1 --normalize_strategy zscore \
#     --form velocity --batch_size 64 --gradient_accumulation_steps 4 --epochs 1000 --num_workers 8 --pin_memory --prefetch_factor 1 \
#      --warmup_epochs 0 --loss_type fourier --fourier_logscale True \
#     --use_writer  # Disabled for now


# CUDA_VISIBLE_DEVICES=0 python3 -u NSE/train_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO \
#     --modes 32 --width 32 --n_layers 4 --T_in 1 --T_ar 1  --normalize 1 --normalize_strategy zscore \
#     --form velocity --batch_size 64 --gradient_accumulation_steps 4 --epochs 1000 --num_workers 8 --pin_memory --prefetch_factor 1 \
#      --warmup_epochs 300 --loss_type fourier --fourier_logscale False \
#     --use_writer  # Disabled for now


# CUDA_VISIBLE_DEVICES=0 python3 -u NSE/train_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO \
#     --modes 32 --width 32 --n_layers 4 --T_in 1 --T_ar 1  --normalize 1 --normalize_strategy zscore \
#     --form velocity --batch_size 64 --gradient_accumulation_steps 4 --epochs 1000 --num_workers 8 --pin_memory --prefetch_factor 1 \
#      --warmup_epochs 300 --loss_type fourier --fourier_logscale True \
#     --use_writer  # Disabled for now

# CUDA_VISIBLE_DEVICES=0 python3 -u NSE/test_AR_NO_Dedalus.py --dataset ns2d_dedalus_big --model FNO \
#     --modes 32 --width 32 --n_layers 4 --T_in 1 --T_ar 1  --normalize 1 --normalize_strategy zscore \
#     --form velocity --batch_size 64 --gradient_accumulation_steps 4 --epochs 1000 --num_workers 8 --pin_memory --prefetch_factor 1 \
#      --warmup_epochs 0 --loss_type fourier --fourier_logscale True --dataset_type long



# CUDA_VISIBLE_DEVICES=0 python3 NSE/utils_plot.py --dataset='ns2d_dedalus' --model='FNO' --normalize_strategy='zscore'

#################################################################################################################################



################################################################ SWE #################################################################
# Train the model (including resume training)
# CUDA_VISIBLE_DEVICES=0 python3 SWE/train_AR_NO.py --dataset='sw2d_pda' --model='FNO' --lr_method='cossin' --T_in=7 --use_writer --resume_path=True --epochs=2000 --batch_size=128
# CUDA_VISIBLE_DEVICES=0 python3 SWE/train_AR_NO.py --dataset='sw2d_pda' --model='wavelet_transformer' --lr_method='cossin' --T_in=7 --use_writer --resume_path=True --epochs=2000 --batch_size=128
# CUDA_VISIBLE_DEVICES=0 python3 SWE/train_AR_NO.py --dataset='sw2d_pda' --model='HFS' --lr_method='cossin' --T_in=7 --epochs=2000 --use_writer --resume_path=True --batch_size=128
# CUDA_VISIBLE_DEVICES=0 python3 SWE/train_AR_NO.py --dataset='sw2d_pda' --model='WaveletTransV2' --lr_method='cossin' --T_in=7 --epochs=2000 --batch_size=128 --use_writer --resume_path=True

# Test the model
# CUDA_VISIBLE_DEVICES=0 python3 SWE/test_AR_NO.py --dataset='sw2d_pda' --model='FNO' --T_in=7 --epochs=2000
# CUDA_VISIBLE_DEVICES=0 python3 SWE/test_AR_NO.py --dataset='sw2d_pda' --model='wavelet_transformer' --T_in=7 --epochs=2000
# CUDA_VISIBLE_DEVICES=0 python3 SWE/test_AR_NO.py --dataset='sw2d_pda' --model='HFS' --T_in=7 --epochs=2000

# CUDA_VISIBLE_DEVICES=0 python3 SWE/utils_plot.py --dataset='sw2d_pda' --model='HFS' 
################################################################ SWE #################################################################



