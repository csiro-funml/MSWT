#!/bin/bash

# Multi-GPU training script using PyTorch Lightning
# Usage: ./run_lightning_multi_gpu.sh [options]
#SBATCH --time=00:20:00
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

# Default parameters
GPUS=4
BATCH_SIZE=100
LEARNING_RATE=1e-4
MAX_EPOCHS=1000
STRATEGY="ddp"
PRECISION="16-mixed"
NUM_WORKERS=4
MODEL="diffusion"
DATASET="ns2d_pda"
COMMENT=""
LOG_PATH="/scratch3/wan410/operator_learning_model/"
VAL_CHECK_INTERVAL=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --comment)
            COMMENT="$2"
            shift 2
            ;;
        --log_path)
            LOG_PATH="$2"
            shift 2
            ;;
        --val_check_interval)
            VAL_CHECK_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --gpus <n>                   Number of GPUs to use (default: 4, -1 for all)"
            echo "  --batch_size <n>             Batch size per GPU (default: 100)"
            echo "  --learning_rate <lr>         Learning rate (default: 1e-4)"
            echo "  --max_epochs <n>             Maximum number of epochs (default: 1000)"
            echo "  --strategy <strategy>        Training strategy (default: ddp)"
            echo "                              Options: ddp, ddp_spawn, dp, ddp_sharded"
            echo "  --precision <precision>      Training precision (default: 16-mixed)"
            echo "                              Options: 16-mixed, 32, bf16-mixed"
            echo "  --num_workers <n>            Number of data loading workers (default: 4)"
            echo "  --model <name>               Model type (default: diffusion)"
            echo "  --dataset <name>             Dataset name (default: ns2d_pda)"
            echo "  --comment <text>             Comment for logging"
            echo "  --log_path <path>            Log directory path"
            echo "  --val_check_interval <n>     Validation check interval in epochs (default: 10)"
            echo "  -h, --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Train on 4 GPUs with default settings"
            echo "  $0"
            echo ""
            echo "  # Train on 2 GPUs with larger batch size"
            echo "  $0 --gpus 2 --batch_size 200"
            echo ""
            echo "  # Train with mixed precision and custom learning rate"
            echo "  $0 --precision 16-mixed --learning_rate 2e-4"
            echo ""
            echo "  # Train with specific strategy for better memory efficiency"
            echo "  $0 --strategy ddp_sharded --precision bf16-mixed"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting PyTorch Lightning multi-GPU training with the following settings:"
echo "  GPUs: $GPUS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Strategy: $STRATEGY"
echo "  Precision: $PRECISION"
echo "  Num workers: $NUM_WORKERS"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Comment: $COMMENT"
echo "  Log path: $LOG_PATH"
echo "  Validation check interval: $VAL_CHECK_INTERVAL"
echo ""

# Build the command
CMD="python train_diffusion_multiple_gpus.py"
CMD="$CMD --gpus $GPUS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --max_epochs $MAX_EPOCHS"
CMD="$CMD --strategy $STRATEGY"
CMD="$CMD --precision $PRECISION"
CMD="$CMD --num_workers $NUM_WORKERS"
CMD="$CMD --model $MODEL"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --log_path $LOG_PATH"
CMD="$CMD --val_check_interval $VAL_CHECK_INTERVAL"

if [ "$COMMENT" != "" ]; then
    CMD="$CMD --comment $COMMENT"
fi

echo "Running: $CMD"
echo ""

# Execute the command
eval $CMD
