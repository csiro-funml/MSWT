#!/bin/bash

# Multi-GPU training script for diffusion neural operator
# Usage: ./run_multi_gpu.sh [options]

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
WORLD_SIZE=4
MODEL="diffusion"
DATASET="ns2d_pda"
COMMENT=""
LOG_PATH="/scratch3/wan410/operator_learning_model/"
USE_WRITER=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --world_size)
            WORLD_SIZE="$2"
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
        --use_writer)
            USE_WRITER=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --world_size <n>     Number of GPUs to use (default: 4)"
            echo "  --model <name>       Model type (default: diffusion)"
            echo "  --dataset <name>     Dataset name (default: ns2d_pda)"
            echo "  --comment <text>     Comment for logging"
            echo "  --log_path <path>    Log directory path"
            echo "  --use_writer         Enable tensorboard logging"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Starting multi-GPU training with the following settings:"
echo "  GPUs: $WORLD_SIZE"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Comment: $COMMENT"
echo "  Log path: $LOG_PATH"
echo "  Use writer: $USE_WRITER"
echo ""

# Build the command
CMD="python train_diffusion_multiple_gpus.py --distributed --world_size $WORLD_SIZE --model $MODEL --dataset $DATASET --log_path $LOG_PATH"

if [ "$COMMENT" != "" ]; then
    CMD="$CMD --comment $COMMENT"
fi

if [ "$USE_WRITER" = true ]; then
    CMD="$CMD --use_writer"
fi

echo "Running: $CMD"
echo ""

# Execute the command
eval $CMD
