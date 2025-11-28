#!/bin/bash
# Model Comparison Ablation Study
# Tests different model architectures: FNO, HFS, etc.
# All models use the same base settings: modes/width/n_layers come from --model_size (small/medium/large), T_out=1

source "$(dirname "$0")/common_config.sh"

echo "=========================================="
echo "Model Comparison Ablation Study"
echo "=========================================="

# Models to test
# Note: Some models don't use modes/width/n_layers parameters
MODELS_LIST=("FNO" "HFS" "wavelet_transformer")

# Global size preset for all models (FNO/HFS/Wavelet) -> controls modes/width or target_params.
# Set to 'medium' to compare the medium HFS (~32-40M) preset the training script provides.
MODEL_SIZE="small"

FIXED_T_OUT=1

# Function to build training command for a specific model
build_model_train_cmd() {
    local model=$1
    local cmd="CUDA_VISIBLE_DEVICES=0 python3 -u NSE/train_AR_NO_Dedalus.py"
    cmd="$cmd --dataset $DATASET --model $model"
    
    # Add model-specific parameters
    # FNO/Wavelet/HFS pull architecture from --model_size via the training script presets
    cmd="$cmd --model_size $MODEL_SIZE"
    
    cmd="$cmd --T_in $T_IN --T_out $FIXED_T_OUT"
    cmd="$cmd --normalize $NORMALIZE --normalize_strategy $NORMALIZE_STRATEGY"
    cmd="$cmd --form $FORM"
    cmd="$cmd --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --num_workers $NUM_WORKERS --pin_memory --prefetch_factor $PREFETCH_FACTOR"
    cmd="$cmd --warmup_epochs 0"
    
    echo "$cmd"
}

# Function to build testing command for a specific model
build_model_test_cmd() {
    local model=$1
    local cmd="CUDA_VISIBLE_DEVICES=0 python3 -u NSE/test_AR_NO_Dedalus.py"
    cmd="$cmd --dataset $DATASET --model $model"
    
    # Add model-specific parameters
    # FNO/Wavelet/HFS pull architecture from --model_size via the training script presets
    cmd="$cmd --model_size $MODEL_SIZE"
    
    cmd="$cmd --T_in $T_IN --T_out $FIXED_T_OUT"
    cmd="$cmd --normalize $NORMALIZE --normalize_strategy $NORMALIZE_STRATEGY"
    cmd="$cmd --form $FORM"
    cmd="$cmd --batch_size $BATCH_SIZE --epochs $EPOCHS"
    cmd="$cmd --num_workers $NUM_WORKERS --pin_memory --prefetch_factor $PREFETCH_FACTOR"
    cmd="$cmd --warmup_epochs 0"
    cmd="$cmd --num_steps 30 --dataset_type long --save_type pth"
    
    echo "$cmd"
}

# Training commands for different models
echo ""
echo "=== TRAINING COMMANDS ==="
for model in "${MODELS_LIST[@]}"; do
    echo ""
    echo "# Training: model=$model (size=$MODEL_SIZE preset)"
    cmd=$(build_model_train_cmd $model)
    echo "$cmd"
done

# Testing commands for different models
echo ""
echo "=== TESTING COMMANDS ==="
for model in "${MODELS_LIST[@]}"; do
    echo ""
    echo "# Testing: model=$model (size=$MODEL_SIZE preset)"
    cmd=$(build_model_test_cmd $model)
    echo "$cmd"
done

echo ""
echo "=========================================="
echo "To run a specific experiment, uncomment the desired command above"
echo "=========================================="
