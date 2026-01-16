#!/bin/bash

# Script to submit multiple jobs with different test seeds
# Usage: ./submit_multiseed.sh [config_path] [test_ratio]
# Example: ./submit_multiseed.sh configs/HFS_periodic.yaml 0.25

# Default values
CONFIG_PATH="${1:-configs/HFS_periodic.yaml}"
TEST_RATIO="${2:-0.25}"

# Seeds to iterate over
# SEEDS=(42)
SEEDS=(42 43 44 45 46)

# Base directory (directory where submit_bash.sh is located)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Submitting jobs with multiple seeds"
echo "Config: $CONFIG_PATH"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="

# Loop over seeds
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "Creating and submitting job for seed $SEED..."
    
    # Create a temporary job script for this seed
    TEMP_SCRIPT="${BASE_DIR}/submit_bash_seed${SEED}.sh"
    
    # Create the script with all the SBATCH directives and setup
    cat > "$TEMP_SCRIPT" << EOF
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
module load ffmpeg
source \$HOME/.venvs/pytorch/bin/activate

# Print job info immediately (helps verify job started)
echo "=========================================="
echo "Job started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Working directory: \$(pwd)"
echo "SLURM allocated \$SLURM_CPUS_PER_TASK CPUs for this job"
echo "Test seed: ${SEED}"
echo "=========================================="

###############################################################NSE TORCHCFD ############################################################
# Training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python3 train_operator_AR_rell2_2d.py --config_path ${CONFIG_PATH} --test_ratio 0.25 --test_seed ${SEED}
# python3 test_operator_AR_rell2_2d.py --config_path ${CONFIG_PATH} --test_seed ${SEED}

python3 test_operator_AR_rell2_2d.py --config_path ${CONFIG_PATH} --test_seed ${SEED}
#################################################################################################################################
EOF

    # Make the script executable
    chmod +x "$TEMP_SCRIPT"
    
    # Submit the job
    JOB_ID=$(sbatch "$TEMP_SCRIPT" 2>&1 | awk '{print $4}')
    
    if [ $? -eq 0 ] && [ ! -z "$JOB_ID" ]; then
        echo "  ✓ Job submitted successfully! Job ID: $JOB_ID (seed $SEED)"
    else
        echo "  ✗ ERROR: Failed to submit job for seed $SEED"
        echo "  Error output: $JOB_ID"
    fi
    
    # Optionally remove the temporary script after submission
    # Uncomment the next line if you want to clean up immediately
    rm -f "$TEMP_SCRIPT"
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Temporary scripts are in: $BASE_DIR"
echo "To clean up temporary scripts, run: rm -f ${BASE_DIR}/submit_bash_seed*.sh"
echo "=========================================="
