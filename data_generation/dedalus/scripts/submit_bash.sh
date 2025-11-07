#!/bin/bash


#SBATCH --time=00:30:00          # Increased time for processing 40k timesteps (0.3 hours)
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=16        # Use 16 CPUs for multiprocessing (adjust based on node availability)
#SBATCH --output=slurm-%j.out     # Explicit output file (job ID will be inserted)
#SBATCH --error=slurm-%j.err      # Explicit error file
# Note: GPU not needed for preprocessing, removed --gres=gpu:1


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

# Run Python with unbuffered output (-u) so output appears immediately
python3 -u compute_statistics.py