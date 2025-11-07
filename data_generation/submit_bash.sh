#!/bin/bash


#SBATCH --time=04:00:00          # Increased time for processing 40k timesteps (4 hours)
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=16        # Use 16 CPUs for multiprocessing (adjust based on node availability)
# Note: GPU not needed for preprocessing, removed --gres=gpu:1


module load pytorch/2.5.1-py312-cu122-mpi
source $HOME/.venvs/pytorch/bin/activate

# SLURM_CPUS_PER_TASK is automatically set by SLURM based on --cpus-per-task
echo "SLURM allocated $SLURM_CPUS_PER_TASK CPUs for this job"

python3 preprocess.py
