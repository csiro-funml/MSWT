#!/bin/bash


#SBATCH --time=00:30:00

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)


module load pytorch/2.5.1-py312-cu122-mpi

python3 preprocess.py