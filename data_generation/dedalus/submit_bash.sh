#!/bin/bash


#SBATCH --time=03:30:00

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)


# python3 preprocess.py
python3 comparison.py