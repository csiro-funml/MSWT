#!/bin/bash


#SBATCH --time=00:10:00

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32        # Increased CPUs for DataLoader workers (H100 can handle more)
#SBATCH --output=slurm-%j.out     # Explicit output file (job ID will be inserted)
#SBATCH --error=slurm-%j.err      # Explicit error file

module load pytorch/2.5.1-py312-cu122-mpi
source $HOME/.venvs/pytorch/bin/activate


# python3 data_preprocessing.py
# python3 2d_train_rel_l2.py --config_path config/LUCIE.yaml
# python3 2d_train_rel_l2.py --config_path config/HFS_sphere.yaml
# python3 2d_train_rel_l2.py --config_path config/MSWT_sphere.yaml --seed 42

python3 2d_test_rel_l2.py --config_path config/MSWT_sphere.yaml --seed 45
# python3 2d_test_rel_l2.py --config_path config/HFS_sphere.yaml --seed 43
# python3 2d_test_rel_l2.py --config_path config/LUCIE.yaml --seed 43
# python3 load_era5_for_gadi.py