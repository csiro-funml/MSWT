#!/bin/bash


#SBATCH --time=00:20:00           # Increased time for longer training with larger batches

#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32        # Increased CPUs for DataLoader workers (H100 can handle more)
#SBATCH --output=slurm-%j.out     # Explicit output file (job ID will be inserted)
#SBATCH --error=slurm-%j.err      # Explicit error file

module load pytorch/2.5.1-py312-cu122-mpi
source /datasets/work/oa-turb-arc/work/multiscale_wavelet_transformers_Xuesong_ICML2026/venv/bin/activate

# test MSWT on NS2D
python3 NS2D_ChaoticKolmogorovFlow/test_operator_AR_rell2_2d.py --config_path NS2D_ChaoticKolmogorovFlow/configs/new_dataset_direc/MSWT.yaml 

# test MSWT on SW2D
python3 SW2D_PDA/test_operator_AR_rell2_2d.py --config_path SW2D_PDA/configs/new_dataset_direc/MSWT_periodic.yaml 

# test MSWT on ERA5
python3 ERA5/2d_test_rel_l2.py --config_path ERA5/configs/new_dataset_direc/MSWT_sphere.yaml 