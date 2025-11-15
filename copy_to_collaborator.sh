#!/bin/bash

#SBATCH --time=001:00:00           # Increased time for longer training with larger batches

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=32        # Increased CPUs for DataLoader workers (H100 can handle more)
#SBATCH --output=slurm-%j.out     # Explicit output file (job ID will be inserted)
#SBATCH --error=slurm-%j.err      # Explicit error file

set -e  # Exit on error

# cp /scratch3/wan410/operator_learning_model/ns2d_dedalus_big_FNO_mod32_wid32_lay4_ntrain32006_formvelocity_lossfourier_logscaleTrue_warmup0/test_data_prediction_long.npz /datasets/work/oa-tcch/work/forMichael/test_data_prediction_long_spectral_reg.npz
cp /scratch3/wan410/operator_learning_model/ns2d_dedalus_big_FNO_mod32_wid32_lay4_ntrain32006_normalizer_zscore_form_velocity/test_data_prediction_long.npz /datasets/work/oa-tcch/work/forMichael/test_data_prediction_long.npz
echo "Copied prediction files to collaborator directory"

