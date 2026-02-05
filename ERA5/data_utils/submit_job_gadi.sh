#!/bin/bash


#PBS -l ncpus=8
#PBS -l mem=190GB
#PBS -l jobfs=200GB
#PBS -q normal
#PBS -P v14
#PBS -l walltime=00:30:00
#PBS -l wd

source /scratch/v14/xw5868/miniforge3/bin/activate
python3 load_era5_for_gadi.py