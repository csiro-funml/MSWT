import scipy.io
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import sys

import h5py
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
# todo: load all the data from the mat file in the folder:
# /data/large/pdearena/sw2d_pda/train, stack the needed variables and save it as a numpy array
def load_sw_data_split_and_save(datapath):
    data = np.load(os.path.join(datapath, 'sw2d_dataset.npz'))
    print(data)
    X_train, X_val, X_test, y_train, y_val, y_test = data['X_train'], data['X_val'], data['X_test'], data['Y_train'], data['Y_val'], data['Y_test']
    print("X_train shape: ", X_train.shape) # (N, C, H, W)  (28720, 2, 256, 128)  # 80 realizations of 360 steps
    print("X_val shape: ", X_val.shape) # (N, C, H, W)  # 10 realizations of 360 steps
    print("X_test shape: ", X_test.shape) # (N, C, H, W)  # 10 realizations of 360 steps
    print("y_train shape: ", y_train.shape)
    print("y_val shape: ", y_val.shape)
    print("y_test shape: ", y_test.shape)

    return X_test

if __name__ == '__main__':
    # X_test = load_sw_data_split_and_save('/datasets/work/oa-tcch/work/forXuesong')
    data_folder = '/scratch3/wan410/operator_learning_data/Dedalus/ShallowWater'
    X_test = np.load(os.path.join(data_folder, 'sw2d_test_dataset.npz'))['X_test']
    print("X_test shape: ", X_test.shape)
    
    # X_test (N*T, C, H, W) 
    T = 359
    C = 2
    H = 256
    W = 128
    
    # Reshape from (N*T, C, H, W) to (N, T, C, H, W)
    N = X_test.shape[0] // T
    assert X_test.shape[0] % T == 0, f"X_test.shape[0] ({X_test.shape[0]}) must be divisible by T ({T})"
    X_new = X_test.reshape(N, T, C, H, W)
    print(f"Reshaped X_test from {X_test.shape} to X_new shape: {X_new.shape}")
    
    # Plot x_new[0, t, 0, h, w] at every 10 time intervals
    save_folder = os.path.join(data_folder, 'sanity_plot')
    os.makedirs(save_folder, exist_ok=True)
    print(f"Saving plots to: {save_folder}")
    
    # Get the first trajectory, first channel
    trajectory_0_channel_0 = X_new[0, :, 0, :, :]  # (T, H, W)
    
    # Plot at every 10 time steps
    time_steps = list(range(0, T, 10))  # [0, 10, 20, 30, ...]
    print(f"Plotting at time steps: {time_steps}")
    
    for t in time_steps:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(trajectory_0_channel_0[t], cmap='RdBu_r', origin='lower')
        ax.set_title(f'Trajectory 0, Channel 0, Time Step {t}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Width (W)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Height (H)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(save_folder, f'trajectory0_channel0_t{t:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
        plt.close()
    
    print(f"Successfully saved {len(time_steps)} plots to {save_folder}")
