import sys
import os
# Add parent directory to Python path to access utils and models
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import h5py
import numpy as np
import torch
import time

DATASET_DICT = {}
DATASET_LIST = []

name = 'ns2d_dedalus'
DATASET_DICT[name] =  {'data_path': '/datasets/work/oa-tcch/work/forXuesong/new/realisation_0000/snapshots/snapshots_s1.h5', 
                          }
DATASET_DICT[name]['train_range'] = (2000, 7000)  # 5k samples (100-350 s)
DATASET_DICT[name]['test_range'] = (7500, 8500) # 1k samples (375s - 425s)
DATASET_DICT[name]['val_range'] = (7000, 7500) # 500 samples(350-375s)        
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 30   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 7     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 30
DATASET_DICT[name]['in_size'] = (256, 256)
DATASET_DICT[name]['n_channels'] = 3
# DATASET_DICT[name]['downsample'] = (2, 2)
DATASET_DICT[name]['downsample'] = (1, 1)
DATASET_DICT[name]['temporal_downsample'] = 4

class DedalusDataset2D():
    def __init__(self, data_name, t_in=10, t_ar = 1, form='vorticity', normalize=False, train='train', downsample=None, temporal_downsample=None):
        ## /datasets/work/oa-tcch/work/forXuesong/data/realisation_0000/snapshots
        self.data_name = data_name
        self.data_path = DATASET_DICT[data_name]['data_path']
        self.norm_mean, self.norm_std = self.get_normalizer()
   

    def get_normalizer(self):
        # use 100 samples from the training set to get the MIN-MAX normalizer
        print("getting the normalizer")
        time_start = time.time()
        data_norm = []
        with h5py.File(self.data_path, 'r') as f: # (T, H, W,C)
            for sample_idx in range(100):
                if self.form == 'vorticity':
                    vorticity = np.array(f['tasks/vorticity'][sample_idx])
                    streamfunction = np.array(f['tasks/streamfunction'][sample_idx])
                    data_norm.append([vorticity, streamfunction])
                else:
                    pressure = np.array(f['tasks/pressure'][sample_idx])
                    velocity_x = np.array(f['tasks/velocity'][sample_idx,0,...])
                    velocity_y = np.array(f['tasks/velocity'][sample_idx,1,...])
                    data_norm.append([pressure, velocity_x, velocity_y])
        
        data_norm = np.stack(data_norm) # (100, C, H, W)
        # print("data_norm shape", data_norm.shape)
        data_mean =np.min(data_norm, axis=(0, 2, 3)) # (C,)
        data_std = (np.max(data_norm, axis=(0, 2, 3)) - data_mean) # (C,)
        print("data_mean", data_mean, "data_std", data_std)
        time_end = time.time()
        print("time taken to get the normalizer", time_end - time_start)
        # add timestep with 0 mean and 1 std 
        data_mean = np.concatenate([data_mean, np.zeros(1)], axis=-1)
        data_std = np.concatenate([data_std, np.ones(1)], axis=-1)
        return data_mean, data_std


if __name__ == '__main__':
    # Test single sample loading time
    print("Testing single sample loading time...")
    dataset = DedalusDataset2D(name, t_in=7, t_ar=1, form='vorticity', normalize=True, train='train')