import sys
import os
# Add parent directory to Python path to access utils and models
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import h5py
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader

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
DATASET_DICT[name]['downsample'] = (2, 2)
DATASET_DICT[name]['temporal_downsample'] = 4

class DedalusDataset2D(Dataset):
    def __init__(self, data_name, t_in=10, t_ar = 1, form='vorticity', normalize=False, train='train', downsample=None, temporal_downsample=None):
        ## /datasets/work/oa-tcch/work/forXuesong/data/realisation_0000/snapshots
        super().__init__()
        self.data_name = data_name
        self.data_path = DATASET_DICT[data_name]['data_path']
        self.n_size = DATASET_DICT[data_name]['%s_range'%train][1] - DATASET_DICT[data_name]['%s_range'%train][0]
        self.start_idx = DATASET_DICT[data_name]['%s_range'%train][0]     
        self.train = train
        self.temporal_downsample = DATASET_DICT[data_name]['temporal_downsample'] if temporal_downsample is None else temporal_downsample
        self.downsample = DATASET_DICT[data_name]['downsample'] if downsample is None else downsample
        self.res = (DATASET_DICT[data_name]['in_size'][0]//self.downsample[0], DATASET_DICT[data_name]['in_size'][1]//self.downsample[1])
        self.t_in = t_in
        self.t_out = t_ar
        self.form = 'vorticity'
        self.n_channels = 3 if self.form == 'vorticity' else 4
        self.norm_mean, self.norm_std = self.get_normalizer()
   
    def __getitem__(self, index):
        """
        input: (T_in, H, W, C) sample every temporal_downsample steps
        output: [T_out, H, W, C] sample every temporal_downsample steps
        """
        data = []
        start_idx = index + self.start_idx # (skip train/val)
       
        with h5py.File(self.data_path, 'r') as f: # (T, H, W,C) 
            for sample_idx in range(start_idx, start_idx + self.temporal_downsample * (self.t_in + self.t_out), self.temporal_downsample):
                timestep = np.array(f['scales/timestep'][sample_idx]) # (1,) 
                H, W = f['tasks/vorticity'][sample_idx].shape
                timestep_aug = np.tile(timestep, (H, W))               
                if self.form == 'vorticity':
                    vorticity = np.array(f['tasks/vorticity'][sample_idx]) # (H, W)
                    streamfunction = np.array(f['tasks/streamfunction'][sample_idx]) # (H, W)
                    data.append([vorticity, streamfunction, timestep_aug])
                else:
                    pressure = np.array(f['tasks/pressure'][sample_idx])
                    velocity_x = np.array(f['tasks/velocity'][sample_idx,0,...])
                    velocity_y = np.array(f['tasks/velocity'][sample_idx,1,...])
                    data.append([pressure, velocity_x, velocity_y, timestep_aug])
            data = torch.from_numpy(np.array(data)) # (T_in + T_out, C, H, W)
            # print("data shape", data.shape)
            if self.downsample != (1, 1):
                data = self.downsample_x(data, H//self.downsample[0])
            data = data.permute(2, 3, 0, 1) # (T, C, H, W) -> (H, W, T, C)
            x = data[..., :self.t_in, :]
            y = data[..., self.t_in:self.t_in + self.t_out, :]
            return x, y            

    def downsample_x(self, u, N):
        """
        Downsample a real-valued input using FFT
        Args:
            u: Input tensor of shape (T, C, H, W)
            N: Target size for downsampling
        Returns:
            Downsampled tensor of shape (T, C, N, N)
        """
        # Get original size
        T, C, H, W = u.shape
        
        # Compute FFT
        u_hat = torch.fft.rfft2(u, norm='forward')
        
        # Create frequency selection mask
        freqs_h = torch.fft.fftfreq(H, d=1/H)
        freqs_w = torch.fft.rfftfreq(W, d=1/W)
        
        # Select frequencies within [-N/2, N/2-1] range
        sel_h = torch.logical_and(freqs_h >= -N/2, freqs_h <= N/2-1)
        sel_w = torch.logical_and(freqs_w >= -N/2, freqs_w <= N/2-1)
        
        # Apply frequency selection
        u_hat_down = u_hat[:, :, sel_h][:, :, :, sel_w]
        
        # Compute inverse FFT
        u_down = torch.fft.irfft2(u_hat_down, s=(N, N), norm='forward')
        
        return u_down

    def get_normalizer(self):
        # use 100 samples from the training set to get the MIN-MAX normalizer
        print("getting the normalizer")
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
        
        # add timestep with 0 mean and 1 std 
        data_mean = np.concatenate([data_mean, np.zeros(1)], axis=-1)
        data_std = np.concatenate([data_std, np.ones(1)], axis=-1)
        data_mean = torch.from_numpy(data_mean)
        data_std = torch.from_numpy(data_std)
        return data_mean, data_std
    
    def denormalize_x(self, x):
        x = x * (self.norm_std.to(x.device) + 1e-6) +  self.norm_mean.to(x.device)
        return x

    def normalize_x(self, x):
        x = (x - self.norm_mean.to(x.device)) / (self.norm_std.to(x.device) + 1e-6)
        return x

    def __len__(self):
        return self.n_size - self.temporal_downsample * (self.t_in + self.t_out) 



if __name__ == '__main__':
    # Test single sample loading time
    print("Testing single sample loading time...")
    dataset = DedalusDataset2D(name, t_in=7, t_ar=1, form='vorticity', normalize=True, train='train')
    
    # Time single sample loading
    start_time = time.time()
    x, y = dataset[0]
    single_sample_time = time.time() - start_time
    print(f"Single sample loading time: {single_sample_time:.4f} seconds")
    print(f"Sample shapes: x={x.shape}, y={y.shape}")
    
    # Test batch loading time
    print("\nTesting batch loading time...")
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Warm up (first batch might be slower due to initialization)
        for i, (x_warm, y_warm) in enumerate(data_loader):
            if i == 0:
                print(f"Warm-up batch shapes: x={x_warm.shape}, y={y_warm.shape}")
            break
        
        # Time batch loading
        start_time = time.time()
        batch_count = 0
        for x_batch, y_batch in data_loader:
            batch_count += 1
            if batch_count >= 3:  # Time first 3 batches
                break
        batch_time = time.time() - start_time
        avg_batch_time = batch_time / batch_count
        
        print(f"Total time for {batch_count} batches: {batch_time:.4f} seconds")
        print(f"Average time per batch: {avg_batch_time:.4f} seconds")
        print(f"Time per sample in batch: {avg_batch_time/batch_size:.4f} seconds")
        
        # Test with multiple workers
        print(f"\nTesting with num_workers=2 for batch_size={batch_size}")
        data_loader_multi = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        start_time = time.time()
        batch_count = 0
        for x_batch, y_batch in data_loader_multi:
            batch_count += 1
            if batch_count >= 3:  # Time first 3 batches
                break
        multi_batch_time = time.time() - start_time
        avg_multi_batch_time = multi_batch_time / batch_count
        
        print(f"Total time for {batch_count} batches (multi-worker): {multi_batch_time:.4f} seconds")
        print(f"Average time per batch (multi-worker): {avg_multi_batch_time:.4f} seconds")
        print(f"Speedup: {avg_batch_time/avg_multi_batch_time:.2f}x")