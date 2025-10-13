#!/usr/bin/env python3
"""
Optimized version of DedalusDataset2D with caching and batch operations
"""
import numpy as np
import torch
import h5py
import os
from torch.utils.data import Dataset
from utils.make_master_file import DATASET_DICT

class OptimizedDedalusDataset2D(Dataset):
    def __init__(self, data_name, t_in=10, t_ar=1, form='vorticity', normalize=False, 
                 train='train', downsample=None, temporal_downsample=None, 
                 cache_size_mb=500):
        """
        Optimized version with caching and batch operations
        
        Args:
            cache_size_mb: Maximum memory to use for caching (in MB)
        """
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
        self.form = form
        self.n_channels = 3 if self.form == 'vorticity' else 4
        self.cache_size_mb = cache_size_mb
        
        # Cache for frequently accessed data
        self._cache = {}
        self._cache_indices = set()
        self._file_handle = None
        
        # Pre-load metadata and setup cache
        self._setup_cache()
        self.norm_mean, self.norm_std = self.get_normalizer()
    
    def _setup_cache(self):
        """Setup caching system and pre-load metadata"""
        print("Setting up optimized cache...")
        
        # Estimate cache capacity
        with h5py.File(self.data_path, 'r') as f:
            sample_size = f['tasks/vorticity'][0].nbytes  # bytes per sample
            if self.form != 'vorticity':
                sample_size += f['tasks/pressure'][0].nbytes + f['tasks/velocity'][0].nbytes
            else:
                sample_size += f['tasks/streamfunction'][0].nbytes
            
            # Add timestep overhead
            sample_size += f['scales/timestep'][0].nbytes * f['tasks/vorticity'][0].shape[0] * f['tasks/vorticity'][0].shape[1]
            
            cache_capacity = (self.cache_size_mb * 1024 * 1024) // sample_size
            print(f"Cache capacity: {cache_capacity} samples ({self.cache_size_mb}MB)")
            
            # Pre-load frequently accessed data
            total_samples = f['tasks/vorticity'].shape[0]
            cache_indices = np.linspace(0, total_samples-1, min(cache_capacity, total_samples), dtype=int)
            
            self._preload_cache_indices(cache_indices)
    
    def _preload_cache_indices(self, indices):
        """Pre-load specific indices into cache"""
        print(f"Pre-loading {len(indices)} samples into cache...")
        
        with h5py.File(self.data_path, 'r') as f:
            for idx in indices:
                self._cache[idx] = self._load_single_sample(f, idx)
                self._cache_indices.add(idx)
        
        print(f"Cache loaded: {len(self._cache)} samples")
    
    def _load_single_sample(self, f, sample_idx):
        """Load a single sample from HDF5 file"""
        timestep = np.array(f['scales/timestep'][sample_idx])
        H, W = f['tasks/vorticity'][sample_idx].shape
        timestep_aug = np.tile(timestep, (H, W))
        
        if self.form == 'vorticity':
            vorticity = np.array(f['tasks/vorticity'][sample_idx])
            streamfunction = np.array(f['tasks/streamfunction'][sample_idx])
            return [vorticity, streamfunction, timestep_aug]
        else:
            pressure = np.array(f['tasks/pressure'][sample_idx])
            velocity_x = np.array(f['tasks/velocity'][sample_idx, 0, ...])
            velocity_y = np.array(f['tasks/velocity'][sample_idx, 1, ...])
            return [pressure, velocity_x, velocity_y, timestep_aug]
    
    def _load_sample_batch(self, indices):
        """Load multiple samples efficiently in one file operation"""
        data_batch = []
        
        with h5py.File(self.data_path, 'r') as f:
            for idx in indices:
                if idx in self._cache:
                    data_batch.append(self._cache[idx])
                else:
                    # Load and cache
                    sample_data = self._load_single_sample(f, idx)
                    self._cache[idx] = sample_data
                    self._cache_indices.add(idx)
                    data_batch.append(sample_data)
                    
                    # Simple cache eviction if needed
                    if len(self._cache) > len(self._cache_indices) * 1.2:
                        self._evict_cache()
        
        return data_batch
    
    def _evict_cache(self):
        """Simple LRU cache eviction"""
        if len(self._cache) > len(self._cache_indices) * 1.1:
            # Remove oldest entries (simple approach)
            keys_to_remove = list(self._cache.keys())[:len(self._cache)//4]
            for key in keys_to_remove:
                self._cache.pop(key, None)
    
    def __getitem__(self, index):
        """
        Optimized version with caching and batch operations
        """
        start_idx = index + self.start_idx
        sample_indices = list(range(start_idx, 
                                  start_idx + self.temporal_downsample * (self.t_in + self.t_out), 
                                  self.temporal_downsample))
        
        # Load batch of samples efficiently
        data_batch = self._load_sample_batch(sample_indices)
        
        # Convert to tensor
        data = torch.from_numpy(np.array(data_batch))  # (T_in + T_out, C, H, W)
        
        # Apply downsampling if needed
        if self.downsample != (1, 1):
            T, C, H, W = data.shape
            data = self.downsample_x(data, H//self.downsample[0])
        
        # Reshape to (H, W, T, C)
        data = data.permute(2, 3, 0, 1)  # (T, C, H, W) -> (H, W, T, C)
        
        # Split input and output
        x = data[..., :self.t_in, :]
        y = data[..., self.t_in:self.t_in + self.t_out, :]
        
        return x, y
    
    def downsample_x(self, u, N):
        """Same as original - FFT-based downsampling"""
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
        """Optimized normalizer using cached samples"""
        print("Getting optimized normalizer...")
        
        # Use cached samples if available, otherwise load a few
        if len(self._cache) > 10:
            sample_indices = list(self._cache.keys())[:10]
            data_norm = [self._cache[idx] for idx in sample_indices]
        else:
            # Load fresh samples
            data_norm = []
            with h5py.File(self.data_path, 'r') as f:
                for sample_idx in range(min(10, self.n_size)):
                    data_norm.append(self._load_single_sample(f, sample_idx))
        
        data_norm = np.stack(data_norm)  # (10, C, H, W)
        data_mean = np.min(data_norm, axis=(0, 2, 3))  # (C,)
        data_std = (np.max(data_norm, axis=(0, 2, 3)) - data_mean)  # (C,)
        
        print(f"Optimized normalizer - mean: {data_mean}, std: {data_std}")
        
        # Add timestep normalization
        data_mean = np.concatenate([data_mean, np.zeros(1)], axis=-1)
        data_std = np.concatenate([data_std, np.ones(1)], axis=-1)
        
        return torch.from_numpy(data_mean), torch.from_numpy(data_std)
    
    def denormalize_x(self, x):
        x = x * (self.norm_std.to(x.device) + 1e-6) + self.norm_mean.to(x.device)
        return x

    def normalize_x(self, x):
        x = (x - self.norm_mean.to(x.device)) / (self.norm_std.to(x.device) + 1e-6)
        return x

    def __len__(self):
        return self.n_size - self.temporal_downsample * (self.t_in + self.t_out)
    
    def __del__(self):
        """Cleanup cache when dataset is destroyed"""
        if hasattr(self, '_cache'):
            self._cache.clear()


class FullyCachedDedalusDataset2D(Dataset):
    """
    Fully cached version - loads entire dataset into memory
    Use only for small datasets or when you have lots of RAM
    """
    def __init__(self, data_name, t_in=10, t_ar=1, form='vorticity', normalize=False, 
                 train='train', downsample=None, temporal_downsample=None):
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
        self.form = form
        self.n_channels = 3 if self.form == 'vorticity' else 4
        
        # Load entire dataset into memory
        print("Loading entire dataset into memory...")
        self._load_full_dataset()
        self.norm_mean, self.norm_std = self.get_normalizer()
    
    def _load_full_dataset(self):
        """Load entire dataset into memory - use with caution!"""
        with h5py.File(self.data_path, 'r') as f:
            total_samples = f['tasks/vorticity'].shape[0]
            print(f"Loading {total_samples} samples into memory...")
            
            # Load all data at once
            if self.form == 'vorticity':
                self.vorticity_data = np.array(f['tasks/vorticity'])
                self.streamfunction_data = np.array(f['tasks/streamfunction'])
            else:
                self.pressure_data = np.array(f['tasks/pressure'])
                self.velocity_data = np.array(f['tasks/velocity'])
            
            self.timestep_data = np.array(f['scales/timestep'])
            
            print("Full dataset loaded into memory")
    
    def __getitem__(self, index):
        """Ultra-fast access from memory"""
        start_idx = index + self.start_idx
        sample_indices = list(range(start_idx, 
                                  start_idx + self.temporal_downsample * (self.t_in + self.t_out), 
                                  self.temporal_downsample))
        
        data_batch = []
        H, W = self.vorticity_data.shape[1], self.vorticity_data.shape[2]
        
        for sample_idx in sample_indices:
            timestep = self.timestep_data[sample_idx]
            timestep_aug = np.tile(timestep, (H, W))
            
            if self.form == 'vorticity':
                vorticity = self.vorticity_data[sample_idx]
                streamfunction = self.streamfunction_data[sample_idx]
                data_batch.append([vorticity, streamfunction, timestep_aug])
            else:
                pressure = self.pressure_data[sample_idx]
                velocity_x = self.velocity_data[sample_idx, 0, ...]
                velocity_y = self.velocity_data[sample_idx, 1, ...]
                data_batch.append([pressure, velocity_x, velocity_y, timestep_aug])
        
        data = torch.from_numpy(np.array(data_batch))
        
        # Apply downsampling if needed
        if self.downsample != (1, 1):
            T, C, H, W = data.shape
            data = self.downsample_x(data, H//self.downsample[0])
        
        data = data.permute(2, 3, 0, 1)
        x = data[..., :self.t_in, :]
        y = data[..., self.t_in:self.t_in + self.t_out, :]
        
        return x, y
    
    def downsample_x(self, u, N):
        """Same downsampling as original"""
        T, C, H, W = u.shape
        u_hat = torch.fft.rfft2(u, norm='forward')
        freqs_h = torch.fft.fftfreq(H, d=1/H)
        freqs_w = torch.fft.rfftfreq(W, d=1/W)
        sel_h = torch.logical_and(freqs_h >= -N/2, freqs_h <= N/2-1)
        sel_w = torch.logical_and(freqs_w >= -N/2, freqs_w <= N/2-1)
        u_hat_down = u_hat[:, :, sel_h][:, :, :, sel_w]
        u_down = torch.fft.irfft2(u_hat_down, s=(N, N), norm='forward')
        return u_down
    
    def get_normalizer(self):
        """Fast normalizer using in-memory data"""
        if self.form == 'vorticity':
            data_norm = np.stack([self.vorticity_data[:10], self.streamfunction_data[:10]], axis=1)
        else:
            data_norm = np.stack([
                self.pressure_data[:10], 
                self.velocity_data[:10, 0, ...], 
                self.velocity_data[:10, 1, ...]
            ], axis=1)
        
        data_mean = np.min(data_norm, axis=(0, 2, 3))
        data_std = (np.max(data_norm, axis=(0, 2, 3)) - data_mean)
        
        data_mean = np.concatenate([data_mean, np.zeros(1)], axis=-1)
        data_std = np.concatenate([data_std, np.ones(1)], axis=-1)
        
        return torch.from_numpy(data_mean), torch.from_numpy(data_std)
    
    def denormalize_x(self, x):
        x = x * (self.norm_std.to(x.device) + 1e-6) + self.norm_mean.to(x.device)
        return x

    def normalize_x(self, x):
        x = (x - self.norm_mean.to(x.device)) / (self.norm_std.to(x.device) + 1e-6)
        return x

    def __len__(self):
        return self.n_size - self.temporal_downsample * (self.t_in + self.t_out)
