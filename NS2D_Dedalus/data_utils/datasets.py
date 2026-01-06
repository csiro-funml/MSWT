import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
# from .utils import get_grid3d, convert_ic, torch2dgrid
import h5py
import os
import glob
from numpy.lib.format import open_memmap

def load_save_dedalus_data(datapath='/datasets/work/oa-tcch/work/forXuesong/realisation_0000/snapshots/snapshots_s1',
                           save_path='/scratch3/wan410/operator_learning_data/Dedalus/Forcing_with_low_freq_energy',
                           out_name='dedalus_data_train.npy',
                           chunk_size=64):
    """
    Reconstruct Dedalus virtual dataset slices (snapshots_s1_p*.h5) into a single
    NumPy file using streaming writes to avoid loading everything into memory.

    Args:
        datapath: Path to snapshots_s1 (either directory containing p* files or
                  base path without the _p*.h5 suffix).
        save_path: Directory to write the .npy file.
        out_name: File name for the saved array.
        chunk_size: Number of timesteps processed per write chunk.
    Returns:
        Path to the saved npy file.
    """
    # Resolve slice files (p0, p1, ...) similar to preprocess_dedalus_to_shards
    if os.path.isdir(datapath):
        pattern = os.path.join(datapath, 'snapshots_s1_p*.h5')
    else:
        pattern = datapath.rstrip('.h5') + '_p*.h5'
    slice_files = sorted(glob.glob(pattern))
    if not slice_files:
        raise FileNotFoundError(f'No slice files found with pattern: {pattern}')

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, out_name)

    # Probe shape from the first slice
    with h5py.File(slice_files[0], 'r') as f0:
        T, H, W_partial = f0['tasks/vorticity'].shape
        has_forcing = 'tasks/forcing' in f0
    W_full = W_partial * len(slice_files)
    if not has_forcing:
        raise KeyError('tasks/forcing not found in slice files; expected forcing + vorticity.')

    # Preallocate .npy with header so it stays compatible with np.load memmap
    data_mem = open_memmap(out_path, mode='w+', dtype=np.float32, shape=(T, 3, H, W_full))
    print(f'Writing reconstructed data to {out_path}')
    print(f'Total timesteps: {T}, slices: {len(slice_files)}, shape: (T= {T}, C=3, H={H}, W={W_full})')

    # Keep file handles open for speed
    handles = [h5py.File(fp, 'r') for fp in slice_files]
    try:
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            # Collect slices for this chunk
            vort_chunks = []
            forc_chunks = []
            for h in handles:
                vort_chunks.append(h['tasks/vorticity'][start:end])              # (chunk, H, W_part)
                forc_chunks.append(h['tasks/forcing'][start:end])               # (chunk, 2, H, W_part)
            # Concatenate along width
            vorticity_full = np.concatenate(vort_chunks, axis=2)                 # (chunk, H, W)
            forcing_full = np.concatenate(forc_chunks, axis=3)                   # (chunk, 2, H, W)
            # Stack channels: forcing_x, forcing_y, vorticity
            chunk = np.concatenate(
                (forcing_full, np.expand_dims(vorticity_full, axis=1)),
                axis=1
            ).astype(np.float32, copy=False)                                     # (chunk, 3, H, W)
            data_mem[start:end] = chunk
            if (start // chunk_size) % 50 == 0:
                print(f'Processed {end}/{T} timesteps')
    finally:
        for h in handles:
            h.close()
    # Ensure data is flushed to disk
    data_mem.flush()
    print(f'Data saved to: {out_path}')
    return out_path




class NSLoader2D(Dataset):
    def __init__(self, datapath1,
                 nx, nt,
                 datapath2=None, sub=1, sub_t=1,
                 N=None, t_interval=1.0,
                 n_samples=None, offset=0,
                 train=True):
        '''
        Load data from npy and reshape to (N, X, Y, T)
        Args:
            datapath1: path to data
            nx:
            nt:
            datapath2: path to second part of data, default None
            sub:
            sub_t:
            N:
            t_interval:
            n_samples: number of trajectories to keep (defaults to N)
            offset: starting index for slicing
        '''
        self.S = nx // sub
        self.T = int(nt * t_interval) // sub_t + 1
        self.time_scale = t_interval
        self.train = train
        data1 = np.load(datapath1)
        data1 = torch.tensor(data1, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]

        if datapath2 is not None:
            data2 = np.load(datapath2)
            data2 = torch.tensor(data2, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
        if t_interval == 0.5:
            data1 = self.extract(data1)
            if datapath2 is not None:
                data2 = self.extract(data2)
        part1 = data1.permute(0, 2, 3, 1)
        if datapath2 is not None:
            part2 = data2.permute(0, 2, 3, 1)
            self.data = torch.cat((part1, part2), dim=0)
        else:
            self.data = part1
        total = self.data.shape[0]
        if offset >= total:
            raise ValueError(f'Offset {offset} exceeds dataset size {total}.')
        if n_samples is None:
            if N is None:
                n_samples = total
            else:
                n_samples = N
        start = max(0, offset)
        end = total if n_samples is None else min(total, start + n_samples)
        self.data = self.data[start:end] # (N, X, Y, T)
        self.num_samples = self.data.shape[0]
        self.max_time_index = self.data.shape[-1] - 1

    def normalize(self):
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.data = (self.data - self.mean) / self.std # average over spatial and temporal dimensions
        return self.data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        t = np.random.randint(0, self.max_time_index) if self.train else  0
        return sample[..., t], sample[..., t + 1]



if __name__ == '__main__':
    load_save_dedalus_data()
