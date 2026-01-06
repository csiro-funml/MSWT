import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
# from .utils import get_grid3d, convert_ic, torch2dgrid
import h5py
import os

def load_save_dedalus_data(datapath='/datasets/work/oa-tcch/work/forXuesong/realisation_0000/snapshots', save_path='/scratch3/wan410/operator_learning_data/Dedalus/Forcing_with_low_freq_energy'):
    file  = h5py.File(os.path.join(datapath, 'snapshots_s1.h5'), 'r')
    print("file loaded")
    pressure = np.expand_dims(np.array(file['tasks/pressure']), axis=1) # (N, 1, H, W)
    velocity = np.array(file['tasks/velocity']) # (N, 2, H, W)
    forcing = np.array(file['tasks/forcing']) # (N, 2, H, W)
    vorticity = np.expand_dims(np.array(file['tasks/vorticity']), axis=1) # (N, 1, H, W)
    
    data = np.concatenate((pressure, velocity, forcing, vorticity), axis=1) # (N, 5, H, W)
    print("data shape: ", data.shape)
    np.save(os.path.join(save_path, 'dedalus_data_train.npy'), data)
    print("data saved to: ", os.path.join(save_path, 'dedalus_data_train.npy'))
    return data




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