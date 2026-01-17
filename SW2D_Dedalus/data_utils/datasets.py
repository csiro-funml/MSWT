import scipy.io
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import h5py
from tqdm import tqdm
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

    # save the training/ val /test set in separate npz files
    np.savez(os.path.join(datapath, 'sw2d_train_dataset.npz'), X_train=X_train, y_train=y_train)
    np.savez(os.path.join(datapath, 'sw2d_val_dataset.npz'), X_val=X_val, y_val=y_val)
    np.savez(os.path.join(datapath, 'sw2d_test_dataset.npz'), X_test=X_test, y_test=y_test)
    print("Training/ val /test set saved to: ", os.path.join(datapath, 'sw2d_train_dataset.npz'), os.path.join(datapath, 'sw2d_val_dataset.npz'), os.path.join(datapath, 'sw2d_test_dataset.npz'))
    # print("data shape: ", data.shape)
    return data


class SWLoader2D(Dataset):
    def __init__(self, datapath1,
                 nx, ny, nt,
                 datapath2=None, sub=1, sub_t=1,
                 N=None, t_interval=1.0,
                 n_samples=None, offset=0,
                 train=True,
                 normalizer_path=None):
        '''
        Load data from npy and reshape to (N, X, Y, T, C)
        Args:
            datapath1: path to data
            nx:
            ny:
            nt:
            nc:
            datapath2: path to second part of data, default None
            sub:
            sub_t:
            N:
            t_interval:
            n_samples: number of trajectories to keep (defaults to N)
            offset: starting index for slicing
        '''
        S1 = nx // sub
        S2 = ny // sub
        self.S = (S1, S2)
        self.T = (int(nt * t_interval) // sub_t + 1, 1)
        self.time_scale = t_interval
        self.train = train
        data1 = np.load(datapath1)
        data1 = torch.tensor(data1, dtype=torch.float)[..., ::sub, ::sub, ::sub_t, :]

        if datapath2 is not None:
            data2 = np.load(datapath2)
            data2 = torch.tensor(data2, dtype=torch.float)[..., ::sub, ::sub, ::sub_t, :]
        if t_interval == 0.5:
            data1 = self.extract(data1)
            if datapath2 is not None:
                data2 = self.extract(data2)

        if datapath2 is not None:
            self.data = torch.cat((data1, data2), dim=0)
        else:
            self.data = data1
        total = self.data.shape[0]
        if offset >= total:
            raise ValueError(f'Offset {offset} exceeds dataset size {total}.')
        n_samples = total

        start = max(0, offset)
        end = total if n_samples is None else min(total, start + n_samples)
        self.data = self.data[start:end] # (N, X, Y, T)
        self.num_samples = self.data.shape[0]
        self.max_time_index = self.data.shape[-2] - 1

        self.normalize(normalizer_path)
        print("normalized data", "mean shape: ", self.mean.shape, "std shape: ", self.std.shape)

    def normalize(self, normalizer_path=None):
        #todo: will load the normalizer from the saved path if normalizer_path is not None
        # if stored, the shape is (1, H, W) for different channels, need to concatenate them to (1, H, W, C)
        if normalizer_path is not None:
            normalizer = torch.load(normalizer_path)
            vars = ['vor', 'pres']
            mean = []
            std = []
            for var in vars:
                mean.append(normalizer[var]['mean'].squeeze()) # (1, H, W)
                std.append(normalizer[var]['std'].squeeze()) # (1, H, W)
            mean = torch.stack(mean, dim=0) # (C, H, W)
            std = torch.stack(std, dim=0) # (C, H, W)
            self.mean = mean.permute(1, 2, 0)[None, :, :, None, :] # (C, H, W) -> (1, H, W, 1, C)
            self.std = std.permute(1, 2, 0)[None, :, :, None, :] # (C, H, W) -> (1, H, W, 1, C)
            
        else:
            self.mean = self.data.mean()
            self.std = self.data.std()
        self.data = (self.data - self.mean) / self.std # average over spatial and temporal dimensions
        return self.data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        t = np.random.randint(0, self.max_time_index) if self.train else  0
        return sample[..., t, :], sample[..., t + 1, :]



if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     # state = 'train'
    #     state = 'val'
    #     folder = '/scratch3/wan410/operator_learning_data/pdearena/sw2d_pda/' + state
    # else:
    #     folder = 'pdearena/sw2d_pda/train'
    # data = load_save_sw_data(folder,max_files=4000, state=state)

    # loader = SWLoader2D(datapath1='pdearena/sw2d_pda/train/sw2d_pda_data_train.npy',
    #                     nx=96, ny=192, nt=87, nc=2,
    #                     sub=1, sub_t=1,
    #                     N=4000, t_interval=1.0,
    #                     n_samples=100, offset=0,
    #                     train=True)
    # print(data.shape)
    load_sw_data_split_and_save('/scratch3/wan410/operator_learning_data/Dedalus/ShallowWater')
    # data = SWLoader2D(datapath1='pdearena/sw2d_pda/train/sw2d_pda_data_train.npy',
    #                     nx=96, ny=192, nt=87, nc=2,
    #                     sub=1, sub_t=1,
    #                     N=4000, t_interval=1.0,
    #                     n_samples=100, offset=0,
    #                     train=True,
    #                     normalizer_path='pdearena/sw2d_pda/normstats.pt')
    # print(data.shape)