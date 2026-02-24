import scipy.io
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import h5py
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
# todo: load all the data from the mat file in the folder:
# /data/large/pdearena/sw2d_pda/train, stack the needed variables and save it as a numpy array
def load_ns2d_data_split_and_save(load_dir, save_dir):
    data = np.load(os.path.join(load_dir, 'kolmogorov_dataset.npz'))
    print(data)
    for key in data.keys():
        print(key)
    X_train, X_val, X_test, y_train, y_val, y_test = data['X_train'], data['X_val'], data['X_test'], data['Y_train'], data['Y_val'], data['Y_test']
    
    # only save the first 100 steps for prototyping
    
    print("X_train shape: ", X_train.shape) # (N, C, H, W)  (28720, 2, 256, 128)  # 80 realizations of 360 steps
    print("X_val shape: ", X_val.shape) # (N, C, H, W)  # 10 realizations of 360 steps
    print("X_test shape: ", X_test.shape) # (N, C, H, W)  # 10 realizations of 360 steps
    print("y_train shape: ", y_train.shape)
    print("y_val shape: ", y_val.shape)
    print("y_test shape: ", y_test.shape)
    
    # load realization name 
    train_realisation = data['train_realisations']
    times_train = data['times_train']
    val_realisation = data['val_realisations']
    times_val = data['times_val']
    test_realisation = data['test_realisations']
    times_test = data['times_test']

    # save the small prototype training/ val /test set in one npz file
    
    
    # np.savez(os.path.join(save_dir, 'sw2d_train_dataset.npz'), X_train=X_train, y_train=y_train, train_realisations=train_realisation, times_train=times_train)
    # np.savez(os.path.join(save_dir, 'sw2d_val_dataset.npz'), X_val=X_val, y_val=y_val, val_realisations=val_realisation, times_val=times_val)
    # np.savez(os.path.join(save_dir, 'sw2d_test_dataset.npz'), X_test=X_test, y_test=y_test, test_realisations=test_realisation, times_test=times_test)
    # print("Training/ val /test set saved to: ", os.path.join(save_dir, 'sw2d_train_dataset.npz'), os.path.join(save_dir, 'sw2d_val_dataset.npz'), os.path.join(save_dir, 'sw2d_test_dataset.npz'))
    # print("data shape: ", data.shape)
    return data


class NS2DLoader2D(Dataset):
    def __init__(self, datapath, state='train', train=True, normalizer_path=None, save_normalizer_path=None):
        '''
        Load data from npz files (sw2d_train_dataset.npz, sw2d_val_dataset.npz, sw2d_test_dataset.npz)
        Args:
            datapath: path to directory containing the npz files
            state: 'train', 'val', or 'test'
            train: if True, data is for training (random sampling), else deterministic
            normalizer_path: path to saved normalizer file (required for val/test, optional for train)
            save_normalizer_path: path to save normalizer after computing from training set
        '''
        self.train = train
        self.state = state
        
        # Load data from npz file
        npz_filename = f'sw2d_{state}_dataset.npz'
        npz_path = os.path.join(datapath, npz_filename)
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")
        
        data_dict = np.load(npz_path)
        # Keys match the state: X_train/y_train for train, X_val/y_val for val, X_test/y_test for test
        X_key = f'X_{state}'
        y_key = f'y_{state}'
        
        # Fallback: try to find any X/y keys if state-specific ones don't exist
        if X_key not in data_dict or y_key not in data_dict:
            keys = list(data_dict.keys())
            X_key = [k for k in keys if k.startswith('X')][0] if any(k.startswith('X') for k in keys) else None
            y_key = [k for k in keys if k.startswith('y')][0] if any(k.startswith('y') for k in keys) else None
            if X_key is None or y_key is None:
                raise KeyError(f"Could not find X and y keys in {npz_path}. Expected X_{state}/y_{state}. Available keys: {keys}")
        
        X_data = data_dict[X_key]
        y_data = data_dict[y_key]
        
        # X_data and y_data are (N, C, H, W) = (N, 2, 256, 128)
        # Stack X and y: (N, C, H, W) -> (N*2, C, H, W) if we want pairs, or just use X and y separately
        # For now, we'll use X as input and y as target
        self.X_data = torch.tensor(X_data, dtype=torch.float32)  # (N, C, H, W)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)  # (N, C, H, W)
        print("before rotation: ", self.X_data.shape, self.y_data.shape)
        # Rotate data counter-clockwise by 90 degrees: (N, 2, 256, 128) -> (N, 2, 128, 256)
        self.X_data = torch.rot90(self.X_data, k=1, dims=[-2, -1])  # Rotate last two dimensions
        self.y_data = torch.rot90(self.y_data, k=1, dims=[-2, -1])  # Rotate last two dimensions
        print("after rotation: ", self.X_data.shape, self.y_data.shape)

        self.S = (self.X_data.shape[-2], self.X_data.shape[-1]) # (H, W) = (128, 256)
        
        self.num_samples = self.X_data.shape[0]
        print(f"Loaded {state} dataset: {self.num_samples} samples, shape: {self.X_data.shape}")
        
        # Normalize the data
        self.normalize(normalizer_path, save_normalizer_path)
        print(f"Normalized data - mean shape: {self.mean.shape}, std shape: {self.std.shape}")

    def normalize(self, normalizer_path=None, save_normalizer_path=None):
        '''
        Normalize data with mean and std of shape (C, H, W) = (2, 128, 256)
        For training set: compute and save normalizer
        For val/test sets: load saved normalizer
        '''
        if normalizer_path is not None and os.path.exists(normalizer_path):
            # Load saved normalizer
            normalizer = torch.load(normalizer_path)
            self.mean = normalizer['mean']  # (C, H, W)
            self.std = normalizer['std']    # (C, H, W)
            print(f"Loaded normalizer from {normalizer_path}")
        elif self.state == 'train':
            # Compute normalizer from training data
            # Compute mean and std over all samples (N dimension), keeping (C, H, W)
            self.mean = self.X_data.mean(dim=0)  # (C, H, W)
            self.std = self.X_data.std(dim=0)    # (C, H, W)
            print(f"Computed normalizer from training data - mean shape: {self.mean.shape}, std shape: {self.std.shape}")
            
            # Save normalizer if path provided
            if save_normalizer_path is not None:
                os.makedirs(os.path.dirname(save_normalizer_path) if os.path.dirname(save_normalizer_path) else '.', exist_ok=True)
                torch.save({
                    'mean': self.mean,
                    'std': self.std
                }, save_normalizer_path)
                print(f"Saved normalizer to {save_normalizer_path}")
        else:
            raise ValueError(f"normalizer_path must be provided for {self.state} set")
        
        # Normalize the data: (N, C, H, W) with (C, H, W) mean/std
        self.X_data = (self.X_data - self.mean) / (self.std + 1e-8)
        self.y_data = (self.y_data - self.mean) / (self.std + 1e-8)
        
        # Permute data once here: (N, C, H, W) -> (N, H, W, C) for easier access in __getitem__
        self.X_data = self.X_data.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        self.y_data = self.y_data.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

         # swap the first and second channel (I want the fitst channel to be the vorticity)
        self.X_data = self.X_data[..., [1, 0]]
        self.y_data = self.y_data[..., [1, 0]]
        print(f"Permuted data to (N, H, W, C) - final shape: {self.X_data.shape}")
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        '''
        Returns input and target with shape (H, W, C)
        '''
        # Data is already in (N, H, W, C) format, so just return slices
        return self.X_data[idx], self.y_data[idx]  # (H, W, C), (H, W, C)

    
    def transform_rollout(self, T=359) :
        """
        reshape self.X_data and self.y_data from (N, H, W, C) to (N, T, H, W, C)
        """
        assert self.X_data.shape[0] % T == 0, "Number of samples must be divisible by T"
        n_traj = self.X_data.shape[0] // T
        
        self.X_data = rearrange(self.X_data,   '(n t) h w c -> n h w t c', t=T)
        self.y_data = rearrange(self.y_data,   '(n t) h w c -> n h w t c', t=T)
        # self.X_data = rearrange(self.X_data,   '(t n) h w c -> n h w t c', t=T)
        # self.y_data = rearrange(self.y_data,   '(t n) h w c -> n h w t c', t=T)
        self.num_samples = n_traj
        # self.num_samples = 1
        self.T = T
        print(f"Reshaped data to (N, T, H, W, C) - final shape: {self.X_data.shape}")
        return self.X_data, self.y_data




def verify_time_steps(folder_path, state='test'):
    # Load data from npz file
    npz_filename = f'sw2d_{state}_dataset.npz'
    npz_path = os.path.join(folder_path, npz_filename)
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")
    
    data_dict = np.load(npz_path)
    # Keys match the state: X_train/y_train for train, X_val/y_val for val, X_test/y_test for test
    X_key = f'X_{state}'
    y_key = f'y_{state}'
    print(data_dict.keys())
    X_data = data_dict[X_key]
    y_data = data_dict[y_key]
    realisation_names = data_dict['test_realisations']
    times = data_dict['times_test']
    print("realisation_names shape: ", realisation_names.shape)
    print("times shape: ", times.shape)
    print("realisation_names: ", realisation_names)
    print("times: ", times)
    return X_data, y_data



if __name__ == '__main__':
    load_ns2d_data_split_and_save(load_dir='/datasets/work/oa-tcch/work/forXuesong', save_dir='/scratch3/wan410/operator_learning_data/Dedalus/')
    # verify_time_steps(folder_path='/scratch3/wan410/operator_learning_data/Dedalus/ShallowWater', state='test')