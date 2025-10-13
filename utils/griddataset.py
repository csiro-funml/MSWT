#!/usr/bin/env python  
#-*- coding:utf-8 _*-
from re import X
import torch
import torch.nn.functional as F
import time
import numpy as np
import pickle
import os
import h5py
from functools import partial
from typing import Sequence
from einops import rearrange
# from sklearn.preprocessing import QuantileTransformer
import scipy.fft


from utils.make_master_file import DATASET_DICT
from utils.normalizer import init_normalizer, UnitTransformer, PointWiseUnitTransformer, MinMaxTransformer, TorchQuantileTransformer, IdentityTransformer
from torch.utils.data import Dataset
from utils.make_master_file import DATASET_DICT
from utils.utilities import downsample, resize



# get current directory
current_dir = os.getcwd()



class MixedTemporalDataset(Dataset):
    # _num_datasets = 0
    # _num_channels = 0
    def __init__(self, data_names, n_list = None, res = 128,t_in = 10, t_ar = 1, n_channels = None, normalize=False,train=True,data_weights=None, pad=0):
        '''
        Dataset class for training pretraining multiple datasets
        :param data_names: names of datasets, specified in make_master_file.py
        :param n_list: num of training samples per dataset, should corresponds to the order of data_names
        :param res: input resolution for the model, 64/128/256/512/1024
        :param t_in: input timesteps, 10 for default
        :param t_ar: steps for auto-regressive pretraining, 1 for default
        :param n_channels: number of channels for dataset, if None, it auto reads max number of channels from config file, should be specified for test dataset
        :param normalize: if normalize data,  reversible instance normalization is implemented in each model
        :param train: if it is train dataset or (in distribution) test dataset
        '''
        # set global configs
        # if train:
        #     MixedTemporalDataset._num_datasets = len(data_names)
        #     MixedTemporalDataset._num_channels = max([DATASET_DICT[name]['n_channels'] for name in data_names])
        self.data_names = data_names if isinstance(data_names, list) else [data_names]
        self.data_weights = data_weights if data_weights is not None else [1] * len(self.data_names)
        self.num_datasets = len(data_names)
        self.t_in = t_in
        self.t_ar = t_ar
        self.train = train
        self.res = res if pad else DATASET_DICT[self.data_names[0]]['in_size']
        self.pad = pad
        self.n_sizes = n_list if n_list is not None else [DATASET_DICT[name]['train_size'] if train else DATASET_DICT[name]['test_size'] for name in self.data_names]        
        self.weighted_sizes = [size * weight for size, weight in zip(self.n_sizes, self.data_weights)]
        # self.cumulative_sizes = np.cumsum(self.n_sizes)
        self.cumulative_sizes = np.cumsum(self.weighted_sizes)

        self.t_tests = [DATASET_DICT[name]['t_test'] for name in self.data_names]
        self.downsamples = [DATASET_DICT[name]['downsample'] for name in self.data_names]
        # self.n_channels = MixedTemporalDataset._num_channels
        self.n_channels = max([DATASET_DICT[name]['n_channels'] for name in self.data_names]) if n_channels is None else n_channels

        self.data_files = []
        for name in self.data_names:
            if DATASET_DICT[name]['scatter_storage']:
                def open_hdf5_file(path, idx):
                    return h5py.File(f'{path}/data_{idx}.hdf5', 'r')['data'][:]
                path = DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path']
                self.data_files.append(partial(open_hdf5_file, path))
                # if DATASET_DICT[name]['scatter_storage']:
                #     if train:
                #         self.data_files.append(lambda x, name=name:h5py.File(DATASET_DICT[name]['train_path'] + '/data_{}.hdf5'.format(x),'r')['data'])
                #     else:
                #         self.data_files.append(lambda x, name=name:h5py.File(DATASET_DICT[name]['test_path'] + '/data_{}.hdf5'.format(x),'r')['data'])
            else:
                self.data_files.append(h5py.File(DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path'], 'r'))
            # self.data_files = [h5py.File(DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path'], 'r') for name in self.data_names]


        self.normalize = normalize
        self.normalizers = []
        if normalize:
            print('Using normalizer for inputs')
            for data in self.data_files:
                self.normalizers.append(UnitTransformer(torch.from_numpy(data['data'][:500]).float()))    ### use 500 for normalization


    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, T, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1) # Cmax, H, W
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res),mode='bilinear').squeeze(0).permute(1, 2, 0)
        x = x.view(*x.shape[:2], T, C)
        x_new = torch.ones([*x.shape[:-1], self.n_channels])
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new

    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:2], 1, x.shape[-1])    ## target mask shape H,W,1,C
        kx, ky = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1]
        if kx ==0 or ky == 0:
            # print('warnings: target resolution < data resolution')
            kx = 1 if kx ==0 else kx
            ky = 1 if ky == 0 else ky
        msk[::kx, ::ky, :, :size_orig[-1]] = 1

        return msk

    def __len__(self):
        return self.cumulative_sizes[-1]




    def __getitem__(self, idx):
        '''
        Logic of getitem: first find which dataset idx is in, then reshape it to H,W,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))

        if dataset_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        data_idx //= self.data_weights[dataset_idx]
        # t_0 = time.time()
        sample = torch.from_numpy(self.data_files[dataset_idx](data_idx)[:] if callable(self.data_files[dataset_idx]) else self.data_files[dataset_idx]['data'][data_idx][:]).float()
        # sample = torch.from_numpy(np.array(self.data_files[dataset_idx]['data'][data_idx],dtype=np.float32))
        if sample.ndim == 3:    ### augment channel dim
            sample = sample.unsqueeze(-1)

        # print(time.time() - t_0)
        orig_size = list(sample.shape)
        orig_size[-1] = DATASET_DICT[self.data_names[dataset_idx]]['pred_channels'] if 'pred_channels' in DATASET_DICT[self.data_names[dataset_idx]].keys() else orig_size[-1]
        sample = self.pad_data(sample) if self.pad else sample # INTERPOLATION


        if self.train:  ## sample [0, t_in] and [t_in, t_in+ t_ar] for training ,trucated if too long
            start_idx = np.random.randint(max(sample.shape[-2] - (self.t_in + self.t_ar) + 1, 1))
            x, y = sample[..., start_idx: start_idx + self.t_in,:], sample[..., start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            # msk = msk[...,start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
            # print("x shape", x.shape, 'y shape', y.shape, "msk shape", msk.shape, )
        else: ## test datasets returns full trajectory
            start_idx = 0
            x, y = sample[..., start_idx:start_idx + self.t_in,:], sample[..., self.t_in:self.t_in + self.t_tests[dataset_idx],:]
            # msk = msk[..., self.t_in:self.t_in + self.t_tests[dataset_idx],:]
            msk = self.get_target_mask(sample, orig_size)

        if self.normalize:
            # x = self.normalizers[int(dataset_idx)].transform(x, inverse=False)
            x = (x.unsqueeze(0) - self.normalizers[int(dataset_idx)].mean[..., start_idx: start_idx + self.t_in,:]) / (self.normalizers[int(dataset_idx)].std[..., start_idx: start_idx + self.t_in,:] + 1e-6)
            x = x.squeeze()

        ### downsample
        if self.downsamples[dataset_idx] != (1, 1):
            x, y = x[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]], y[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]]

        idx_cls = torch.LongTensor([dataset_idx])   #TODO(hzk): now return relative idx in given datasets, finally we need global idx
        return x, y, msk, idx_cls



class MixedMaskedDataset(Dataset):
    # _num_datasets = 0
    # _num_channels = 0
    def __init__(self, data_names, n_list = None, res = 128,t_in = 10, t_ar = 1, n_channels = None, normalize=False,train=True,data_weights=None):
        '''
        Dataset class for training pretraining multiple datasets
        :param data_names: names of datasets, specified in make_master_file.py
        :param n_list: num of training samples per dataset, should corresponds to the order of data_names
        :param res: input resolution for the model, 64/128/256/512/1024
        :param t_in: input timesteps, 10 for default
        :param t_ar: steps for auto-regressive pretraining, 1 for default
        :param n_channels: number of channels for dataset, if None, it auto reads max number of channels from config file, should be specified for test dataset
        :param normalize: if normalize data,  reversible instance normalization is implemented in each model
        :param train: if it is train dataset or (in distribution) test dataset
        '''
        # set global configs
        # if train:
        #     MixedTemporalDataset._num_datasets = len(data_names)
        #     MixedTemporalDataset._num_channels = max([DATASET_DICT[name]['n_channels'] for name in data_names])
        self.data_names = data_names if isinstance(data_names, list) else [data_names]
        self.data_weights = data_weights if data_weights is not None else [1] * len(self.data_names)
        self.num_datasets = len(data_names)
        self.t_in = t_in
        self.t_ar = t_ar
        self.train = train
        self.res = res
        self.n_sizes = n_list if n_list is not None else [DATASET_DICT[name]['train_size'] if train else DATASET_DICT[name]['test_size'] for name in self.data_names]
        self.weighted_sizes = [size * weight for size, weight in zip(self.n_sizes, self.data_weights)]
        # self.cumulative_sizes = np.cumsum(self.n_sizes)
        self.cumulative_sizes = np.cumsum(self.weighted_sizes)

        self.t_tests = [DATASET_DICT[name]['t_test'] for name in self.data_names]
        self.downsamples = [DATASET_DICT[name]['downsample'] for name in self.data_names]
        # self.n_channels = MixedTemporalDataset._num_channels
        self.n_channels = max([DATASET_DICT[name]['n_channels'] for name in self.data_names]) if n_channels is None else n_channels

        self.data_files = []
        for name in self.data_names:
            if DATASET_DICT[name]['scatter_storage']:
                def open_hdf5_file(path, idx):
                    return h5py.File(f'{path}/data_{idx}.hdf5', 'r')['data'][:]
                path = DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path']
                self.data_files.append(partial(open_hdf5_file, path))
                # if DATASET_DICT[name]['scatter_storage']:
                #     if train:
                #         self.data_files.append(lambda x, name=name:h5py.File(DATASET_DICT[name]['train_path'] + '/data_{}.hdf5'.format(x),'r')['data'])
                #     else:
                #         self.data_files.append(lambda x, name=name:h5py.File(DATASET_DICT[name]['test_path'] + '/data_{}.hdf5'.format(x),'r')['data'])
            else:
                self.data_files.append(h5py.File(DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path'], 'r'))
            # self.data_files = [h5py.File(DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path'], 'r') for name in self.data_names]


        self.normalize = normalize
        self.normalizers = []
        if normalize:
            print('Using normalizer for inputs')
            for data in self.data_files:
                self.normalizers.append(UnitTransformer(torch.from_numpy(data['data'][:500]).float()))    ### use 500 for normalization


    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, T, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1) # Cmax, H, W
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res),mode='bilinear').squeeze(0).permute(1, 2, 0)
        x = x.view(*x.shape[:2], T, C)
        x_new = torch.ones([*x.shape[:-1], self.n_channels])    # use 1 for void padding
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new

    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:2], 1, x.shape[-1])    ## target mask shape H,W,1,C
        kx, ky = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1]
        if kx ==0 or ky == 0:
            # print('warnings: target resolution < data resolution')
            kx = 1 if kx ==0 else kx
            ky = 1 if ky == 0 else ky
        msk[::kx, ::ky, :, :size_orig[-1]] = 1

        return msk

    def get_masked_input(self, x):
        '''
        :param x:  single data, H, W, T, C
        :param size_orig:  original size of x
        :return: masked input, TODO: downsampling resolution
        '''
        x_new = x.clone()
        x_new[:,:,-1,:] = -1
        return x_new


    def __len__(self):
        return self.cumulative_sizes[-1]




    def __getitem__(self, idx):
        '''
        Logic of getitem: first find which dataset idx is in, then reshape it to H,W,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))

        if dataset_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        data_idx //= self.data_weights[dataset_idx]
        # t_0 = time.time()
        sample = torch.from_numpy(self.data_files[dataset_idx](data_idx)[:] if callable(self.data_files[dataset_idx]) else self.data_files[dataset_idx]['data'][data_idx][:]).float()
        # sample = torch.from_numpy(np.array(self.data_files[dataset_idx]['data'][data_idx],dtype=np.float32))
        if sample.ndim == 3:    ### augment channel dim
            sample = sample.unsqueeze(-1)

        # print(time.time() - t_0)
        orig_size = list(sample.shape)
        sample = self.pad_data(sample)


        if self.train:  ## sample [0, t_in] and [t_in, t_in+ t_ar] for training ,trucated if too long
            start_idx = np.random.randint(max(sample.shape[-2] - self.t_in + 1, 1))
            x = sample[..., start_idx: start_idx + self.t_in,:]
            # msk = msk[...,start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            x_msk = self.get_masked_input(x)
            # x_msk = x


            target_msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else: ## test datasets returns full trajectory
            x_msk, x = sample[...,:self.t_in,:], sample[..., self.t_in-1:self.t_in + self.t_tests[dataset_idx],:]
            target_msk = self.get_target_mask(sample, orig_size)
            x_msk = self.get_masked_input(x_msk)
        ### downsample
        if self.downsamples[dataset_idx] != (1, 1):
            x_msk, x = x_msk[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]], x[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]]

        idx_cls = torch.LongTensor([dataset_idx])   #TODO(hzk): now return relative idx in given datasets, finally we need global idx
        return x_msk, x, target_msk, idx_cls



class SteadyDataset2D(Dataset):
    def __init__(self, data_name, n_train=None, res=128, n_channels = None, normalize=False, train=True):
        '''
        :param data_name:
        :param n_train:
        :param res:
        :param t_in:
        :param t_ar:
        :param n_channels:
        :param normalize:
        :param train:
        '''
        self.data_name = data_name
        self.n_size = n_train if n_train is not None else DATASET_DICT[data_name]['train_size'] if train else DATASET_DICT[data_name]['test_size']
        self.train = train
        self.res = res
        self.n_channels = DATASET_DICT[data_name]['n_channels'] if n_channels is None else n_channels
        self.downsample = DATASET_DICT[data_name]['downsample']



        if DATASET_DICT[self.data_name]['scatter_storage']:
            def open_hdf5_file(path, idx, name):
                return h5py.File(f'{path}/data_{idx}.hdf5', 'r')[name][:]

            path = DATASET_DICT[self.data_name]['train_path'] if train else DATASET_DICT[self.data_name]['test_path']
            self.data_files = partial(open_hdf5_file, path)
        else:
            self.data_files = h5py.File(DATASET_DICT[self.data_name]['train_path'] if train else DATASET_DICT[self.data_name]['test_path'], 'r')

    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1)  # Cmax, H, W, L
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res), mode='bilinear').squeeze(0).permute(1, 2, 0).unsqueeze(-2)
        # x = resize(x, [self.res, self.res])
        x_new = torch.ones([*x.shape[:-1], self.n_channels])
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new


    def shuffle_channels(self, x, y):
        idx1, idx2 = torch.randperm(x.shape[-1])[:2]
        x[..., [idx1, idx2]] = x[..., [idx2, idx1]]
        y[...,[idx1, idx2]] = y[..., [idx2, idx1]]
        return x, y


    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:2], 1, x.shape[-1])  ## target mask shape H,W,1,C
        kx, ky = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1]
        if kx == 0 or ky == 0:
            # print('warnings: target resolution < data resolution')
            kx = 1 if kx == 0 else kx
            ky = 1 if ky == 0 else ky
        msk[::kx, ::ky, :, :size_orig[-1]] = 1

        return msk


    def __getitem__(self, idx):
        '''
        Logic of getitem:  reshape data to H,W,L,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''
        # t_0 = time.time()
        sample_x = torch.from_numpy(self.data_files(idx,name='x')[:] if callable(self.data_files) else self.data_files['x'][idx]).float()
        sample_y = torch.from_numpy(self.data_files(idx,name='y')[:] if callable(self.data_files) else self.data_files['y'][idx]).float()

        # sample = torch.from_numpy(np.array(self.data_files[dataset_idx]['data'][data_idx],dtype=np.float32))
        if sample_x.ndim == 2:    ### augment channel dim
            sample_x = sample_x.unsqueeze(-1)
            sample_y = sample_y.unsqueeze(-1)


        # sample_x, sample_y = self.shuffle_channels(sample_x, sample_y)

        # print(time.time() - t_0)
        orig_size = list(sample_x.shape)
        orig_size[-1] = DATASET_DICT[self.data_name]['pred_channels'] if 'pred_channels' in DATASET_DICT[self.data_name].keys() else orig_size[-1]
        x, y = self.pad_data(sample_x), self.pad_data(sample_y)


        if self.train:  ## sample [0, t_in] and [t_in, t_in+ t_ar] for training ,trucated if too long
            msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else: ## test datasets returns full trajectory
            msk = self.get_target_mask(x, orig_size)


        ### downsample
        if self.downsample != (1, 1, 1):
            x, y = x[::self.downsample[0],::self.downsample[1]], y[::self.downsample[0],::self.downsample[1]]

        # idx_cls = torch.LongTensor([dataset_idx])
        return x, y, msk

    def __len__(self):
        return self.n_size



class DiffusionDataset2D(Dataset):
    def __init__(self, data_name, n_train=None, t_in=10, t_ar = 1, n_channels = None, normalize=False, train='train'):
        '''

        :param data_name:
        :param n_train:
        :param res:
        :param t_in:
        :param t_ar:
        :param n_channels:
        :param normalize:
        :param train:
        '''
        self.data_name = data_name
        if  n_train is not None:
            self.n_size = 10 * n_train if (data_name == 'ns2d_pda' and train == 'train') else n_train # (use sliding windoer to augment data, mimicing the NO-diffusion paper)
        else:
            self.n_size = DATASET_DICT[data_name]['%s_size'%train]
        self.train = train == 'train'
        self.res = DATASET_DICT[self.data_name]['in_size']
        self.t_in = t_in
        self.t_ar = t_ar
        self.t_test = DATASET_DICT[data_name]['t_test']
        self.n_channels = DATASET_DICT[data_name]['n_channels'] if n_channels is None else n_channels
        self.downsample = DATASET_DICT[data_name]['downsample']


        if DATASET_DICT[self.data_name]['scatter_storage']:
            def open_hdf5_file(path, idx):
                return h5py.File(f'{path}/data_{idx}.hdf5', 'r')['data'][:]

            path = DATASET_DICT[self.data_name]['%s_path'%train]
            self.data_files = partial(open_hdf5_file, path)
        else:
            self.data_files = h5py.File(DATASET_DICT[self.data_name]['train_path'] if train else DATASET_DICT[self.data_name]['test_path'], 'r')

        
        # if normalize
        self.normalize = normalize
        print("normalizing state", normalize)
        if normalize:
            if 'normalizer_path' in DATASET_DICT[self.data_name].keys(): # shawllow water dataset, just load the parameters
                print("loading the normalizer from the saved path")
                normstat = torch.load(DATASET_DICT[self.data_name]['normalizer_path'])  # {"u" :{"mean":torch.tensor[], "std": torch.tensor},  "v", ... , "pres"}
                norm_mean = torch.cat([normstat["u"]["mean"].permute(1, 2, 0), normstat["v"]["mean"].permute(1, 2, 0), normstat["pres"]["mean"].unsqueeze(-1)], dim=-1)
                # print(norm_mean, norm_mean.shape)
                norm_std = torch.cat([normstat["u"]["std"].permute(1, 2, 0), normstat["v"]["std"].permute(1, 2, 0), normstat["pres"]["std"].unsqueeze(-1)], dim=-1)
                self.norm_mean = norm_mean
                self.norm_std = norm_std
                   
            else: # other datasets, manually save the data
                print("compute mean and std from the training set")
                data_norm = []
                for idx in range(100):
                    data_file_path = DATASET_DICT[self.data_name]['train_path'] # use the training set to normalize the data
                    data_file = h5py.File(f'{data_file_path}/data_{idx}.hdf5', 'r')['data'][:]
                    temp = torch.from_numpy(data_file).type(torch.float32)
                    # print("data input", temp.shape)
                    data_norm.append(temp)
                data_norm = torch.stack(data_norm)
                # print(data_norm.shape)

                # z-score normalization
                # self.norm_mean = torch.mean(data_norm, dim=(0, -2)) # the shape should be (H, W, C)
                # self.norm_std = torch.std(data_norm, dim=(0, -2)) 
                
                # min-max normalization following: https://github.com/vivekoommen/NeuralOperator_DiffusionModel/blob/main/case_1_kolmogorov/no_dm/fno/dm/dm_postprocess.ipynb
                # data-norm, shape: (B, H, W, T, C)
                inp_min = torch.amin(data_norm, dim=(0, 1, 2, 3)) # (C,) 
                inp_max = torch.amax(data_norm, dim=(0, 1, 2,3)) # (C, )
                
                self.norm_mean = inp_min # (C,)
                self.norm_std = (inp_max - inp_min) #(C, )
                print("min shape", inp_min.shape, self.norm_mean.numpy(), "max shape", inp_max.shape, self.norm_std.numpy())

    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, T, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1) # Cmax, H, W
        x = F.interpolate(x.unsqueeze(0), size=(self.res[0], self.res[1]),mode='bilinear').squeeze(0).permute(1, 2, 0)
        x = x.view(*x.shape[:2], T, C)
        x_new = torch.ones([*x.shape[:-1], self.n_channels])    # use 1 for void padding
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new

    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:2], 1, x.shape[-1])    ## target mask shape H,W,1,C
        kx, ky = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1]
        if kx ==0 or ky == 0:
            # print('warnings: target resolution < data resolution')
            kx = 1 if kx ==0 else kx
            ky = 1 if ky == 0 else ky
        msk[::kx, ::ky, :, :size_orig[-1]] = 1

        return msk

    def __getitem__(self, idx):
        '''
        Logic of getitem:  reshape data to H,W,L,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''
        # t_0 = time.time()
        if (self.data_name == 'ns2d_pda') and self.train:
            idx_raw = idx
            idx = idx_raw//10
            start_idx = idx_raw%10
        else:
            start_idx = None
        sample = torch.from_numpy(self.data_files(idx)[:] if callable(self.data_files) else self.data_files['data'][idx][:]).float() # (H, W, T_all, C)

        # just use three channels: (u, v, pres)
        if sample.shape[-1]>self.n_channels: # shallow water
            sample = sample[..., [0, 1, -1]]

        # sample = torch.from_numpy(np.array(self.data_files[dataset_idx]['data'][data_idx],dtype=np.float32))
        if sample.ndim == 3:    ### augment channel dim
            sample = sample.unsqueeze(-1)

        # now just using three channels to test out

        # print(time.time() - t_0)
        orig_size = list(sample.shape)
        # sample = self.pad_data(sample) if self.pad else sample # INTERPOLATION


        if self.train:  ## sample [0, t_in] and [t_in, t_in+ t_ar] for training ,trucated if too long
            start_idx = np.random.randint(max(sample.shape[-2] - (self.t_in + self.t_ar) + 1, 1)) if start_idx is None else start_idx
            x, y = sample[..., start_idx: start_idx + self.t_in,:], sample[..., start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            # msk = msk[...,start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else: ## test datasets returns full trajectory
            start_idx = 0
            x, y = sample[..., start_idx:start_idx + self.t_in,:], sample[..., self.t_in:self.t_in + self.t_test,:]
            # msk = msk[..., self.t_in:self.t_in + self.t_tests[dataset_idx],:]
            msk = self.get_target_mask(sample, orig_size)

        if self.normalize: #(isolate this part)
            # x = self.normalizers[int(dataset_idx)].transform(x, inverse=False)
            # print("pre norm x shape", x.shape) # (H, W, T, C)
            # x = (x - self.norm_mean.to(x.device)) / (self.norm_std.to(x.device))
            x = x
            # print("post norm x shape", x.shape) # (H, W, T, C)

        ### downsample
        if self.downsample != (1, 1):
            x, y = x[::self.downsample[0],::self.downsample[1]], y[::self.downsample[0],::self.downsample[1]]

        # idx_cls = torch.LongTensor([dataset_idx])
        x = x.permute(3, 2, 0, 1) # (H, W, T, C) -> (C, T, H, W)
        y = y.permute(3, 2, 0, 1)
        return x, y

    def denormalize(self, x):
        """
            X shape: (B, H, W, T, C)
        """
        x_denorm = x * (self.norm_std.to(x.device)) +  self.norm_mean.to(x.device)
        return x_denorm

    def normalize(self, x):
        # if self.normalize: #(isolate this part)
            # x = self.normalizers[int(dataset_idx)].transform(x, inverse=False)
            # print("pre norm x shape", x.shape) # (H, W, T, C)
        x = (x - self.norm_mean.to(x.device)) / (self.norm_std.to(x.device))
        return x

    def __len__(self):
        return self.n_size


class TemporalDataset2D(Dataset):
    def __init__(self, data_name, n_train=None, t_in=10, t_ar = 1, n_channels = None, normalize=False, train='train', downsample=None):
        '''

        :param data_name:
        :param n_train:
        :param res:
        :param t_in:
        :param t_ar:
        :param n_channels:
        :param normalize:
        :param train:
        '''
        self.data_name = data_name
        if  n_train is not None:
            self.n_size = n_train
        else:
            self.n_size = DATASET_DICT[data_name]['%s_size'%train]
        self.train = train == 'train'
        self.res = DATASET_DICT[self.data_name]['in_size']
        self.t_in = t_in
        self.t_ar = t_ar
        self.t_test = DATASET_DICT[data_name]['t_test']
        self.n_channels = DATASET_DICT[data_name]['n_channels'] if n_channels is None else n_channels
        self.downsample = DATASET_DICT[data_name]['downsample'] if downsample is None else downsample


        if DATASET_DICT[self.data_name]['scatter_storage']:
            def open_hdf5_file(path, idx):
                return h5py.File(f'{path}/data_{idx}.hdf5', 'r')['data'][:]

            path = DATASET_DICT[self.data_name]['%s_path'%train]
            self.data_files = partial(open_hdf5_file, path)
        else:
            if train == 'train': 
                self.data_files = h5py.File(DATASET_DICT[self.data_name]['train_path'], 'r')
            elif train == 'test':
                self.data_files = h5py.File(DATASET_DICT[self.data_name]['test_path'], 'r')
            elif train == 'var':
                self.data_files = h5py.File(DATASET_DICT[self.data_name]['val_path'], 'r')
            elif train == 'test_long':
                self.data_files = h5py.File(DATASET_DICT[self.data_name]['test_long_path'], 'r')
            else:
                raise ValueError(f"Invalid train type: {train}")

        
        # if normalize
        self.normalize = normalize
        print("normalizing state", normalize)
        if normalize:
            if 'normalizer_path' in DATASET_DICT[self.data_name].keys(): # shawllow water dataset, just load the parameters
                print("loading the normalizer from the saved path")
                normstat = torch.load(DATASET_DICT[self.data_name]['normalizer_path'])  # {"u" :{"mean":torch.tensor[], "std": torch.tensor},  "v", ... , "pres"}
                norm_mean = torch.cat([normstat["u"]["mean"].permute(1, 2, 0), normstat["v"]["mean"].permute(1, 2, 0), normstat["pres"]["mean"].unsqueeze(-1)], dim=-1)
                # print(norm_mean, norm_mean.shape)
                norm_std = torch.cat([normstat["u"]["std"].permute(1, 2, 0), normstat["v"]["std"].permute(1, 2, 0), normstat["pres"]["std"].unsqueeze(-1)], dim=-1)
                self.norm_mean = norm_mean
                self.norm_std = norm_std
                   
            else: # other datasets, manually save the data
                print("compute mean and std from the training set")
                data_norm = []
                for idx in range(100):
                    data_file_path = DATASET_DICT[self.data_name]['train_path'] # use the training set to normalize the data
                    data_file = h5py.File(f'{data_file_path}/data_{idx}.hdf5', 'r')['data'][:]
                    temp = torch.from_numpy(data_file).type(torch.float32)
                    # print("data input", temp.shape)
                    data_norm.append(temp)
                data_norm = torch.stack(data_norm)
                # print(data_norm.shape)

                # z-score normalization (ABANDONED, MIN-MAX IS GOOD ENOUGH)
                # self.norm_mean = torch.mean(data_norm, dim=(0, -2,), keepdim=True) # the shape should be (1, H, W, 1, C)
                # self.norm_std = torch.std(data_norm, dim=(0, -2), keepdim=True) 
                
                # min-max normalization following: https://github.com/vivekoommen/NeuralOperator_DiffusionModel/blob/main/case_1_kolmogorov/no_dm/fno/dm/dm_postprocess.ipynb
                # data-norm, shape: (B, H, W, T, C)
                inp_min = torch.amin(data_norm, dim=(0, 1, 2, 3)) # (C,) 
                inp_max = torch.amax(data_norm, dim=(0, 1, 2,3)) # (C, )
                
                self.norm_mean = inp_min # (C,)
                self.norm_std = (inp_max - inp_min) #(C, )
                print("min shape", inp_min.shape, self.norm_mean.numpy(), "max shape", inp_max.shape, self.norm_std.numpy())

    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, T, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1) # Cmax, H, W
        x = F.interpolate(x.unsqueeze(0), size=(self.res[0], self.res[1]),mode='bilinear').squeeze(0).permute(1, 2, 0)
        x = x.view(*x.shape[:2], T, C)
        x_new = torch.ones([*x.shape[:-1], self.n_channels])    # use 1 for void padding
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new

    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:2], 1, x.shape[-1])    ## target mask shape H,W,1,C
        kx, ky = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1]
        if kx ==0 or ky == 0:
            # print('warnings: target resolution < data resolution')
            kx = 1 if kx ==0 else kx
            ky = 1 if ky == 0 else ky
        msk[::kx, ::ky, :, :size_orig[-1]] = 1

        return msk

    def __getitem__(self, idx):
        '''
        Logic of getitem:  reshape data to H,W,L,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''
        # # t_0 = time.time()
        # if (self.data_name == 'ns2d_pda') and self.train:
        #     idx_raw = idx
        #     idx = idx_raw//10
        #     start_idx = idx_raw%10
        # else:
        start_idx = None
        sample = torch.from_numpy(self.data_files(idx)[:] if callable(self.data_files) else self.data_files['data'][idx][:]).float() # (H, W, T_all, C)

        # just use three channels: (u, v, pres)
        if sample.shape[-1]>self.n_channels: # shallow water
            sample = sample[..., [0, 1, -1]]

        # sample = torch.from_numpy(np.array(self.data_files[dataset_idx]['data'][data_idx],dtype=np.float32))
        if sample.ndim == 3:    ### augment channel dim
            sample = sample.unsqueeze(-1)

        # now just using three channels to test out

        # print(time.time() - t_0)
        orig_size = list(sample.shape)
        # sample = self.pad_data(sample) if self.pad else sample # INTERPOLATION


        if self.train:  ## sample [0, t_in] and [t_in, t_in+ t_ar] for training ,trucated if too long
            start_idx = np.random.randint(max(sample.shape[-2] - (self.t_in + self.t_ar) + 1, 1)) if start_idx is None else start_idx
            x, y = sample[..., start_idx: start_idx + self.t_in,:], sample[..., start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            # msk = msk[...,start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else: ## test datasets returns full trajectory
            start_idx = 0
            x, y = sample[..., start_idx:start_idx + self.t_in,:], sample[..., self.t_in:self.t_in + self.t_test,:]
            # msk = msk[..., self.t_in:self.t_in + self.t_tests[dataset_idx],:]
            msk = self.get_target_mask(sample, orig_size)

        if self.normalize: #(isolate this part)
            # x = self.normalizers[int(dataset_idx)].transform(x, inverse=False)
            # print("pre norm x shape", x.shape) # (H, W, T, C)
            # x = (x - self.norm_mean.to(x.device)) / (self.norm_std.to(x.device))
            x = x
            # print("post norm x shape", x.shape) # (H, W, T, C)

        ### downsample
        if self.downsample != (1, 1):
            # x, y = x[::self.downsample[0],::self.downsample[1]], y[::self.downsample[0],::self.downsample[1]]
            # reshape x from (H, W, T, C) to (T, C, H, W)
            # print("x shape", x.shape, "y shape", y.shape)
            x = x.permute(2, 3, 0, 1).contiguous()
            y = y.permute(2, 3, 0, 1).contiguous()
            x = self.downsample_x(x, self.downsample[0]) # (T, C, N, N)
            y = self.downsample_x(y, self.downsample[0])
            x = x.permute(2, 3, 0, 1).contiguous() # (N, N, T, C)
            y = y.permute(2, 3, 0, 1).contiguous() # (N, N, T, C)
            # print("x shape", x.shape, "y shape", y.shape)

        # # idx_cls = torch.LongTensor([dataset_idx])
        # x = x.permute(3, 2, 0, 1) # (H, W, T, C) -> (C, T, H, W)
        # y = y.permute(3, 2, 0, 1)
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
    

    def denormalize_x(self, x, meanstd=False):
        """
            X shape: (B, H, W, T, C)
        """
        if meanstd == False:
            if  len(self.norm_mean.shape)>1 and len(self.norm_mean.shape) != len(x.shape):
                self.norm_mean = self.norm_mean[None, :, :, None, :]
                self.norm_std = self.norm_std[None, :, :, None, :]
            x_denorm = x * (self.norm_std.to(x.device) + 1e-6) +  self.norm_mean.to(x.device)
        else: # scale mean the var
            x_denorm_mean, x_denorm_pred_std = torch.split(x, x.shape[-1]//2, dim=-1)
            x_denorm_pred_std = torch.exp(x_denorm_pred_std) # get non-negative std
            x_denorm_pred_std = x_denorm_pred_std.clamp(1e-6, 10) # clip the std
            x_denorm_mean = x_denorm_mean * (self.norm_std.to(x.device) + 1e-6) +  self.norm_mean.to(x.device) # (x' = x * std + mean)
            x_denorm_pred_std = x_denorm_pred_std * (self.norm_std.to(x.device)) 
            x_denorm = torch.cat([x_denorm_mean, x_denorm_pred_std], dim=-1)
        return x_denorm

    def normalize_x(self, x):
        # if self.normalize: #(isolate this part)
            # x = self.normalizers[int(dataset_idx)].transform(x, inverse=False)
            # print("pre norm x shape", x.shape) # (H, W, T, C)
        # print("x shape", x.shape)
        # print("norm mean shape", self.norm_mean.shape)
        # print("norm std shape", self.norm_std.shape)

        # x shape torch.Size([64, 96, 192, 10, 3])
        # norm mean shape torch.Size([96, 192, 3])
        # norm std shape torch.Size([96, 192, 3])
        # match the shape of x and norm_mean, norm_std
        if len(self.norm_mean.shape)>1 and len(self.norm_mean.shape) != len(x.shape):
            self.norm_mean = self.norm_mean[None, :, :, None, :]
            self.norm_std = self.norm_std[None, :, :, None, :]
        x = (x - self.norm_mean.to(x.device)) / (self.norm_std.to(x.device) + 1e-6)
        return x

    def __len__(self):
        return self.n_size


class CachedTemporalDataset2D(TemporalDataset2D):
    """
    Drop-in replacement for TemporalDataset2D with file caching.
    
    ONLY CHANGE: Cache HDF5 files in memory instead of opening/closing repeatedly.
    Everything else stays exactly the same.
    """
    
    def __init__(self, *args, cache_size=20, **kwargs):
        self.cache_size = cache_size
        super().__init__(*args, **kwargs)
        
        # Replace the data_files function with cached version if using scatter storage
        if DATASET_DICT[self.data_name]['scatter_storage']:
            self._file_cache = {}
            self._original_data_files = self.data_files
            self.data_files = self._cached_data_files
    
    def _cached_data_files(self, idx):
        """Cached version of file loading."""
        if idx not in self._file_cache:
            # Load file
            data = self._original_data_files(idx)
            
            # Add to cache
            self._file_cache[idx] = data
            
            # Simple LRU: remove oldest if cache is full
            if len(self._file_cache) > self.cache_size:
                oldest_key = next(iter(self._file_cache))
                del self._file_cache[oldest_key]
        
        return self._file_cache[idx]


class LocalTemporalDataset2D(Dataset):
    def __init__(self, data_name, n_train=None, t_in=10, t_ar = 1, n_channels = None, normalize=False, train=True, downsample=None):
        super().__init__()
        data = self.load_data(data_name, train)
        self.data = data

        self.data_name = data_name
        if  n_train is not None:
            self.n_size = n_train
        else:
            self.n_size = DATASET_DICT[data_name]['%s_size'%train]
        self.train = train == 'train'
        self.res = DATASET_DICT[self.data_name]['in_size']
        self.t_in = t_in
        self.t_ar = t_ar
        self.t_test = DATASET_DICT[data_name]['t_test']
        self.n_channels = DATASET_DICT[data_name]['n_channels'] if n_channels is None else n_channels
        self.downsample = DATASET_DICT[data_name]['downsample'] if downsample is None else downsample
        
        # if normalize
        self.normalize = normalize
        print("normalizing state", normalize)
        if normalize:
            if 'normalizer_path' in DATASET_DICT[self.data_name].keys(): # shawllow water dataset, just load the parameters
                print("loading the normalizer from the saved path")
                normstat = torch.load(DATASET_DICT[self.data_name]['normalizer_path'])  # {"u" :{"mean":torch.tensor[], "std": torch.tensor},  "v", ... , "pres"}
                norm_mean = torch.cat([normstat["u"]["mean"].permute(1, 2, 0), normstat["v"]["mean"].permute(1, 2, 0), normstat["pres"]["mean"].unsqueeze(-1)], dim=-1)
                # print(norm_mean, norm_mean.shape)
                norm_std = torch.cat([normstat["u"]["std"].permute(1, 2, 0), normstat["v"]["std"].permute(1, 2, 0), normstat["pres"]["std"].unsqueeze(-1)], dim=-1)
                self.norm_mean = norm_mean
                self.norm_std = norm_std
                   
            else: # other datasets, manually save the data
                print("compute mean and std from the training set")
                data_norm = self.data
                
                inp_min = torch.amin(data_norm, dim=(0, 1, 2, 3)) # (C,) 
                inp_max = torch.amax(data_norm, dim=(0, 1, 2,3)) # (C, )
                
                self.norm_mean = inp_min # (C,)
                self.norm_std = (inp_max - inp_min) #(C, )
                print("min shape", inp_min.shape, self.norm_mean.numpy(), "max shape", inp_max.shape, self.norm_std.numpy())

    
    def load_data(self, name, train_test_state):
        # load data from current directory/ns2d_pda/train/data_0.hdf5
        data_list = []
        # load all the data in the directory {current_dir}/pdearena/ns2d_pda/{train_test_state}/
        data_dir = f'{current_dir}/pdearena/{name}/{train_test_state}/'
        for file in os.listdir(data_dir):
            if file.endswith('.hdf5'):
                data = h5py.File(f'{data_dir}/{file}', 'r')['data'][:]
                data = torch.from_numpy(data).float()
                data_list.append(data)
        data = torch.stack(data_list)
        return data
    
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        start_idx = None
        sample = self.data[idx]

        # just use three channels: (u, v, pres)
        if sample.shape[-1]>self.n_channels: # shallow water
            sample = sample[..., [0, 1, -1]]

        # sample = torch.from_numpy(np.array(self.data_files[dataset_idx]['data'][data_idx],dtype=np.float32))
        if sample.ndim == 3:    ### augment channel dim
            sample = sample.unsqueeze(-1)

        # now just using three channels to test out

        # print(time.time() - t_0)
        orig_size = list(sample.shape)
        # sample = self.pad_data(sample) if self.pad else sample # INTERPOLATION


        if self.train:  ## sample [0, t_in] and [t_in, t_in+ t_ar] for training ,trucated if too long
            start_idx = np.random.randint(max(sample.shape[-2] - (self.t_in + self.t_ar) + 1, 1)) if start_idx is None else start_idx
            x, y = sample[..., start_idx: start_idx + self.t_in,:], sample[..., start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            # msk = msk[...,start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else: ## test datasets returns full trajectory
            start_idx = 0
            x, y = sample[..., start_idx:start_idx + self.t_in,:], sample[..., self.t_in:self.t_in + self.t_test,:]
            # msk = msk[..., self.t_in:self.t_in + self.t_tests[dataset_idx],:]
           
        
        return x, y
    
    def denormalize_x(self, x, meanstd=False):
        """
            X shape: (B, H, W, T, C)
        """
        if meanstd == False:
            if  len(self.norm_mean.shape)>1 and len(self.norm_mean.shape) != len(x.shape):
                self.norm_mean = self.norm_mean[None, :, :, None, :]
                self.norm_std = self.norm_std[None, :, :, None, :]
            x_denorm = x * (self.norm_std.to(x.device) + 1e-6) +  self.norm_mean.to(x.device)
        else: # scale mean the var
            x_denorm_mean, x_denorm_pred_std = torch.split(x, x.shape[-1]//2, dim=-1)
            x_denorm_pred_std = torch.exp(x_denorm_pred_std) # get non-negative std
            x_denorm_pred_std = x_denorm_pred_std.clamp(1e-6, 10) # clip the std
            x_denorm_mean = x_denorm_mean * (self.norm_std.to(x.device) + 1e-6) +  self.norm_mean.to(x.device) # (x' = x * std + mean)
            x_denorm_pred_std = x_denorm_pred_std * (self.norm_std.to(x.device)) 
            x_denorm = torch.cat([x_denorm_mean, x_denorm_pred_std], dim=-1)
        return x_denorm

    def normalize_x(self, x):
        # if self.normalize: #(isolate this part)
            # x = self.normalizers[int(dataset_idx)].transform(x, inverse=False)
            # print("pre norm x shape", x.shape) # (H, W, T, C)
        # print("x shape", x.shape)
        # print("norm mean shape", self.norm_mean.shape)
        # print("norm std shape", self.norm_std.shape)

        # x shape torch.Size([64, 96, 192, 10, 3])
        # norm mean shape torch.Size([96, 192, 3])
        # norm std shape torch.Size([96, 192, 3])
        # match the shape of x and norm_mean, norm_std
        if len(self.norm_mean.shape)>1 and len(self.norm_mean.shape) != len(x.shape):
            self.norm_mean = self.norm_mean[None, :, :, None, :]
            self.norm_std = self.norm_std[None, :, :, None, :]
        x = (x - self.norm_mean.to(x.device)) / (self.norm_std.to(x.device) + 1e-6)
        return x


class DedalusDataset2D(Dataset):
    def __init__(self, data_name, n_train=None, t_in=10, t_ar = 1, form='vorticity', normalize=False, train='train', downsample=None, temporal_downsample=1):
        ## /datasets/work/oa-tcch/work/forXuesong/data/realisation_0000/snapshots
        super().__init__()
        self.data_name = data_name
        self.data_path = DATASET_DICT[data_name]['data_path']
        self.n_size = DATASET_DICT[data_name]['%s_range'%train][1] - DATASET_DICT[data_name]['%s_range'%train][0]
        self.start_idx = DATASET_DICT[data_name]['%s_range'%train][0]     
        self.train = train
        self.temporal_downsample = temporal_downsample
        self.downsample = downsample
        self.t_in = t_in
        self.t_out = t_ar
        self.form = 'vorticity'
        self.n_channels = 2 if self.form == 'vorticity' else 3
        
        # self.norm_mean, self.norm_std = self.get_normalizer()
   
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
            x = data[:self.t_in, ...]
            y = data[self.t_in:self.t_in + self.t_out, ...]
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
        return data_mean, data_std
    
    def __len__(self):
        return self.n_size - self.temporal_downsample * (self.t_in + self.t_out) 

class CNO_NavierStokes2D(Dataset):
    def __init__(self, data_name, which="training", nf=0, n_train = 750, s=64, in_dist = True):
        
        self.s = s
        self.in_dist = in_dist
        #The file:
        
        if in_dist:
            if self.s==64:
                self.file_data = DATASET_DICT[data_name]['data_path']+"/NavierStokes_64x64_IN.h5" #In-distribution file 64x64               
            else:
                self.file_data = DATASET_DICT[data_name]['data_path']+"/NavierStokes_128x128_IN.h5"   #In-distribution file 128x128
        else:
            self.file_data = DATASET_DICT[data_name]['data_path']+"/NavierStokes_128x128_OUT.h5"  #Out-of_-distribution file 128x128
        
        self.reader = h5py.File(self.file_data, 'r') 
        self.N_max = 1024
        self.n_channels = DATASET_DICT[data_name]['n_channels']
        self.n_val  = 128
        self.n_test = 128
        self.min_data = 1.4307903051376343
        self.max_data = -1.4307903051376343
        self.min_model = 2.0603253841400146
        self.max_model= -2.0383243560791016
        
        if which == "train":
            self.n_size = n_train
            self.start = 0
        elif which == "validation":
            self.n_size = self.n_val
            self.start = self.N_max - self.n_val - self.n_test
        elif which == "test":
            self.n_size = self.n_test
            self.start = self.N_max  - self.n_test
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.n_size
    


    def samples_fft(self, u):
        return scipy.fft.fft2(u, norm='forward', workers=-1)

    def samples_ifft(self, u_hat):
        return scipy.fft.ifft2(u_hat, norm='forward', workers=-1).real
    
    
    def downsample(self, u, N):
        N_old = u.shape[-2]
        freqs = scipy.fft.fftfreq(N_old, d=1/N_old)
        sel = np.logical_and(freqs >= -N/2, freqs <= N/2-1)
        u_hat = self.samples_fft(u)
        u_hat_down = u_hat[:,:,sel,:][:,:,:,sel]
        u_down = self.samples_ifft(u_hat_down)
        return u_down

    def __getitem__(self, index):
        
        if self.s == 64 and self.in_dist:
            inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
            labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        else:
            
            inputs = self.reader['Sample_' + str(index + self.start)]["input"][:].reshape(1,1,self.s, self.s)
            labels = self.reader['Sample_' + str(index + self.start)]["output"][:].reshape(1,1, self.s, self.s)
            
            if self.s<128:
                inputs = self.downsample(inputs, self.s).reshape(1, self.s, self.s)
                labels = self.downsample(labels, self.s).reshape(1, self.s, self.s)
            else:
                inputs = inputs.reshape(1, 128, 128)
                labels = labels.reshape(1, 128, 128)
            
            inputs = torch.from_numpy(inputs).type(torch.float32)
            labels = torch.from_numpy(labels).type(torch.float32)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)
        
        
        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid
    

#
# def load_dataset(path):
#     '''
#     Auxiliary function for reading dataset
#     :param path:
#     :return:
#     '''
#     if path.endswith('.pkl'):
#         data = pickle.load(open(path, 'rb'))
#     elif path.endswith('.npy') or path.endswith('.npz'):
#         fp = np.load(path)
#         x = fp['x']
#         y = fp['y']
#         theta = None if fp['theta'].ndim == 0 else fp['theta']
#         data = {'x': x, 'y': y, 'theta': theta}
#     elif path.endswith('.hdf5'):
#         with h5py.File(path, 'r') as fp:
#             x = np.array(fp['x'],dtype=np.float32)
#             y = np.array(fp['y'],dtype=np.float32)
#             theta = None if fp['theta'].ndim == 0 else np.array(fp['theta'],dtype=np.float32)
#             data = {'x': x, 'y': y, 'theta': theta}
#     else:
#         raise ValueError
#     return data


# class GridDataset(Dataset):
#     def __init__(self, name, data=None, data_index=None, downsample_x=(0,), downsample_y=(0,)):
#         super(GridDataset, self).__init__()
#
#         if name not in DATASET_DICT.keys():
#             raise NotImplementedError
#
#         self.meta_info = DATASET_DICT[name]
#         self.downsample_x = downsample_x if downsample_x[0] else self.meta_info['default_downsample_x']
#         self.downsample_y = downsample_y if downsample_y[0] else self.meta_info['default_downsample_y']
#         self.scattered_storage =  ('scatter_stored' in self.meta_info.keys()) and self.meta_info['scatter_stored']
#         self.enable_grid = False
#
#         ### process dataset, initialize attributes
#         if self.scattered_storage:
#             self.data_index = list(data_index)
#             self.path_str = self.meta_info['path']
#
#             ### get shapes
#             x0, y0, theta0 = self.__getitem__(0)
#
#             self.gridsize_x = x0.shape[:-2] if self.meta_info['temporal'] else x0.shape[:-1]
#             self.gridsize_y = y0.shape[:-2] if self.meta_info['temporal'] else y0.shape[:-1]
#
#
#
#         else:
#             if data is None:
#                 data = load_dataset(self.meta_info['path'])
#
#             self.x, self.y = torch.from_numpy(data['x']), torch.from_numpy(data['y'])
#             self.theta = None if data['theta'] == None else torch.from_numpy(data['theta'])
#
#             #### downsample
#             self.x = self.__downsample(self.downsample_x, attr_name='x')
#             self.y = self.__downsample(self.downsample_y, attr_name='y')
#
#             self.gridsize_x = self.x.shape[1:-2] if self.meta_info['temporal'] else self.x.shape[1:-1]
#             self.gridsize_y = self.y.shape[1:-2] if self.meta_info['temporal'] else self.y.shape[1:-1]
#
#
#
#
#
#     def __len__(self):
#         if self.scattered_storage:
#             return len(self.data_index)
#         else:
#             return self.x.shape[0]
#
#     def __getitem__(self, idx):
#         if self.scattered_storage:
#             data = np.load(os.path.join(self.path_str,'data_{}.npz'.format(self.data_index[idx])))
#             x, y = torch.from_numpy(data['x']).unsqueeze(0), torch.from_numpy(data['y']).unsqueeze(0)
#             if hasattr(self, 'x_normalizer'):
#                 x, y = self.x_normalizer.transform(x, inverse=False), self.y_normalizer.transform(y, inverse=False)
#             x, y = self.__downsample(self.downsample_x, data=x), self.__downsample(self.downsample_y, data=y)
#             if self.enable_grid:
#                 x = self.auto_load_grid(data=x)
#             if self.meta_info['theta_dim'] == 0:
#                 theta = torch.zeros([])
#             else:
#                 theta = self.theta_normalizer.transform(torch.from_numpy(data['theta']).unsqueeze(0),inverse=False).squeeze(0)
#             return x.squeeze(0), y.squeeze(0), theta
#         else:
#             if self.theta is None:
#                 return self.x[idx], self.y[idx], torch.zeros([])
#             else:
#                 return self.x[idx], self.y[idx], self.theta[idx]
#
#
#
#
#     #### downscale dataset, support up to 4 dim, must pass either attr_name or data
#     def __downsample(self, downsample, data=None, attr_name=None):
#         if data is None:
#             if attr_name is not None:
#                 data = getattr(self, attr_name)
#             else:
#                 raise ValueError
#         downsample = downsample * self.meta_info['space_dim'] if isinstance(downsample, list) and len(downsample)==1 else downsample
#         if self.meta_info['space_dim'] == 1:
#             if isinstance(downsample, int):
#                 data = data[:,::downsample]
#             else:
#                 data = data[:,::downsample[0]]
#         elif self.meta_info['space_dim'] == 2:
#             if isinstance(downsample, int):
#                 data = data[:,::downsample, ::downsample]
#             else:
#                 data = data[:,::downsample[0],::downsample[1]]
#         elif self.meta_info['space_dim'] == 3:
#             if isinstance(downsample, int):
#                 data = data[:, ::downsample, ::downsample,:: downsample]
#             else:
#                 data = data[:, ::downsample[0], ::downsample[1], ::downsample[2]]
#         elif self.meta_info['space_dim'] == 4:
#             if isinstance(downsample, int):
#                 data = data[:, ::downsample, ::downsample, :: downsample, :: downsample]
#             else:
#                 data = data[:, ::downsample[0], ::downsample[1], ::downsample[2], ::downsample[3]]
#         else:
#             raise ValueError
#
#
#         if attr_name=='x':
#             self.x = data
#         elif attr_name == 'y':
#             self.y = data
#
#         return data
#
#     def get_normalizer(self, type):
#
#         # restore from file
#         if self.scattered_storage:
#             normalizer_data = np.load(os.path.join(self.path_str, 'normalizer_data.npz'))
#             if type == 'unit':
#                 x1, x2, y1, y2, t1, t2 = normalizer_data['unit_mean_x'], normalizer_data['unit_std_x'], normalizer_data['unit_mean_y'], normalizer_data['unit_std_y'], normalizer_data['unit_mean_theta'], normalizer_data['unit_std_theta']
#             elif type == 'pointunit':
#                 x1, x2, y1, y2, t1, t2 = normalizer_data['pointunit_mean_x'], normalizer_data['pointunit_std_x'], normalizer_data['pointunit_mean_y'], normalizer_data['pointunit_std_y'], normalizer_data['pointunit_mean_theta'], normalizer_data['pointunit_std_theta']
#             elif type == 'minmax':
#                 x1, x2, y1, y2, t1, t2 = normalizer_data['minmax_min_x'], normalizer_data['minmax_max_x'], normalizer_data['minmax_min_y'], normalizer_data['minmax_max_y'], normalizer_data['minmax_min_theta'], normalizer_data['minmax_max_theta']
#             else:
#                 x1, x2, y1, y2, t1, t2 = None, None, None, None, None, None
#             self.x_normalizer, self.y_normalizer = init_normalizer(type, x1, x2, eps=1e-7), init_normalizer(type, y1, y2, eps=1e-7)
#             self.theta_normalizer = init_normalizer(type, t1, t2, eps=1e-7) if self.meta_info['theta_dim'] else None
#
#
#         else:
#             if type in ['unit', 'pointunit','minmax','none']:
#                 if type == 'unit':
#                     normalizer = UnitTransformer
#                 elif type == 'pointunit':
#                     normalizer = partial(PointWiseUnitTransformer, temporal=self.meta_info['temporal'])
#                 elif type == 'minmax':
#                     normalizer = MinMaxTransformer
#                 else:
#                     normalizer = IdentityTransformer
#
#
#                 self.x_normalizer = normalizer(self.x, eps=1e-7)
#                 self.y_normalizer = normalizer(self.y, eps=1e-7)
#                 self.theta_normalizer = None if self.theta is None else normalizer(self.theta, eps=1e-7)
#             # elif type == 'quantile':
#             #
#             #     x_normalizer_numpy = QuantileTransformer(output_distribution='normal')
#             #     x_normalizer_numpy = x_normalizer_numpy.fit(self.x.reshape(-1, self.x.shape[-1]))
#             #     x_normalizer = TorchQuantileTransformer(x_normalizer_numpy.output_distribution, x_normalizer_numpy.references_, x_normalizer_numpy.quantiles_)
#             #
#             #     y_normalizer_numpy = QuantileTransformer(output_distribution='normal')
#             #     y_normalizer_numpy = y_normalizer_numpy.fit(self.y.reshape(-1, self.x.shape[-1]))
#             #     y_normalizer = TorchQuantileTransformer(y_normalizer_numpy.output_distribution, y_normalizer_numpy.references_, y_normalizer_numpy.quantiles_)
#             #
#             #     if self.theta is not None:
#             #         theta_normalizer_numpy = QuantileTransformer(output_distribution='normal')
#             #         theta_normalizer_numpy = theta_normalizer_numpy.fit(self.theta.reshape(-1, self.theta.shape[-1]))
#             #         theta_normalizer = TorchQuantileTransformer(theta_normalizer_numpy.output_distribution, theta_normalizer_numpy.references_, theta_normalizer_numpy.quantiles_)
#             #     else:
#             #         theta_normalizer = None
#             else:
#                 raise NotImplementedError
#
#         return self.x_normalizer, self.y_normalizer, self.theta_normalizer
#
#     def apply_normalizer(self, x_normalizer=None, y_normalizer=None, theta_normalizer=None):
#         if x_normalizer is not None:
#             self.x_normalizer = x_normalizer
#             if not self.scattered_storage:
#                 self.x = x_normalizer.transform(self.x, inverse=False)
#         if y_normalizer is not None:
#             self.y_normalizer = y_normalizer
#             if not self.scattered_storage:
#                 self.y = y_normalizer.transform(self.y, inverse=False)
#         if theta_normalizer is not None:
#             self.theta_normalizer = theta_normalizer
#             if not self.scattered_storage:
#                 self.theta = theta_normalizer.transform(self.theta, inverse=False)
#         return
#
#     @staticmethod
#     def get_splits(meta_info):
#         all_ids = list(range(meta_info['size']))
#         train_num, valid_num, test_num = meta_info['split']
#         return all_ids[:train_num],  all_ids[train_num+test_num:], all_ids[train_num:train_num+test_num]
#
#     ###
#     ### assume datatype torch, assert grid before x
#     def auto_load_grid(self, data=None):
#         if data is None:
#             if self.scattered_storage:
#                 self.enable_grid = True
#                 return
#             else:
#                 set_globally = True
#                 data = self.x
#         else:
#             set_globally = False
#         space_dim = self.meta_info['space_dim']
#         if space_dim == 1:
#             grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]))
#             grid = torch.unsqueeze(grid[0], dim=-1)
#         elif space_dim == 2:
#             grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]))
#             grid = torch.stack(grid, dim=-1)
#         elif space_dim == 3:
#             grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]),torch.linspace(0,1, data.shape[3]))
#             grid = torch.stack(grid, dim=-1)
#         elif space_dim == 4:
#             grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]),torch.linspace(0,1, data.shape[3]),torch.linspace(0,1, data.shape[4]))
#             grid = torch.stack(grid, dim=-1)
#         else:
#             raise ValueError('dim should be 1, 2, 3 or 4.')
#         if self.meta_info['temporal']:
#             grid = grid.unsqueeze(-2)
#             data = torch.cat([torch.tile(grid.unsqueeze(0),[data.shape[0]] + [1] * space_dim + [data.shape[-2], 1]), data],dim=-1)
#         else:
#             data = torch.cat([torch.tile(grid.unsqueeze(0),[data.shape[0]] + [1] * grid.ndim), data],dim=-1)
#         if set_globally:
#             self.x = data
#         return data
#




# class GridSubDataset(GridDataset):
#     r"""
#     Subset of a dataset at specified indices.
#
#     Args:
#         dataset (Dataset): The whole Dataset
#         indices (sequence): Indices in the whole set selected for subset
#     """
#
#     def __init__(self, dataset: GridDataset, indices: Sequence):
#         self.dataset = dataset
#         self.indices = indices
#
#         ### set status variables
#         self.meta_info = self.dataset.meta_info
#         self.downsample_x = self.dataset.downsample_x
#         self.downsample_y = self.dataset.downsample_y
#         self.gridsize_x = self.dataset.gridsize_x
#         self.gridsize_y = self.dataset.gridsize_y
#
#         self.x = self.dataset.x[self.indices]
#         self.y = self.dataset.y[self.indices]
#         self.theta = self.dataset.theta[self.indices] if self.dataset.theta is not None else None
#
#
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx], self.theta[idx] if self.theta is not None else torch.zeros([])
#
#     def __len__(self):
#         return len(self.indices)