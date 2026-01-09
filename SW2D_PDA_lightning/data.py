"""DataModule for SW2D_PDA Lightning training."""

import torch
from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from SW2D_PDA.data_utils.datasets import SWLoader2D


class SyntheticStepDataset(Dataset):
    def __init__(self, arr):
        self.arr = arr
        self.max_t = arr.shape[-2] - 1

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        sample = self.arr[idx]
        t = torch.randint(0, self.max_t, ()).item()
        return sample[..., t, :], sample[..., t + 1, :]


class SW2DPDALightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_cfg,
        batch_size,
        test_ratio=0.0,
        test_seed=42,
        synthetic_samples=0,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()
        self.data_cfg = data_cfg
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.test_seed = test_seed
        self.synthetic_samples = synthetic_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_set = None
        self.val_set = None
        self.S_data = None

    def setup(self, stage=None):
        if self.train_set is not None:
            return

        if self.synthetic_samples > 0:
            sub = self.data_cfg.get('sub', 1)
            sub_t = self.data_cfg.get('sub_t', 1)
            nx = self.data_cfg['nx']
            ny = self.data_cfg['ny']
            nt = self.data_cfg['nt']
            time_scale = self.data_cfg.get('time_interval', 1.0)
            s1 = nx // sub
            s2 = ny // sub
            t = int(nt * time_scale) // sub_t + 1
            n_channels = self.data_cfg.get('n_channels', 2)
            data = torch.rand(self.synthetic_samples, s1, s2, t, n_channels)
            full_dataset = SyntheticStepDataset(data)
            self.S_data = (s1, s2)
        else:
            full_dataset = SWLoader2D(
                datapath1=self.data_cfg['datapath'],
                nx=self.data_cfg['nx'],
                ny=self.data_cfg['ny'],
                nt=self.data_cfg['nt'],
                sub=self.data_cfg['sub'],
                sub_t=self.data_cfg['sub_t'],
                N=self.data_cfg['total_num'],
                t_interval=self.data_cfg['time_interval'],
                n_samples=self.data_cfg.get('n_sample', self.data_cfg.get('n_samples', self.data_cfg['total_num'])),
                offset=self.data_cfg.get('offset', 0),
                normalizer_path=self.data_cfg.get('normalizer_path', None),
            )
            self.S_data = full_dataset.S

        if self.test_ratio > 0:
            test_size = max(1, int(len(full_dataset) * self.test_ratio))
            if len(full_dataset) - test_size <= 0:
                raise ValueError('test_ratio is too large; no samples left for training.')
            train_size = len(full_dataset) - test_size
            self.train_set, self.val_set = random_split(
                full_dataset,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(self.test_seed),
            )
            if hasattr(self.val_set, 'train'):
                self.val_set.train = False
        else:
            self.train_set = full_dataset
            self.val_set = None

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.data_cfg['shuffle'],
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.val_set is None:
            return None
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
