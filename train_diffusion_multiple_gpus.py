# ---------------------------------------------------------------------------------------------
# Author: Vivek Oommen (Adapted for PyTorch Lightning multi-GPU training)
# Date: 08/01/2024
# ---------------------------------------------------------------------------------------------

import os
import sys
import math
import time
import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import pickle

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Model imports
from models.diff_Unet import Unet
from models.diffusion import ElucidatedDiffusion

torch.manual_seed(23)

DTYPE = torch.float32


def error_metric(pred, true, Par):
    """Compute relative L2 error"""
    return torch.norm(true - pred, p=2) / torch.norm(true, p=2)


def preprocess(x, y, Par):
    """Preprocess data using sliding window"""
    x = sliding_window_view(x[:, Par['lb']-1:, :, :], window_shape=Par['lf'], axis=1).transpose(0, 1, 4, 2, 3).reshape(-1, Par['lf'], Par['nx'], Par['ny'])
    y = sliding_window_view(y[:, Par['lb']-1:, :, :], window_shape=Par['lf'], axis=1).transpose(0, 1, 4, 2, 3).reshape(-1, Par['lf'], Par['nx'], Par['ny'])
    return x, y


class MyDataset(Dataset):
    """Custom dataset class"""
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample, y_sample = self.transform(x_sample, y_sample)

        return x_sample, y_sample


class DiffusionDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling data loading"""
    
    def __init__(self, log_path, Par, batch_size=100, num_workers=4):
        super().__init__()
        self.log_path = log_path
        self.Par = Par
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        """Load and preprocess data"""
        
        # Load raw data
        x_train = np.load(f"{self.log_path}/train_pred.npz")['pred'][..., 0].transpose(0, 3, 1, 2)
        y_train = np.load(f"{self.log_path}/train_pred.npz")['output'][..., 0].transpose(0, 3, 1, 2)
        
        x_val = np.load(f"{self.log_path}/val_pred.npz")['pred'][..., 0].transpose(0, 3, 1, 2)
        y_val = np.load(f"{self.log_path}/val_pred.npz")['output'][..., 0].transpose(0, 3, 1, 2)
        
        x_test = np.load(f"{self.log_path}/test_pred.npz")['pred'][..., 0].transpose(0, 3, 1, 2)
        y_test = np.load(f"{self.log_path}/test_pred.npz")['output'][..., 0].transpose(0, 3, 1, 2)
        
        # Compute normalization parameters
        inp_min = np.min(x_train, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        inp_max = np.max(x_train, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        out_min = np.min(y_train, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        out_max = np.max(y_train, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        
        # Store normalization parameters
        self.Par.update({
            "inp_shift": torch.tensor(inp_min, dtype=DTYPE),
            "inp_scale": torch.tensor(inp_max - inp_min, dtype=DTYPE),
            "out_shift": torch.tensor(out_min, dtype=DTYPE),
            "out_scale": torch.tensor(out_max - out_min, dtype=DTYPE),
            "nx": x_train.shape[2],
            "ny": x_train.shape[3],
            "nf": 1,
            "lb": 1,
            "lf": 1
        })
        
        # Normalize data
        shift = self.Par['inp_shift'].detach().cpu().numpy()
        scale = self.Par['inp_scale'].detach().cpu().numpy()
        x_train = (x_train - shift) / scale
        x_val = (x_val - shift) / scale
        x_test = (x_test - shift) / scale
        
        shift = self.Par['out_shift'].detach().cpu().numpy()
        scale = self.Par['out_scale'].detach().cpu().numpy()
        y_train = (y_train - shift) / scale
        y_val = (y_val - shift) / scale
        y_test = (y_test - shift) / scale
        
        self.Par["sigma_data"] = np.std(y_train)
        
        # Preprocess data
        x_train, y_train = preprocess(x_train, y_train, self.Par)
        x_val, y_val = preprocess(x_val, y_val, self.Par)
        x_test, y_test = preprocess(x_test, y_test, self.Par)
        
        # Update parameters
        self.Par.update({
            "channels": x_train.shape[1],
            "self_condition": True
        })
        
        # Convert to tensors and create datasets
        self.train_dataset = MyDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        self.val_dataset = MyDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        self.test_dataset = MyDataset(
            torch.tensor(x_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )


class DiffusionLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for the Diffusion Model"""
    
    def __init__(self, Par, learning_rate=1e-4, weight_decay=0):
        super().__init__()
        self.save_hyperparameters()
        
        self.Par = Par
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Build the model
        self.net = Unet(
            dim=16,
            dim_mults=(1, 2, 4, 8, 8),
            channels=Par["channels"],
            self_condition=Par["self_condition"],
            flash_attn=True
        )
        
        self.model = ElucidatedDiffusion(
            self.net,
            channels=Par["channels"],
            image_size=Par["nx"],
            sigma_data=Par["sigma_data"]
        )
        
        # Store validation metrics
        self.validation_step_outputs = []
        
    def forward(self, h_fidel, l_fidel):
        """Forward pass"""
        return self.model(h_fidel, l_fidel)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        l_fidel, h_fidel = batch
        
        # Compute loss (diffusion model handles the forward pass)
        loss = self.model(h_fidel, l_fidel)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        l_fidel, h_fidel = batch
        
        # Sample from the diffusion model
        pred = self.model.sample(l_fidel)
        
        # Compute validation loss using error metric
        val_loss = error_metric(pred, h_fidel, self.Par)
        
        # Store for epoch-end processing
        self.validation_step_outputs.append(val_loss)
        
        # Log validation loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return val_loss
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Clear the stored outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        l_fidel, h_fidel = batch
        
        # Sample from the diffusion model
        pred = self.model.sample(l_fidel)
        
        # Compute test loss using error metric
        test_loss = error_metric(pred, h_fidel, self.Par)
        
        # Log test loss
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return test_loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Note: Lightning will handle the total_steps calculation
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=1000),  # Will be updated in trainer
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
        }
        
        return [optimizer], [scheduler]


def main():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Multi-GPU training with PyTorch Lightning')
    parser.add_argument('--model', type=str, default='diffusion', help='Model type')
    parser.add_argument('--dataset', type=str, default='ns2d_pda', help='Dataset name')
    parser.add_argument('--comment', type=str, default="", help='Comment for logging')
    parser.add_argument('--log_path', type=str, default='/scratch3/wan410/operator_learning_model/', help='Log path')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs (-1 for all available)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--precision', type=str, default='16-mixed', help='Training precision (16-mixed, 32, bf16-mixed)')
    parser.add_argument('--strategy', type=str, default='ddp', help='Training strategy (ddp, ddp_spawn, etc.)')
    parser.add_argument('--val_check_interval', type=int, default=10, help='Validation check interval (epochs)')
    
    args = parser.parse_args()
    
    # Set up paths and logging
    ntrain = 5200 if args.dataset == 'ns2d_pda' else 0
    comment = args.comment + f'{args.model}_{args.dataset}_ntrain{ntrain}'
    log_path = f'./logs/{time.strftime("%m%d_%H_%M_%S")}{comment}' if len(args.log_path) == 0 else os.path.join('./logs', args.log_path + comment)
    
    # Create log directory
    os.makedirs(log_path, exist_ok=True)
    
    # Initialize parameters dictionary
    Par = {
        "num_epochs": args.max_epochs
    }
    
    # Create data module
    data_module = DiffusionDataModule(
        log_path=log_path,
        Par=Par,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup data to get parameters
    data_module.setup()
    
    # Save parameters
    with open(os.path.join(log_path, 'Par.pkl'), 'wb') as f:
        pickle.dump(Par, f)
    
    # Create Lightning module
    model = DiffusionLightningModule(
        Par=Par,
        learning_rate=args.learning_rate
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=log_path,
        filename='best_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=log_path,
        name='lightning_logs',
        version=None
    )
    
    # Configure trainer for multi-GPU training
    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        strategy=args.strategy,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        check_val_every_n_epoch=args.val_check_interval,
        log_every_n_steps=50,
        enable_progress_bar=True,
        gradient_clip_val=1.0,  # Gradient clipping for stability
    )
    
    # Print training info
    print(f"Starting training with the following configuration:")
    print(f"  GPUs: {trainer.num_devices}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Precision: {args.precision}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Log path: {log_path}")
    print("")
    
    # Start training
    trainer.fit(model, datamodule=data_module)
    
    # Test the model
    trainer.test(model, datamodule=data_module)
    
    print("Training completed!")


if __name__ == '__main__':
    main()
