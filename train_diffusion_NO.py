"""
Take the pretrained neural operator and train a diffusion to denoise the predictions
Mostly borrowed from DM-NO: https://github.com/vivekoommen/NeuralOperator_DiffusionModel/
"""

import os
import sys
import argparse
import math
import time
import datetime
import numpy as np
import pickle
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# Import existing utilities and models

from utils.griddataset import TemporalDataset2D, LocalTemporalDataset2D
from utils.utilities import load_model_from_checkpoint
from models.fno import FNO2d
from models.uno import UNO
from models.wavelet_transform import CrossWaveletTransformer
from models.diff_Unet import Unet
from models.diffusion import ElucidatedDiffusion
torch.manual_seed(23)

DTYPE = torch.float32
scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def error_metric(pred, true, Par):
    """Compute relative L2 error between prediction and ground truth"""
    return torch.norm(true - pred, p=2) / torch.norm(true, p=2)


class DiffusionDataset(Dataset):
    """Dataset that generates predictions using a pretrained neural operator"""
    def __init__(self, base_dataset, neural_operator, device):
        self.base_dataset = base_dataset
        self.neural_operator = neural_operator
        self.device = device
        self.neural_operator.eval()  # Set to evaluation mode

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        ## OPTIMIZE LATER BECAUSE THE MODEL PREDICT ONE SAMPLE AT A TIME AND one step a time
        # Get input and ground truth from base dataset
        x, y = self.base_dataset[idx]  # x: input, y: ground truth
        
        # Generate prediction using neural operator
        with torch.no_grad():
            x_input = x.unsqueeze(0).to(self.device)  # Add batch dimension
            if hasattr(self.base_dataset, 'normalize_x'):
                x_input = self.base_dataset.normalize_x(x_input)
            
            pred = self.neural_operator(x_input)
            
            if hasattr(self.base_dataset, 'denormalize_x'):
                pred = self.base_dataset.denormalize_x(pred)
            
            pred = pred.squeeze(0).cpu()  # Remove batch dimension and move to CPU

        # squeeze the time dimension as one step ahead prediction
        pred = pred.squeeze(-2)
        if len(y.shape) == 4: # multipe steps of ground truth were given
            y = y[:, :, -1, :]
        else:
            y = y.squeeze(-2)
        # permute the dimensions to (N, C, H, W)
        pred = pred.permute(2, 0, 1)
        y = y.permute(2, 0, 1)
        # Return prediction as input to diffusion model, ground truth as target
        return pred.float(), y.float()


def load_pretrained_neural_operator(model_type, dataset_name, ntrain, log_path='./logs'):
    """Load a pretrained neural operator from checkpoint"""

    # create the model path
    comment = '{}_{}_ntrain{}'.format(model_type, dataset_name, ntrain)
    
    # temporary log_path
    log_path = log_path +comment
    if not os.path.exists(log_path):# running tests locallt
        log_path = './logs/' + comment
    model_path = log_path + '/model.pth'
    

    if model_type == "FNO":
        model = FNO2d(model_args.modes, model_args.modes, width=model_args.width,
                      n_channels=model_args.n_channels,
                      in_timesteps=model_args.T_in, out_timesteps=1, 
                      n_layers=model_args.n_layers).to(device)
    elif model_type == 'UNO':
        model = UNO(width=model_args.width, n_channels=model_args.n_channels, 
                   in_timesteps=model_args.T_in, out_timesteps=1).to(device)
    elif model_type == 'wavelet_transformer':
        model = CrossWaveletTransformer(wave='haar', n_channels=model_args.n_channels, 
                                       in_timesteps=model_args.T_in, dim=512, depth=8).to(device)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return model, log_path


def create_args_model_name(model_type, n_channels, T_in):
    """Create args object from checkpoint directory"""
    
    class Args:
        pass
    
    args = Args()
    # CHANGE THIS SETTING TO MATCH THE SAVED MODEL

    if model_type == 'FNO':
        args.modes = 16
        args.width = 64
        args.n_layers = 8
        args.T_in = T_in
        args.n_channels = n_channels
        args.n_layers = 8
    elif model_type == 'UNO':
        args.width = 64
        args.n_channels = 3
        args.T_in = 7
        args.dataset = 'ns2d_pda'
        args.normalize = 1
    elif model_type == 'wavelet_transformer':
        args.wave = 'haar'
        args.n_channels = 3
        args.T_in = 7
        args.dataset = 'ns2d_pda'
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    return args


# Parse command line arguments
parser = argparse.ArgumentParser(description='Train diffusion model on neural operator predictions')
parser.add_argument('--model_name', type=str, default='FNO', 
                   choices=['FNO', 'UNO', 'wavelet_transformer'],
                   help='Type of pretrained neural operator')
parser.add_argument('--dataset', type=str, default='ns2d_pda',
                   help='Dataset name')
parser.add_argument('--T_in', type=int, default=7,
                   help='Input time steps')
parser.add_argument('--batch_size', type=int, default=128,
                   help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=10000,
                   help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                   help='Learning rate')
parser.add_argument('--normalize', type=int, default=1,
                   help='Whether to normalize data')
parser.add_argument('--log_path',type=str,default='/scratch3/wan410/operator_learning_model/')
args = parser.parse_args()


# Load datasets using existing infrastructure
begin_time = time.time()

if not torch.cuda.is_available():
    train_base_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=1, 
                                               n_channels=3, normalize=args.normalize, train='train')
    val_base_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=1, 
                                             n_channels=3, normalize=args.normalize, train='test')
    test_base_dataset = val_base_dataset
else:
    train_base_dataset = TemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=1, 
                                          train='train', normalize=args.normalize)
    val_base_dataset = TemporalDataset2D(args.dataset, n_train=260, t_in=args.T_in, t_ar=1, 
                                        train='val', normalize=args.normalize)
    test_base_dataset = TemporalDataset2D(args.dataset, n_train=260, t_in=args.T_in, t_ar=1, 
                                         n_channels=train_base_dataset.n_channels, 
                                         train='test', normalize=args.normalize)


# Load pretrained neural operator
model_args = create_args_model_name(args.model_name, train_base_dataset.n_channels, args.T_in)
neural_operator, log_path = load_pretrained_neural_operator(args.model_name, args.dataset, train_base_dataset.n_size, args.log_path)
args.log_path = log_path

print(f"Loaded {args.model_name} neural operator")


# Create diffusion datasets, note that it will denormalize the predictions
train_dataset = DiffusionDataset(train_base_dataset, neural_operator, device)
val_dataset = DiffusionDataset(val_base_dataset, neural_operator, device)
test_dataset = DiffusionDataset(test_base_dataset, neural_operator, device)

print(f"Data Loading Time: {time.time() - begin_time:.1f}s")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Get sample data to determine dimensions and compute normalization statistics
sample_pred, sample_true = train_dataset[0]
print(f"Sample prediction shape: {sample_pred.shape}")
print(f"Sample true shape: {sample_true.shape}")

# Compute normalization statistics from a subset of training data
print("Computing normalization statistics...")
preds_for_norm = []
trues_for_norm = []
for i in range(min(100, len(train_dataset))):
    pred, true = train_dataset[i]
    preds_for_norm.append(pred)
    trues_for_norm.append(true)

preds_stack = torch.stack(preds_for_norm) # (N, C, H, W)
trues_stack = torch.stack(trues_for_norm) # (N, C, H, W)

# Compute min/max for normalization
inp_min = torch.amin(preds_stack, dim=(0, 2, 3), keepdim=True) # (1, C, 1, 1)
inp_max = torch.amax(preds_stack, dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
out_min = torch.amin(trues_stack, dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
out_max = torch.amax(trues_stack, dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)

# MIN MAX NORMALIZATION
Par = {
    "inp_shift": inp_min.to(device, dtype=DTYPE),
    "inp_scale": (inp_max - inp_min).to(device, dtype=DTYPE),
    "out_shift": out_min.to(device, dtype=DTYPE),
    "out_scale": (out_max - out_min).to(device, dtype=DTYPE),
    "nx": sample_pred.shape[1],
    "ny": sample_pred.shape[2],
    "nf": 1,
    "lb": 1,
    "lf": 1,
    "num_epochs": args.num_epochs
}

# Compute sigma_data for diffusion model, it produces a scalar? should be a sigma for each channel
Par["sigma_data"] = torch.std(trues_stack, dim=(0, 2, 3)).to(device)
print(f"Sigma data: {Par['sigma_data']}")

# Update parameters
Par.update({
    "channels": sample_pred.shape[0],
    "self_condition": True
})

print("Parameters:")
for key, value in Par.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {value}")

# Save parameters
with open(args.log_path + 'Par.pkl', 'wb') as f:
    pickle.dump(Par, f)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Define Network Architecture
net = Unet(
    dim=16,
    dim_mults=(1, 2, 4, 8, 8),
    channels=Par["channels"],
    self_condition=Par["self_condition"],
    flash_attn=True
).to(device).to(torch.float32)

# Print model summary
print(f"Model created with {sum(p.numel() for p in net.parameters())} parameters")

model = ElucidatedDiffusion(
    net,
    channels=Par["channels"],
    image_size=Par["nx"],
    sigma_data=Par["sigma_data"]
).to(device)

# Profile the model

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max=Par['num_epochs'] * len(train_loader))

# Training loop
num_epochs = Par['num_epochs']
best_val_loss = float('inf')
best_model_id = 0

os.makedirs('models', exist_ok=True)
t0 = time.time()

print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    begin_time = time.time()
    model.train()
    train_loss = 0.0

    train_time = time.time()
    for l_fidel, h_fidel in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        # l_fidel is the NO output, h_fidel is the ground truth
        # Normalize data
        l_fidel = (l_fidel - Par['inp_shift'].cpu()) / (Par['inp_scale'].cpu() + 1e-6)
        h_fidel = (h_fidel - Par['out_shift'].cpu()) / (Par['out_scale'].cpu() + 1e-6)
        
        with autocast(device_type=device.type):
            loss = model(h_fidel.to(device), l_fidel.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * l_fidel.shape[0]

        # Update learning rate
        scheduler.step()

    train_loss /= len(train_loader)
    train_time = time.time()-train_time

    # Validation
    # if epoch !=0 and epoch % 10 == 0:
    if epoch % 10 == 0: # temporarily for testing
        val_time = time.time()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for l_fidel, h_fidel in val_loader:
                # Normalize data
                l_fidel = (l_fidel - Par['inp_shift'].cpu()) / (Par['inp_scale'].cpu() + 1e-6)
                h_fidel = (h_fidel - Par['out_shift'].cpu()) / (Par['out_scale'].cpu() + 1e-6)
                
                with autocast(device_type=device.type):
                    # Sample from diffusion model using the low-fidelity input as conditioning
                    pred = model.sample(l_fidel.to(device))
                    loss = error_metric(pred, h_fidel.to(device), Par)
                val_loss += loss.item() * l_fidel.shape[0]

        val_loss /= len(val_loader)

            # Save the model if validation loss is the lowest so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_id = epoch+1
            torch.save(model.state_dict(), f'{args.log_path}/model_diffusion.pt')
        
        if epoch % 500 == 0:
            torch.save(model.state_dict(), f'{args.log_path}/model_diffusion_{epoch}.pt')

        val_time = time.time() - val_time
        
        time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
        elapsed_time = time.time() - begin_time
        print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, train time: {train_time:.2f}, val time: {val_time:.2f}')

    else:
        time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
        print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, LR: {scheduler.get_last_lr()[0]:.4e}, train time: {train_time:.2f}')


print('Training finished.')
print(f"Training Time: {time.time() - t0:.1f}s")

# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for l_fidel, h_fidel in test_loader:
        # Normalize data
        l_fidel = (l_fidel - Par['inp_shift'].cpu()) / (Par['inp_scale'].cpu() + 1e-6)
        h_fidel = (h_fidel - Par['out_shift'].cpu()) / (Par['out_scale'].cpu() + 1e-6)
        
        with autocast():
            pred = model.sample(l_fidel.to(device))
            loss = error_metric(pred, h_fidel.to(device), Par)
        test_loss += loss.item() * l_fidel.shape[0]

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')