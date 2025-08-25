# ---------------------------------------------------------------------------------------------
# Author: Vivek Oommen
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
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import argparse
from torch.utils.tensorboard import SummaryWriter
from models.diff_Unet import Unet
from models.diffusion import ElucidatedDiffusion
from torchvision.utils import make_grid

torch.manual_seed(23)
import pickle

DTYPE = torch.float32

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='FNO') # FNO, ViT, UNO, CNO, Oformer, Transolver, DPOT, Crossformer, 
parser.add_argument('--dataset',type=str, default='ns2d_pda') # ['ns2d_fno_1e-3', 'ns2d_pda', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 'sw2d_pda'], note: pdb is the pde bench
parser.add_argument('--use_writer', action='store_true', default=False)
parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='/scratch3/wan410/operator_learning_model/')
parser.add_argument('--batch_size',type=int,default=512)
args = parser.parse_args()


def error_metric(pred,true, Par):
    #re-normalize
    # true = true*Par['out_scale'] + Par['out_shift']
    # true = true*Par['out_scale'] + Par['out_shift']
    return torch.norm(true-pred, p=2)/torch.norm(true, p=2)

class MyDataset(Dataset):
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
    
def preprocess(x,y, Par):
    x = sliding_window_view(x[:,Par['lb']-1:], window_shape=Par['lf'], axis=1 )
    # merge the batch dimension and the shift dimension to make it look like one step (N, C, H, W)
    x = x.transpose(0,1,5,2,3,4).reshape(-1, Par['channels'], Par['nx'], Par['ny'])
    y = sliding_window_view(y[:,Par['lb']-1:], window_shape=Par['lf'], axis=1 )
    y = y.transpose(0,1,5,2,3,4).reshape(-1, Par['channels'], Par['nx'], Par['ny'])

    print('x: ', x.shape)
    print('y: ', y.shape)
    print()
    return x,y

res = 128
begin_time = time.time()


ntrain = 5200 if args.dataset == 'ns2d_pda' else 0
comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)


# inp = np.load(f"/oscar/data/gk/voommen/no_diffusion/kolmogrov/res_{res}/matcho/Y_PRED.npy") #low-fidelity
# out = np.load(f"/oscar/data/gk/voommen/no_diffusion/kolmogrov/res_{res}/matcho/Y_TRUE.npy") #high-fidelity


# randomly generate x_train, y_train, x_val, y_val, x_test, y_test for testing
if not torch.cuda.is_available():
    n_train_toy, T_in,T_out, C = 4, 7, 7, 3
    x_train = np.random.randn(n_train_toy, res, res, T_in, C).transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
    y_train = np.random.randn(n_train_toy, res, res, T_in, C).transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
    x_val = np.random.randn(n_train_toy, res, res, T_out, C).transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
    y_val = np.random.randn(n_train_toy, res, res, T_out, C).transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
    x_test = np.random.randn(n_train_toy, res, res, T_out, C).transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
    y_test = np.random.randn(n_train_toy, res, res, T_out, C).transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
else:
    x_train = np.load(f"{log_path}/diffusion/train_pred.npz")['pred'].transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
    y_train = np.load(f"{log_path}/diffusion/train_pred.npz")['output'].transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)

    x_val = np.load(f"{log_path}/diffusion/val_pred.npz")['pred'].transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
    y_val = np.load(f"{log_path}/diffusion/val_pred.npz")['output'].transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)

    x_test = np.load(f"{log_path}/diffusion/test_pred.npz")['pred'].transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
    y_test = np.load(f"{log_path}/diffusion/test_pred.npz")['output'].transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)


print("x_train shape", x_train.shape, "y_train shape", y_train.shape)
print("x_val shape", x_val.shape, "y_val shape", y_val.shape)
print("x_test shape", x_test.shape, "y_test shape", y_test.shape)

print(f"Data Loading Time: {time.time() - begin_time:.1f}s")
C = x_train.shape[2] # number of channels

#  (N, T, C, H, W) -> (1, T, C, 1, 1), get the min/max per channel and time step
inp_min = np.min(x_train, axis=(0,3,4)).reshape(1,-1,C, 1,1) # (1, T, C, 1, 1)
inp_max = np.max(x_train, axis=(0,3,4)).reshape(1,-1,C, 1,1) # (1, T, C, 1, 1)
out_min = np.min(y_train, axis=(0,3,4)).reshape(1,-1,C, 1,1) # (1, T, C, 1, 1)
out_max = np.max(y_train, axis=(0,3,4)).reshape(1,-1,C, 1,1) # (1, T, C, 1, 1)



Par = {"inp_shift" : torch.tensor(inp_min, dtype=DTYPE, device=device),
       "inp_scale" : torch.tensor(inp_max - inp_min, dtype=DTYPE, device=device),
       "out_shift" : torch.tensor(out_min, dtype=DTYPE, device=device),
       "out_scale" : torch.tensor(out_max - out_min, dtype=DTYPE, device=device),
       "nx"        : x_train.shape[-2],
       "ny"        : x_train.shape[-1],
       "nf"        : 1,
       "lb"        : 1,
       "lf"        : 1,
       "num_epochs": 10000,
       "channels"  : C
       }

# Normalizing the data to [0,1]
shift = Par['inp_shift'].detach().cpu().numpy()
scale = Par['inp_scale'].detach().cpu().numpy()
x_train = (x_train - shift)/scale
x_val = (x_val - shift)/scale
x_test = (x_test - shift)/scale

shift = Par['out_shift'].detach().cpu().numpy()
scale = Par['out_scale'].detach().cpu().numpy()
y_train = (y_train - shift)/scale
y_val = (y_val - shift)/scale
y_test = (y_test - shift)/scale

Par["sigma_data"] = np.std(y_train, axis=(0,1,3,4)) # I feel the sigma_data should be per channel

# Traj splitting
begin_time = time.time()
print('\nTrain Dataset')
x_train, y_train = preprocess(x_train, y_train, Par)
print('\nValidation Dataset')
x_val, y_val = preprocess(x_val, y_val, Par)
print('\nTest Dataset')
x_test, y_test = preprocess(x_test, y_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

Par.update({"self_condition" : True
            })

print("Par")
# save Par to a pth file
torch.save(Par, log_path + '/Par.pth')

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_val_tensor   = torch.tensor(x_val,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)

x_test_tensor  = torch.tensor(x_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

train_dataset = MyDataset(x_train_tensor, y_train_tensor)
val_dataset = MyDataset(x_val_tensor, y_val_tensor)
test_dataset = MyDataset(x_test_tensor, y_test_tensor)

# Define data loaders

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Define Network Architecture
net = Unet(
    dim = 16,
    dim_mults = (1, 2, 4, 8, 8),
    channels = Par["channels"],
    self_condition = Par["self_condition"],
    flash_attn = True
).to(device).to(torch.float32)


model = ElucidatedDiffusion(net,
                                channels = Par["channels"],
                                image_size=Par["nx"],
                                sigma_data=Par["sigma_data"])

# Adjust the dimensions as per your model's input size
dummy_x = torch.tensor(torch.randn(1, Par["channels"], Par["nx"], Par["ny"]),   dtype=DTYPE, device=device)
dummy_input = (dummy_x, dummy_x)


optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * len(train_loader) )  # Adjust T_max as needed

# Training loop
num_epochs = Par['num_epochs']
best_val_loss = float('inf')
best_model_id = 0


if args.use_writer:
    writer = SummaryWriter(log_dir=log_path)

t0 = time.time()
for epoch in tqdm(range(num_epochs)):
    begin_time = time.time()
    model.train()
    train_loss = 0.0

    train_time = time.time()
    for l_fidel, h_fidel  in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            loss = model(h_fidel.to(device), l_fidel.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        # Update learning rate
        # scheduler.step()

    train_loss /= len(train_loader)
    train_time = time.time()-train_time
    if args.use_writer:
        writer.add_scalar("train_loss", train_loss, epoch)

    # Validation
    if epoch % 10 == 0:
        val_time = time.time()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for l_fidel, h_fidel in val_loader:
                with autocast(device_type=device.type):
                    pred = model.sample(l_fidel.to(device))
                    loss   = error_metric(pred, h_fidel.to(device), Par)
                val_loss += loss.item()

        val_loss /= len(val_loader)

            # Save the model if validation loss is the lowest so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_id = epoch+1
            torch.save(model.state_dict(), f'{log_path}/best_model_diffusion.pt')

        val_time = time.time() - val_time
        if args.use_writer:
            writer.add_scalar("val_loss", val_loss, epoch)

            # get one sample from the test set then save the prediction to writer
            l_fidel, h_fidel = next(iter(test_loader))
            h_fidel = h_fidel[0].unsqueeze(0).to(device) # only one sample
            l_fidel = l_fidel[0].unsqueeze(0).to(device)
            pred, sampling_images = model.sample(l_fidel.to(device), save_sampling_images=True) # (n_sample, B, C, H, W)
            # print("l_fidel.shape", l_fidel.shape, "h_fidel.shape", h_fidel.shape, "pred.shape", pred.shape, "sampling_images.shape", sampling_images.shape)
            channel_idx = 0
            sampling_images = sampling_images[:, :, channel_idx].cpu() # (n_sample, 1, H, W), # use the batch dimension as the fake channel dimension
            pred = pred[:, channel_idx].cpu() # use the batch dimension as the fake channel dimension
            l_fidel = l_fidel[:, channel_idx].cpu()
            h_fidel = h_fidel[:, channel_idx].cpu()
            
            sampling_images = make_grid(sampling_images, nrow=10)
            writer.add_image("NO_DM_sampling", sampling_images, epoch)
            writer.add_image("NO_DM_pred", pred, epoch)
            writer.add_image("NO_pred", l_fidel)
            writer.add_image("ground_truth", h_fidel)
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
        with autocast(device_type=device.type):
            pred = model.sample(l_fidel.to(device))
            loss   = error_metric(pred, h_fidel.to(device), Par)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')