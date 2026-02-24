# verify the PDE loss computation
# use the ground truth of the data to see if the boundary condition, and PDE loss is satisfied

import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.criterion import LpLoss, PINO_loss3d_vel, get_forcing_vel
# from argparse import ArgumentParser
import torch
import numpy as np


def load_ns_ground_truth(datapath, nt=63, device=None):
    data= np.load(os.path.join(datapath, 'kolmogorov_dataset.npz'))
    X_train, y_train = data['X_train'], data['y_train']
    nc, nx, ny = X_train.shape[1:]
    X_train = X_train.reshape(-1, nt, nc, nx, ny)
    y_train = y_train.reshape(-1, nt, nc, nx, ny)
    
    u = np.concatenate([X_train[:, :1], y_train], axis=1) # (N, nt+1, nc, nx, ny) concatenate the initial condition and the trajectory
    u = torch.from_numpy(u).to(device)
    return u, nx



def verify_pde_loss(data, forcing, device):
    """ 
    data: (N, nt+1, 3, nx, ny) from load_ns_ground_truth, compute PINO_loss to test if the PDE loss is zero
    # u: Predicted fields, shape (batchsize, 3, nx, ny, nt).
    # Channels: ux, uy, p.
    # u0: Initial condition, shape (batchsize, 3, nx, ny) or broadcastable.
    """
    total_loss_f = 0.0
    total_loss_ic = 0.0
    data = data.permute(0, 2, 3, 4, 1) # (N, nt+1, 3, nx, ny) -> (N, 3, nx, ny, nt+1)
    for i in range(len(data)):
        print(f'Sample {i} shape: {data[i].shape}')
        u = data[i].unsqueeze(0) # one realization (1, 3, nx, ny, nt+1)
        u0 = u[..., 0] # initial condition (1, 3, nx, ny)
        loss_ic, loss_f = PINO_loss3d_vel(u, u0, forcing)
        print(f'Sample {i} PDE loss: {loss_f.item()}, IC loss: {loss_ic.item()}')
        total_loss_f += loss_f.item()
        total_loss_ic += loss_ic.item()
    print(f'average PDE loss: {total_loss_f / len(data)}')
    print(f'average IC loss: {total_loss_ic / len(data)}')
    return total_loss_f / len(data), total_loss_ic / len(data)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    data, S_forcing = load_ns_ground_truth(datapath='/scratch3/wan410/operator_learning_data/Dedalus/kolmogorov_dataset.npz', device=device)
    # parse options
    forcing = get_forcing_vel(S_forcing).to(device)
    
    verify_pde_loss(data, forcing, device)


