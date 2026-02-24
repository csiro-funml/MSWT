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
    """
    load the ground truth data from the dataset
    return:
        u: (N, nc, nx, ny, nt+1)
        S_forcing: int
    """
    data= np.load(os.path.join(datapath, 'kolmogorov_dataset.npz'))
    X_train, y_train = data['X_train'], data['y_train']
    nc, nx, ny = X_train.shape[1:]
    X_train = X_train.reshape(-1, nt, nc, nx, ny)
    y_train = y_train.reshape(-1, nt, nc, nx, ny)
    
    u = np.concatenate([X_train[:, :1], y_train], axis=1) # (N, nt+1, nc, nx, ny) concatenate the initial condition and the trajectory
    u = u.permute(0, 2, 3, 4, 1) # (N, nt+1, 3, nx, ny) -> (N, 3, nx, ny, nt+1)
    u = torch.from_numpy(u).to(device)
    S_forcing = nx
    return u, S_forcing



def verify_pde_loss(u, forcing):
    """ 
    u: (N, nc, nx, ny, nt+1) from load_ns_ground_truth, compute PINO_loss to test if the PDE loss is zero
    forcing: (1, 2, nx, ny, 1) from get_forcing_vel
    return:
        total_loss_cont: float
        total_loss_ic: float
        total_loss_momx: float
        total_loss_momy: float
    """
    total_loss_cont = 0.0
    total_loss_ic = 0.0
    total_loss_momx = 0.0
    total_loss_momy = 0.0 
    for i in range(len(u)):
        # print(f'Sample {i} shape: {data[i].shape}')
        u_i = u[i].unsqueeze(0) # one realization (1, 3, nx, ny, nt+1)
        u0_i = u_i[..., 0] # initial condition (1, 3, nx, ny)
        loss_ic, loss_cont, loss_momx, loss_momy= PINO_loss3d_vel(u_i, u0_i, forcing)
        print(f'Sample {i} PDE loss: {loss_cont.item()}, IC loss: {loss_ic.item()}, momx loss: {loss_momx.item()}, momy loss: {loss_momy.item()}')
        total_loss_cont += loss_cont.item()
        total_loss_ic += loss_ic.item()
        total_loss_momx += loss_momx.item()
        total_loss_momy += loss_momy.item()
    print(f'average PDE loss: {total_loss_cont / len(u)}')
    print(f'average IC loss: {total_loss_ic / len(u)}')
    print(f'average momx loss: {total_loss_momx / len(u)}')
    print(f'average momy loss: {total_loss_momy / len(u)}')
    return total_loss_cont / len(u), total_loss_ic / len(u), total_loss_momx / len(u), total_loss_momy / len(u)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    u, S_forcing = load_ns_ground_truth(datapath='/scratch3/wan410/operator_learning_data/Dedalus', device=device)
    # parse options
    forcing = get_forcing_vel(S_forcing).to(device)
    
    verify_pde_loss(u, forcing)


