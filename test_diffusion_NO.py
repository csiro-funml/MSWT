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
import pandas as pd
from utils.criterion import RelL2Norm, RMSE, BoundaryRMSE, MaxAbsError, GlobalMaxAbsError, SpectralError

torch.manual_seed(23)

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
    # merge the batch dimension and the shift dimension to make it look like one step 
    x = x.transpose(0,1,5,2,3,4).reshape(-1, Par['channels'], Par['nx'], Par['ny'])
    y = sliding_window_view(y[:,Par['lb']-1:], window_shape=Par['lf'], axis=1 )
    y = y.transpose(0,1,5,2,3,4).reshape(-1, Par['channels'], Par['nx'], Par['ny'])

    print('x: ', x.shape)
    print('y: ', y.shape)
    print()
    return x,y


def load_preprocessing_predictions(args, just_load_path=False):
    res = 128

    ntrain = 5200 if args.dataset == 'ns2d_pda' else 0
    comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
    log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
    log_path = log_path + f'/diffusion'
    if os.path.exists(log_path):
        print(f"Loading data from {log_path}")
    else:
        log_path = './logs/' + comment
    if just_load_path:
        return None, None, None, log_path
    
        # randomly generate x_train, y_train, x_val, y_val, x_test, y_test for testing
    if not torch.cuda.is_available():
        n_train_toy, T_in,T_out, C = 4, 7, 7, 3
        x_test = np.random.randn(n_train_toy, res, res, T_out, C).transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
        y_test = np.random.randn(n_train_toy, res, res, T_out, C).transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
    else:
        x_test = np.load(f"{log_path}/test_pred.npz")['pred'].transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)
        y_test = np.load(f"{log_path}/test_pred.npz")['output'].transpose(0, 3, 4, 1, 2) # (N, H, W, T, C) -> (N, T, C, H, W)

    # load parameters using the robust helper function
    Par = torch.load(f"{log_path}/Par.pth", map_location=device)
    # I will shift the data the original scale during prediction (so as to be compatible with other baselines)

    x_test_tensor  = torch.tensor(x_test,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

    test_dataset = MyDataset(x_test_tensor, y_test_tensor)

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
    # load the model
    model.load_state_dict(torch.load(f"{log_path}/best_model_diffusion.pt", map_location=device))
    model.to(device)
    
    return model, test_loader, Par, log_path

def predict_and_save(model, test_loader, save=False, log_path=None, Par=None):
    # Normalizing the data to [0,1]
    shift_x = Par['inp_shift'].to(device) # (1, T, C, 1, 1)
    scale_x = Par['inp_scale'].to(device) # (1, T, C, 1, 1)
    shift_y = Par['out_shift'].to(device) # (1, T, C, 1, 1)
    scale_y = Par['out_scale'].to(device) # (1, T, C, 1, 1)
    
    # Testing loop
    model.eval()
    save_data = {}
    save_data['output'] = []
    save_data['pred'] = []

    with torch.no_grad():
        for y_cond, y_gt in test_loader:
            with autocast(device_type=device.type):
                # two things:  normalize the data by x scale, and also looping over time steps
                # loop over time steps
                y_cond = y_cond.to(device)
                y_gt = y_gt.to(device)
                pred_i = []
                y_cond_norm = (y_cond - shift_x)/scale_x # (N, T, C, H, W)
                for t in range(y_cond.shape[1]):
                    y_cond_t = y_cond_norm[:, t, :, :, :]
                    # sample the data
                    pred_t = model.sample(y_cond_t.to(device))
                    pred_i.append(pred_t)
                pred = torch.stack(pred_i, dim=0).permute(1, 0, 2, 3, 4) # (T, N, C, H, W) -> (N, T, C, H, W)
                # print("pred shape", pred.shape)
                # print("scale_y shape", scale_y.shape)
                pred = pred*scale_y + shift_y
                # save predictions and ground truth
                save_data['pred'].append(pred)
                save_data['output'].append(y_gt)
        
        # concate the predictions and ground truth
        save_data['pred'] = torch.cat(save_data['pred'], dim=0).permute(0, 3, 4, 1, 2) # (N, T, C, H, W) -> (N, H, W, T, C)
        save_data['output'] = torch.cat(save_data['output'], dim=0).permute(0, 3, 4, 1, 2) # (N, T, C, H, W) -> (N, H, W, T, C)
        
        # save predictions and ground truth
        # print the shape of the np_data
        print("save_data shape", save_data['output'].shape, save_data['pred'].shape)

        # save to npz
        if save:
            torch.save(save_data, f'{log_path}/test_data_diffusion_prediction.pth')
        
        return save_data


def compute_evalutation_metrics(save_data, model_name='', log_path=''):
    pred, target = save_data['pred'], save_data['output'] # shape: (N, H, W, T, C)
    print("pred shape", pred.shape)
    print("target shape", target.shape)
    # for step in [0, -1]: # first step and last step
        # for c in range(pred.shape[-1]):
            # low_err, mid_err, high_err = SpectralError()(pred[..., step, c][:, :, :, None, None], target[..., step, c][:, :, :, None, None])
            

    loss_dict = {}
    loss_dict['rel_l2_loss'] = RelL2Norm() # rel L2 loss
    loss_dict['rmse'] = RMSE()
    loss_dict['boundary_rmse'] = BoundaryRMSE()
    loss_dict['max_avg'] = MaxAbsError()
    loss_dict['max_global'] = GlobalMaxAbsError()
    loss_dict['spectral_error'] = SpectralError(model_name=model_name, save_path=log_path, low_percentile=0.70, high_percentile=0.97)
    
    step_dict = {0: "t=1", -1: "t=T"}

    # Standard error metrics
    print("\n=== Channel-wise Error Metrics ===")
    save_df = pd.DataFrame(columns=["step", "channel", "metric", f"{model_name}"])
    for step in [0, -1]: # first step and last step
        for c in range(pred.shape[-1]):
            # evaluate different metrics per channel
            for key, loss_func in loss_dict.items():
                if key == 'spectral_error':
                    # (B, H, W, T, C)
                    loss_metric = loss_func(pred[..., step, c][:, :, :, None, None], target[..., step, c][:, :, :, None, None], channel=c, 
                                            time_step=step, save_plot=True)
                    # loss metric is a dict with keys: 'low_err', 'mid_err', 'high_err', 'k_low', 'k_high'
                    print("frequency bands", loss_metric['k_low'], loss_metric['k_high'])
                    for band_key, val in loss_metric.items():
                        if band_key == 'k_low' or band_key == 'k_high':
                            continue
                        print(f"Channel {c} {step_dict[step]} {band_key}: {val:.6f}")
                        new_row = pd.Series({"step": step_dict[step], "channel": c, "metric": band_key, f"{model_name}": val}).to_frame().T
                        save_df = pd.concat([save_df, new_row], ignore_index=True)
                else:
                    
                    loss_metric = loss_func(pred[..., step, c][:, :, :, None, None], target[..., step, c][:, :, :, None, None])
                    print(f"Channel {c} {step_dict[step]} {key}: {loss_metric.item():.6f}")   
                    new_row = pd.Series({"step": step_dict[step], "channel": c, "metric": key, f"{model_name}": loss_metric.item()}).to_frame().T
                    save_df = pd.concat([save_df, new_row], ignore_index=True)
    print(save_df.head(n=16))
    save_df.to_csv(f"{log_path}/evalutation_metrics_{model_name}_diffusion.csv", index=False)
    return loss_dict
    


if __name__ == '__main__':
    
    #### 1. predict and save the data
    model, test_loader, Par, log_path = load_preprocessing_predictions(args, just_load_path=False)
    
    #### 2. load the save_data
    save_data = predict_and_save(model, test_loader, save=True, log_path=log_path, Par=Par)
    # save_data = torch.load(f'{log_path}/test_data_diffusion_prediction.pth', map_location=device)
    
    #### 3. compute different types of metrics
    compute_evalutation_metrics(save_data, model_name=args.model, log_path=log_path)