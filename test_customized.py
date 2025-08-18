# TEST THE Predictions of the model

import sys
import os
# sys.path.append(['.','./../'])
# os.environ['OMP_NUM_THREADS'] = '16'
import warnings
import json
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange


from timeit import default_timer
from torch.utils.tensorboard import SummaryWriter
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint, resume_training_from_checkpoint

from utils.griddataset import MixedTemporalDataset, TemporalDataset2D, TemporalDataset2D_multiscale, LocalTemporalDataset2D
from utils.make_master_file import DATASET_DICT
from models.fno import FNO2d
from models.uno import UNO
from models.wavelet_transform import CrossWaveletTransformer
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils.criterion import RelL2Norm, RMSE, BoundaryRMSE, MaxAbsError, GlobalMaxAbsError, get_frequency_bands_from_cumulative_energy, aggregate_spectral_energy_by_bands

warnings.filterwarnings("ignore")

################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='FNO') # FNO, ViT, UNO, CNO, Oformer, Transolver, DPOT, Crossformer, wavelet_transformer
parser.add_argument('--dataset',type=str, default='ns2d_pda') # ['ns2d_fno_1e-3', 'ns2d_pda', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 'sw2d_pda'], note: pdb is the pde bench
parser.add_argument('--resume_path',type=int, default=0 if not torch.cuda.is_available() else 1) # use random weights if not cuda available
parser.add_argument('--use_writer', action='store_true',default=False)



# ### dataset details
parser.add_argument('--T_in', type=int, default=7)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--pad', type=int, default=0)
parser.add_argument('--normalize',type=int, default=1)


# ### FNO/UNO params 
parser.add_argument('--n_layers',type=int, default=8)
parser.add_argument('--modes', type=int, default=16)
# parser.add_argument('--modes', type=int, default=32)
parser.add_argument('--width', type=int, default=64)
# parser.add_argument('--use_ln',type=int, default=0)
parser.add_argument('--act',type=str, default='gelu')


# ### DPOT
# parser.add_argument('--patch_size',type=int, default=8)
parser.add_argument('--n_blocks',type=int, default=8)
parser.add_argument('--mlp_ratio',type=int, default=1)
parser.add_argument('--out_layer_dim', type=int, default=32)

# ### ViT
parser.add_argument('--patch_size',type=int, default=16)
# parser.add_argument('--coord',type=str, default='fourier', choices=['fourier', 'cartesian', 'learnable', 'spherical'])
# parser.add_argument('--prenorm',type=str, default='standard', choices=['standard', 'lognorm', 'instancenorm', 'batchnorm'])


###### optimizer and training setups
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--save_everyepoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb'])
parser.add_argument('--beta1',type=float,default=0.9)
parser.add_argument('--beta2',type=float,default=0.9)
parser.add_argument('--lr_method',type=str, default='cossin') # cyclic for ViT perhaps
parser.add_argument('--grad_clip',type=float, default=10000.0)
parser.add_argument('--step_size', type=int, default=20)
parser.add_argument('--step_gamma', type=float, default=0.5)
parser.add_argument('--warmup_epochs',type=int, default=100)

parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='/scratch3/wan410/operator_learning_model/')

args = parser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Current working directory: {os.getcwd()}")


def load_data_model(just_load_path=False):
    ################################################################
    # load some toy data to run locally
    if not torch.cuda.is_available():
        train_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, n_channels=3, normalize=args.normalize, train='train')
        test_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels=3, normalize=args.normalize, train='test')
    else:
        # load data and dataloader
        train_dataset = TemporalDataset2D(args.dataset, t_in = args.T_in, t_ar = args.T_ar, train='train', normalize=args.normalize)
        test_dataset = TemporalDataset2D(args.dataset, n_train=260, t_in=args.T_in, t_ar=-1, n_channels = train_dataset.n_channels, train='test', normalize=args.normalize)


    
    ntrain, ntest = len(train_dataset), len(test_dataset)
    ntrain = 5200 if args.dataset == 'ns2d_pda' else ntrain # for testing
    print(args.dataset)
    print('Train num {}, Test num {}'.format(ntrain, ntest))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
    # val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)

    # ntrain, ntest = len(train_dataset), len(test_dataset)

    if not args.pad:
        args.res = train_dataset.res  # use original dataset  resolution to train the model

    comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
    log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
    if not os.path.exists(log_path):# running tests locallt
        log_path = './logs/' + comment
    model_path = log_path + '/model.pth'
    print(model_path)
    
    # if just_load_path, return the log_path
    if just_load_path:
        return None, test_loader,log_path
    
    if args.use_writer:
        writer = SummaryWriter(log_dir=log_path)
        # write params (usually you only do this once)
        json.dump(vars(args),
            open(os.path.join(log_path, 'params.json'), 'w'),
            indent=4)
        # open the log file in append mode, line‑buffered
        fp = open(os.path.join(log_path, 'logs.txt'), 'a', buffering=1)
        # redirect stdout there (so all prints go into logs.txt)
        sys.stdout = fp
    else:
        writer = None

    print('args',args)
    # find the maximum number of steps in the test dataset
    max_steps = test_dataset[0][1].shape[-2]
    print('Train num {} train len {} test num {} test max steps {}'.format(train_dataset.n_size, ntrain, ntest, max_steps))



    ################################################################
    # load model
    ################################################################
    if args.model == "FNO":
        model = FNO2d(args.modes, args.modes, width=args.width,
                    n_channels=train_dataset.n_channels,
                    in_timesteps = args.T_in, out_timesteps=1, 
                    n_layers = args.n_layers).to(device)
    elif args.model == 'UNO':
        model = UNO( width=args.width, n_channels=train_dataset.n_channels, in_timesteps = args.T_in,  out_timesteps=1).to(device)
    elif args.model == 'wavelet_transformer':
        model = CrossWaveletTransformer(wave='haar', n_channels=train_dataset.n_channels, in_timesteps = args.T_in, dim=512, depth=8).to(device)
    else:
        raise NotImplementedError


    print(model)
    count_parameters(model)

    start_epoch = 0
    best_loss_epoch = 0
    if args.resume_path:
        print('Loading models and resume from {}'.format(model_path))
        args.resume_path = model_path
        model, optimizer, scheduler, start_epoch = resume_training_from_checkpoint(model, args.resume_path, device, optimizer=None, scheduler=None)
        print("resume training from epoch:", start_epoch)
        best_loss_epoch = start_epoch

    return model, test_loader, log_path

################################################################
# Function 1 Report Average step, step-wise, and full prediction relative l2 norm
################################################################
def predict_and_save(model, test_loader, save=False, log_path=None):
    """
    test_error(model, test_loader, save=False)
    Args:
        model: the model to test
        test_loader: the test loader
        save: whether to save the results to a pthfile
    Returns:
        save_data = {'input': torch.Tensor, 'output': torch.Tensor, 'pred': torch.Tensor} for computing the error
    """
    with torch.no_grad():
        model.eval()
        test_dataset = test_loader.dataset
        max_steps = test_dataset[0][1].shape[-2]

        save_data = {'input': [], 'output': [], 'pred': []}
        # autoregressive computing  
        for xx, yy in test_loader:
            save_data['input'].append(xx)
            xx = xx.to(device)
            yy = yy.to(device)
            xx = test_dataset.normalize_x(xx) # normalize the input before the autoregressive predicting
            for t in range(0, yy.shape[-2], args.T_bundle):
                im = model(xx)
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -2)
                xx = torch.cat((xx[..., args.T_bundle:,:], im), dim=-2)
            # denormalize the pred at the final step (get better results)   
            pred = test_dataset.denormalize_x(pred)    
        
            # # save the data to np_data
            # # print("save input and output shape", xx.shape, yy.shape)
            save_data['output'].append(yy)
            save_data['pred'].append(pred)

        # organzie np_data
        save_data['input'] = torch.cat(save_data['input'], axis=0)
        save_data['output'] = torch.cat(save_data['output'], axis=0)
        save_data['pred'] = torch.cat(save_data['pred'], axis=0)

        # print the shape of the np_data
        print("save_data shape", save_data['input'].shape, save_data['output'].shape, save_data['pred'].shape)

        # save to npz
        if save:
            torch.save(save_data, f'{log_path}/test_data.pth')

        return save_data


def tobe_tested_metrics(pred, target):
     # find the spectral band edges from the truth using OLD method
    target_reshape = rearrange(target, 'b h w t c -> (b t) h w c')
    k_low, k_high, freq_bins, cumulative_energy = get_frequency_bands_from_cumulative_energy(
        target_reshape, low_percentile=0.33, high_percentile=0.67
    )
    print(f"New method - k_low: {k_low}, k_high: {k_high} (discrete frequency bins)")
        # Show cumulative energy distribution
    print(f"\nCumulative energy at key points:")
    print(f"At k={k_low}: {cumulative_energy[k_low]:.3f}")
    print(f"At k={k_high}: {cumulative_energy[k_high]:.3f}")
    
    
    # Compute spectral errors by frequency bands for each time step
    print("\n=== Spectral Error Analysis by Frequency Bands ===")
    
    
    # First step spectral errors
    pred_first = rearrange(pred[..., [0], :], 'b h w t c -> (b t) h w c') 
    target_first = rearrange(target[..., [0], :], 'b h w t c -> (b t) h w c')
    low_err_first, mid_err_first, high_err_first = aggregate_spectral_energy_by_bands(
        pred_first, target_first, k_low, k_high
    )
    
    # Last step spectral errors
    pred_last = rearrange(pred[..., [-1], :], 'b h w t c -> (b t) h w c')
    target_last = rearrange(target[..., [-1], :], 'b h w t c -> (b t) h w c')
    low_err_last, mid_err_last, high_err_last = aggregate_spectral_energy_by_bands(
        pred_last, target_last, k_low, k_high
    )
    
    # Mean spectral errors across all time steps
    pred_mean = rearrange(pred, 'b h w t c -> (b t) h w c')
    target_mean = rearrange(target, 'b h w t c -> (b t) h w c')
    low_err_mean, mid_err_mean, high_err_mean = aggregate_spectral_energy_by_bands(
        pred_mean, target_mean, k_low, k_high
    )
    
    print(f"First Step - Low: {low_err_first:.6f}, Mid: {mid_err_first:.6f}, High: {high_err_first:.6f}")
    print(f"Last Step  - Low: {low_err_last:.6f}, Mid: {mid_err_last:.6f}, High: {high_err_last:.6f}")
    print(f"Mean Steps - Low: {low_err_mean:.6f}, Mid: {mid_err_mean:.6f}, High: {high_err_mean:.6f}")




def compute_evalutation_metrics(save_data, model_name='', log_path=''):
    pred, target = save_data['pred'], save_data['output'] # shape: (B, H, W, T, C)
    
    
    loss_dict = {}
    loss_dict['rel_l2_loss'] = RelL2Norm() # rel L2 loss
    loss_dict['rmse'] = RMSE()
    loss_dict['boundary_rmse'] = BoundaryRMSE()
    loss_dict['max_avg'] = MaxAbsError()
    loss_dict['max_global'] = GlobalMaxAbsError()
    
    step_dict = {0: "t=1", -1: "t=T"}

    # Standard error metrics
    print("\n=== Channel-wise Error Metrics ===")
    save_df = pd.DataFrame(columns=["step", "channel", "metric", f"{model_name}"])
    for step in [0, -1]: # first step and last step
        for c in range(pred.shape[-1]):
            # evaluate different metrics per channel
            for key, loss_func in loss_dict.items():
                # (B, H, W, T, C)
                loss_metric = loss_func(pred[..., step, c][:, :, :, None, None], target[..., step, c][:, :, :, None, None])
                print(f"Channel {c} {step_dict[step]} {key}: {loss_metric.item():.6f}")   
                new_row = pd.Series({"step": step_dict[step], "channel": c, "metric": key, f"{model_name}": loss_metric.item()}).to_frame().T
                save_df = pd.concat([save_df, new_row], ignore_index=True)
    print(save_df.head())
    save_df.to_csv(f"{log_path}/evalutation_metrics_{model_name}.csv", index=False)
    return loss_dict
    


if __name__ == '__main__':
    
    #### 1. predict and save the data
    model, test_loader, log_path = load_data_model(just_load_path=True)
    # save_data = predict_and_save(model, test_loader, save=True, log_path=log_path)
    
    #### 2. load the save_data
    save_data = torch.load(f'{log_path}/test_data.pth', map_location=device)
    
    #### 3. compute different types of metrics
    compute_evalutation_metrics(save_data, model_name=args.model, log_path=log_path)

    