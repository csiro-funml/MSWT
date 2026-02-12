import argparse
import os
import numpy as np
from math import ceil, sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.utils.checkpoint import checkpoint
from torch.cuda import amp
import math
# from LUCIE_inference import inference
from models.periodic_mswt import PeriodicMSWT2D_Patching
from models.fno import FNO2d
from models.torch_harmonics_local import *
from lucie_inference import inference
from data_utils.data_utils import load_data_era5
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from models.high_frequency_scaling import ResUNet


def eval_model(model, data_inp, data_tar, prog_means, prog_stds, diag_means, diag_stds, diff_stds):
    rollout_steps = 2920
    rollout = torch.tensor(inference(model, rollout_steps, data_inp[0:1].to(device), data_inp[:1460,-2:].to(device), 1, prog_means, prog_stds, diag_means, diag_stds, diff_stds)).to(device)
    rollout_clim = torch.mean(rollout[1460:],dim=0)
    clim_bias = torch.mean(torch.abs(rollout_clim - true_clim))
    print("2 year rollout bias", clim_bias.item())
    return clim_bias


def out_of_sample_eval(model, data_inp, data_tar, prog_means, prog_stds, diag_means, diag_stds, diff_stds):
    forcing = data_inp[:1460,-2:]   # repeating tisr and constant oro
    # print(forcing.shape)
    rollout_step = 14600 # 
    initial_frame_idx = 16000+100
    forcing_initial_idx = (16000+100) % 1460 + 1
    rollout = torch.tensor(inference(model, rollout_step, data_inp[initial_frame_idx].unsqueeze(0).to(device), forcing.to(device), forcing_initial_idx, prog_means, prog_stds, diag_means, diag_stds, diff_stds)).to(device)
    return rollout, true_clim



def compute_evaluation_metrics(rollout_clim, true_clim, bias, save_dir):
    channels_list = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'precipitation']
    tropical_rollout = rollout_clim[:, 15:31] # (6, 16, 96)
    tropical_true = true_clim[:, 15:31] # (6, 16, 96)
    
    # for each channel, compute the following two metrics
    # mean absolute error
    tropical_mae = torch.mean(torch.abs(tropical_rollout - tropical_true), dim=(1, 2)).cpu().numpy()
    print("tropical_mae.shape: ", tropical_mae.shape, tropical_mae)
    
    # rmse
    tropical_rmse = torch.sqrt(torch.mean(torch.abs(tropical_rollout - tropical_true)**2, dim=(1, 2))).cpu().numpy()
    print("tropical_rmse.shape: ", tropical_rmse.shape, tropical_rmse)
    # 
    
    global_mae = torch.mean(torch.abs(rollout_clim - true_clim), dim=(1, 2)).cpu().numpy()
    print("global_mae.shape: ", global_mae.shape, global_mae)
    global_rmse = torch.sqrt(torch.mean(torch.abs(rollout_clim - true_clim)**2, dim=(1, 2))).cpu().numpy()
    print("global_rmse.shape: ", global_rmse.shape, global_rmse)
    
    df = np.array([tropical_mae, tropical_rmse, global_mae, global_rmse])
    print("df.shape: ", df.shape)

    df = pd.DataFrame(df)
    df['bias'] = bias
    # df = pd.DataFrame(df, index=['tropical_mae', 'tropical_rmse', 'global_mae', 'global_rmse'], columns=channels_list)

    df.to_csv(os.path.join(save_dir, 'evaluation_metrics.csv'))
    # save the plot 
    for channel_idx, channel in enumerate(channels_list):
        fig, axes = plt.subplots(3, 1)
        
        rollout_clim_channel = rollout_clim[channel_idx]
        true_clim_channel = true_clim[channel_idx]
        error_channel = rollout_clim_channel - true_clim_channel
        
        im0 = axes[0].imshow(rollout_clim_channel.cpu().numpy()[::-1, :], label='prediction', cmap='RdBu_r', origin='lower')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(true_clim_channel.cpu().numpy()[::-1, :], label='truth', cmap='RdBu_r', origin='lower')
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(error_channel.cpu().numpy()[::-1, :], label='error', cmap='RdBu_r', origin='lower')
        plt.colorbar(im2, ax=axes[2])
        
         
        fig.tight_layout()

        axes[0].set_title(f'{channel} Prediction')
        axes[1].set_title(f'{channel} Truth')
        axes[2].set_title(f'{channel} Error')
        
        os.makedirs(os.path.join(save_dir, 'evaluation_metrics'), exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'evaluation_metrics', f'{channel}_evaluation_metrics.png'))
        plt.close()
    return df


def evaluate_rollout(rollout, true_clim, save_dir, seed):
    channels_list = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'precipitation']
    rollout_temporal_mean = torch.mean(rollout,dim=0) # (C, H, W)
    print("rollout_temporal_mean.shape: ", rollout_temporal_mean.shape)
    save_df = {'Min bias': {}, 'Max bias': {}, 'Mean bias': {}, 'RMSE': {}}
    for channel_idx, channel in enumerate(channels_list):
        rollout_clim_channel = rollout_temporal_mean[channel_idx] # (H, W)
        true_clim_channel = true_clim[channel_idx] # (H, W)

        # run the conversion from 
        if channel == 'humidity':
            rollout_clim_channel = rollout_clim_channel * 1000
            true_clim_channel = true_clim_channel * 1000
        elif channel == 'surface_pressure':
            rollout_clim_channel = rollout_clim_channel / 100
            true_clim_channel = true_clim_channel / 100
        elif channel == 'precipitation':
            rollout_clim_channel = rollout_clim_channel * 4 * 1000
            true_clim_channel = true_clim_channel * 4 * 1000

        bias = rollout_clim_channel - true_clim_channel
        
        min_bias = bias.min().cpu().item()
        max_bias = bias.max().cpu().item()
        mean_bias = bias.mean().cpu().item()
        rmse = np.sqrt(F.mse_loss(rollout_clim_channel, true_clim_channel).item())
        save_df['Min bias'][channel] = min_bias
        save_df['Max bias'][channel] = max_bias
        save_df['Mean bias'][channel] = mean_bias
        save_df['RMSE'][channel] = rmse
    
    save_df = pd.DataFrame(save_df)
    print(save_df)
    # save_df.to_csv(os.path.join(save_dir, f'evaluation_metrics_seed{seed}.csv'))
    return save_df


def integrate_grid(ugrid, dimensionless=False, polar_opt=0):

    dlon = 2 * torch.pi / nlon
    radius = 1 if dimensionless else radius
    if polar_opt > 0:
        out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * quad_weights[polar_opt:-polar_opt] * dlon * radius**2, dim=(-2, -1))
    else:
        out = torch.sum(ugrid * quad_weights * dlon * radius**2, dim=(-2, -1))
    return out

def l2loss_sphere(prd, tar, relative=False, squared=True):
    loss = integrate_grid((prd - tar)**2, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / integrate_grid(tar**2, dimensionless=True).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss



################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='FNO') # FNO, wavelet_transformer, HFS, UNet, HANO, UNO 
parser.add_argument('--dataset',type=str, default='era5')
parser.add_argument('--resume_path',type=str, default='')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use_writer', action='store_true',default=False)
parser.add_argument('--load_high_res', action='store_true',default=False)

# ### FNO/UNO params 
parser.add_argument('--n_layers',type=int, default=8)
parser.add_argument('--modes', type=int, default=16)
# parser.add_argument('--modes', type=int, default=32)
parser.add_argument('--width', type=int, default=64)
# parser.add_argument('--use_ln',type=int, default=0)
parser.add_argument('--act',type=str, default='gelu')


###### optimizer and training setups
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--save_everyepoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb','lion'])
parser.add_argument('--beta1',type=float,default=0.9)
parser.add_argument('--beta2',type=float,default=0.9)
parser.add_argument('--lr_method',type=str, default='cossin') # cyclic for ViT perhaps
parser.add_argument('--grad_clip',type=float, default=10000.0)
parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='/scratch3/wan410/operator_learning_model/')
parser.add_argument('--config_path',type=str,default='config.yaml')
args = parser.parse_args()

config_file = args.config_path
with open(config_file, 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)
    config['train']['save_name'] = config['train']['save_name'].replace('.pt', f'_seed{args.seed}_best.pt')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Current working directory: {os.getcwd()}")
load_high_res = args.load_high_res
if torch.cuda.is_available():
    data_inp, data_tar, true_clim, prog_means, prog_stds, diag_means, diag_stds, diff_stds = load_data_era5(device, 
                                                                                                demo_index=np.arange(0, 100, 10) + 1,
                                                                                                load_high_res=load_high_res,
                                                                                                folder=config['data']['datapath'])
else:
    print("Using random data")
    data_inp = torch.randn(100,7, 48, 96).to(device)
    data_tar = torch.randn(100,6, 48, 96).to(device)
    true_clim = torch.randn(48, 7).to(device)
    prog_means = torch.randn(5, 7).to(device)
    prog_stds = torch.randn(5, 7).to(device)
    diag_means = torch.randn(1, 7).to(device)
    diag_stds = torch.randn(1, 7).to(device)
    diff_stds = torch.randn(1, 7).to(device)

ntrain = 16000
nval = 100


grid='legendre-gauss'
nlat = 48
nlon = 96
hard_thresholding_fraction = 0.9
lmax = ceil(nlat / 1)
mmax = lmax
modes_lat = int(nlat * hard_thresholding_fraction)
modes_lon = int(nlon//2 * hard_thresholding_fraction)
modes_lat = modes_lon = min(modes_lat, modes_lon)
radius=6.37122E6
cost, quad_weights = legendre_gauss_weights(nlat, -1, 1)
quad_weights = (torch.as_tensor(quad_weights).reshape(-1, 1)).to(device)

# model = FNO2d(modes1=[16, 16, 16, 16], modes2=[16, 16, 16, 16], fc_dim=128, layers=[64, 64, 64, 64, 64, 64], act='gelu',
#     in_dim=7, out_dim=6).to(device)

if __name__ == '__main__':
    model_cfg = config['model']
    model_name = model_cfg.get('name', 'fno').lower()
    print(f"Using model: {model_name}")
    if model_name == 'lucie':
        model = SphericalFourierNeuralOperatorNet(params = {}, spectral_transform='sht', filter_type = "linear", operator_type='dhconv', img_shape=(48, 96),
                num_layers=8, in_chans=7, out_chans=6, scale_factor=1, embed_dim=72, activation_function="silu", big_skip=True, pos_embed="latlon", use_mlp=True,
                                        normalization_layer="instance_norm", hard_thresholding_fraction=hard_thresholding_fraction,
                                        mlp_ratio = 2.).to(device)
    elif model_name == 'hfs':
        model = ResUNet(in_c=model_cfg.get('in_c', 7),
                        out_c=model_cfg.get('out_c', 6),
                        add_sphere_grid=model_cfg.get('add_sphere_grid', True),
                        target_params=model_cfg.get('target_params', 'small'),
                        ).to(device)
    elif model_name == 'mswt_sphere':
         model = PeriodicMSWT2D_Patching(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
            add_grid=model_cfg.get('add_grid', False),
            add_sphere_grid=model_cfg.get('add_sphere_grid', True),
            patch_size=model_cfg.get('patch_size', None),
            local_attention_size=model_cfg.get('local_attention_size', None),
        ).to(device)
    elif model_name == 'mswt_patch_sphere':
        model = PeriodicMSWT2D_Patching(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
            add_grid=model_cfg.get('add_grid', False),
            add_sphere_grid=model_cfg.get('add_sphere_grid', True),
            patch_size=model_cfg.get('patch_size', None),
            local_attention_size=model_cfg.get('local_attention_size', None),
            residual_connection=model_cfg.get('residual_connection', False),
        ).to(device)
    else:
        raise ValueError(f'Model {model_name} not supported')
    print("number of parameters: ", sum(p.numel() for p in model.parameters()))
    # print('model structure: ', model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    # scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)


    save_dir = config['train']['save_dir'] if torch.cuda.is_available() else 'saved_models'
    tensorboard_dir = config['train'].get('tensorboard_dir')
    
    ckpt_path = os.path.join(save_dir, config['train']['save_name'])
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Model loaded from {ckpt_path}")
    else:
        print(f"Model not found at {ckpt_path}, use random weights")
        
    # clim_bias = eval_model(model, data_inp, data_tar, prog_means, prog_stds, diag_means, diag_stds, diff_stds)
    # print("2 year rollout bias", clim_bias.item())

    load_results = False
    load_path  = os.path.join(save_dir, f'rollout_model_{model_name}_seed{args.seed}.pt')
    if load_results and os.path.exists(load_path):
        rollout = torch.load(load_path)
        true_clim = torch.load(os.path.join(save_dir, 'true_clim.pt'))
    else:
        rollout, true_clim = out_of_sample_eval(model, data_inp, data_tar, prog_means, prog_stds, diag_means, diag_stds, diff_stds)
        torch.save(rollout, load_path)
        torch.save(true_clim, os.path.join(save_dir, 'true_clim.pt'))   
    
    evaluate_rollout(rollout, true_clim, save_dir, args.seed)

