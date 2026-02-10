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
from tqdm import tqdm
import random
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
# from LUCIE_inference import inference
from models.periodic_mswt import PeriodicMSWT2D_Patching, PeriodicMSWT2D_Patching_Efficient
from models.high_frequency_scaling import ResUNet
from models.torch_harmonics_local import *
from lucie_inference import inference
from data_utils.data_utils import load_data_era5
from utils.utils import LpLoss, save_checkpoint
import yaml





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



def train_model(model, train_loader, val_loader, optimizer, scheduler=None, nepochs=20, nfuture=0, num_examples=256, num_valid=8, reg_rate=0, save_dir='', save_name='', config=None, writer=None, start_ep=0):
    
    infer_bias = 1e+80
    recall_count = 0
    start_ep = start_ep
    pbar = tqdm(range(start_ep, nepochs), dynamic_ncols=True, smoothing=0.1)
    for epoch in pbar:
        
        optimizer.zero_grad()

        model.train()
        batch_num = 0
        train_loss = 0
        for inp, tar in train_loader:
            batch_num += inp.shape[0]
            loss = 0

            inp = inp.to(device)
            tar = tar.to(device)
            prd = model(inp)


            loss_delta = l2loss_sphere(prd[:,:5,:,:], tar[:,:5,:,:], relative=True)
            loss_tp = torch.mean((prd[:,5:,:,:]-tar[:,5:,:,:])**2)
            loss = loss_delta + loss_tp / tar.shape[1]

            lat_index = np.r_[7:15, 32:40]
            # lat_index = np.r_[0:48]
            # quad_weight_reg = quad_weights.reshape(1,1,48,1)[:,:,lat_index,:]
            out_fft = torch.mean(torch.abs(torch.fft.rfft(prd[:,:,lat_index,:],dim=3)),dim=2)
            target_fft = torch.mean(torch.abs(torch.fft.rfft(tar[:,:,lat_index,:],dim=3)),dim=2)
            loss_reg = 0.05 * torch.mean(torch.abs(out_fft - target_fft))

            if epoch > 150:
                loss = loss + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        pbar.set_description(f"Train Loss: {train_loss / batch_num:.4f}")
        if writer is not None:
            writer.add_scalar('train_loss', train_loss / batch_num, epoch)
        
        if epoch % 10 == 0:
            rollout_steps = 2920
            rollout = torch.tensor(inference(model, rollout_steps, data_inp[0:1].to(device), data_inp[:1460,-2:].to(device), 1, prog_means, prog_stds, diag_means, diag_stds, diff_stds)).to(device)
            rollout_clim = torch.mean(rollout[1460:],dim=0)
            clim_bias = torch.mean(torch.abs(rollout_clim - true_clim)/torch.abs(true_clim)) # relative bias
            print("epoch",epoch, "2 yearl rollout bias", clim_bias.item())
            if writer is not None:
                writer.add_scalar('eval_loss_rollout_clim_bias', clim_bias, epoch)
            if epoch > 60:
                print(f"Saving current model at epoch {epoch}")
                save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'],
                            model, 
                            epoch,
                            optimizer, scheduler)
                if clim_bias <= infer_bias:
                    infer_bias = clim_bias
                    print(f"Saving best model at epoch {epoch}")
                    save_checkpoint(config['train']['save_dir'],
                                    config['train']['save_name'].replace('.pt', f'_best.pt'),
                                    model, 
                                    epoch,
                                    optimizer, scheduler)
                    recall_count = 0
                # else:
                #     print(f"epoch {epoch} Loading model from {os.path.join(config['train']['save_dir'], config['train']['save_name'])}")
                #     ckpt = torch.load(os.path.join(config['train']['save_dir'], config['train']['save_name']), map_location=device)
                #     model.load_state_dict(ckpt['model'])
                #     recall_count += 1
                #     if recall_count > 3:
                #         print(f"Breaking at epoch {epoch}")
                #         break
                    



################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='FNO') # FNO, wavelet_transformer, HFS, UNet, HANO, UNO 
parser.add_argument('--dataset',type=str, default='era5')
parser.add_argument('--resume_path',type=str, default='')
parser.add_argument('--use_writer', action='store_true',default=False)
parser.add_argument('--seed', type=int, default=42)
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
parser.add_argument('--resume_training', action='store_true',default=False)
args = parser.parse_args()

config_file = args.config_path
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

with open(config_file, 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)
    config['train']['save_name'] = config['train']['save_name'].replace('.pt', f'_seed{args.seed}.pt')

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



train_set = TensorDataset(data_inp[:ntrain],data_tar[:ntrain])
val_set = TensorDataset(data_inp[ntrain:ntrain+nval],data_tar[ntrain:ntrain+nval])
num_workers = 16 if torch.cuda.is_available() else 0
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=num_workers)



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
    elif model_name == 'mswt_residual_sphere_efficient':
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
    print('model structure: ', model)

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])


    save_dir = config['train']['save_dir'] if torch.cuda.is_available() else 'saved_models'
    tensorboard_dir = config['train'].get('tensorboard_dir')
    if tensorboard_dir is None:
        tensorboard_dir = os.path.join(save_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    start_ep = 0
    if args.resume_training:
        ckpt_path = os.path.join(config['train']['save_dir'], config['train']['save_name'])
        if ckpt_path is not None and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            parsed_ep = ckpt['epoch'] 
            model.load_state_dict(ckpt['model'])
            if ckpt.get('optim') is not None:
                optimizer.load_state_dict(ckpt['optim'])
            if ckpt.get('scheduler') is not None:
                scheduler.load_state_dict(ckpt['scheduler'])
                sched_epoch = scheduler.state_dict().get('last_epoch', -1) + 1
                parsed_ep = max(parsed_ep, sched_epoch)
            start_ep = max(parsed_ep, 0)
            print(f'Weights loaded from {ckpt_path}, resuming at epoch {start_ep + 1}')
        else:
            print('resume_training requested but no checkpoint found; starting from scratch.')
    

    train_model(model, train_loader, val_loader, optimizer, scheduler=scheduler, nepochs=config['train']['epochs'], start_ep=start_ep, config=config, writer=writer)
    torch.save(model.state_dict(), 'model.pth')
