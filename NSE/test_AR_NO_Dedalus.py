# TEST THE Predictions of the model

import sys
import os
# Add parent directory to Python path to access utils and models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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

from utils.griddataset import MixedTemporalDataset, TemporalDataset2D, LocalTemporalDataset2D, MemmapDedalusDataset2D, MemmapDedalusBigDataset2D
from utils.make_master_file import DATASET_DICT
# from models.fno import FNO2d
from models.fno import FNO2d_Tin1_Tout1 as FNO2d
from models.wavelet_transform import CrossWaveletTransformer, CrossWaveletTransSkipConnection
from models.wavelet_transform_exploration import WaveletTransformer
from models.high_frequency_scaling import ResUNet
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as stats
from utils.criterion import RelL2Norm, RMSE, BoundaryRMSE, MaxAbsError, GlobalMaxAbsError, SpectralError, Energy_Enstropy_SpectrumError
from visualizations import plot_enstrophy_spectrum, spectrum_2d
from utils.compute_physical_statistics import compute_spectra
from utils.compute_diagnostics import streamfunction_to_velocity
warnings.filterwarnings("ignore")

################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='FNO') # FNO, wavelet_transformer, HFS, UNet, HANO, UNO 
parser.add_argument('--dataset',type=str, default='ns2d_dedalus_big') # ['ns2d_fno_1e-3', 'ns2d_pda', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 'sw2d_pda'], note: pdb is the pde bench
parser.add_argument('--dataset_type',type=str, default='long', choices=['long', 'short'])  # Test-specific: for choosing test vs long test dataset
parser.add_argument('--resume_path',type=str, default='')
parser.add_argument('--use_writer', action='store_true',default=False)
parser.add_argument('--form',type=str, default='vorticity', choices=['vorticity', 'velocity'])


# ### dataset details
parser.add_argument('--T_in', type=int, default=1)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--pad', type=int, default=0)
parser.add_argument('--normalize',type=int, default=1)
parser.add_argument('--normalize_strategy',type=str, default='zscore')
parser.add_argument('--num_steps', type=int, default=300)
parser.add_argument('--num_animation_frames', type=int, default=None, help='Desired number of frames in animation. If None, uses all steps. If specified, calculates step interval automatically.')

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
parser.add_argument('--epochs', type=int, default=2000)  # Needed to construct model path
parser.add_argument('--loss_type', type=str, default='fourier', choices=['fourier', 'rel_l2'])
parser.add_argument('--fourier_logscale', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--warmup_epochs',type=int, default=100)

# Performance optimization arguments (for consistency with training script)
parser.add_argument('--num_workers', type=int, default=None, help='Number of DataLoader workers (default: auto-detect)')
parser.add_argument('--pin_memory', action='store_true', default=True, help='Pin memory for faster CPU->GPU transfers')
parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches to prefetch')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps (not used in testing but kept for consistency)')

parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='/scratch3/wan410/operator_learning_model/')

args = parser.parse_args()
args.fourier_logscale = args.fourier_logscale == 'True'


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Current working directory: {os.getcwd()}")


def load_data_model(just_load_path=False):
    ################################################################
    # load some toy data to run locally
    if not torch.cuda.is_available() and args.dataset != 'ns2d_dedalus_big':
        train_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, n_channels=3, normalize=args.normalize, train='train')
        test_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels=3, normalize=args.normalize, train='test')
    elif args.dataset == 'ns2d_dedalus_big':
        # load data and dataloader for big dedalus dataset
        train_dataset = MemmapDedalusBigDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form=args.form, normalize=args.normalize, train='train', strategy=args.normalize_strategy)
        if args.dataset_type == 'long':
            test_dataset = MemmapDedalusBigDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form=args.form, normalize=args.normalize, train='test_long', strategy=args.normalize_strategy)
        else:
            test_dataset = MemmapDedalusBigDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form=args.form, normalize=args.normalize, train='test', strategy=args.normalize_strategy)
    elif args.dataset != 'ns2d_dedalus':
        # load data and dataloader
        train_dataset = TemporalDataset2D(args.dataset, t_in = args.T_in, t_ar = args.T_ar, train='train', normalize=args.normalize)
        test_dataset = TemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels = train_dataset.n_channels, train='test', normalize=args.normalize)
        # test_dataset = TemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels = train_dataset.n_channels, train='test_long', normalize=args.normalize)
    else:
        # load data and dataloader
        train_dataset = MemmapDedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form='vorticity', normalize=args.normalize, train='train')
        test_dataset = MemmapDedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form='vorticity', normalize=args.normalize, train='test')
    
    ntrain, ntest = len(train_dataset), len(test_dataset)
    ntrain = 5200 if args.dataset == 'ns2d_pda' else ntrain # for testing
    print(args.dataset)
    print('Train num {}, Test num {}'.format(ntrain, ntest))

    # Determine number of workers
    if args.num_workers is None:
        if torch.cuda.is_available():
            num_workers = min(os.cpu_count() or 8, 16)
        else:
            num_workers = 0
    else:
        num_workers = args.num_workers
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.pin_memory if torch.cuda.is_available() else False,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=0 if not torch.cuda.is_available() else 8, pin_memory=torch.cuda.is_available()) # TODO: removed later, jsut to test the traning set results
    # val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)

    # ntrain, ntest = len(train_dataset), len(test_dataset)

    if not args.pad:
        args.res = train_dataset.res  # use original dataset  resolution to train the model

    testing_mode = 'FNO_testing'
    if testing_mode == 'FNO_testing':
        if args.dataset == 'ns2d_dedalus_big':
            comment = args.comment + '{}_{}_mod{}_wid{}_lay{}_ntrain{}_normalizer_{}_form_{}'.format(args.dataset, args.model, args.modes, args.width, args.n_layers, ntrain, args.normalize_strategy, args.form)
            # comment = args.comment + f'{args.dataset}_{args.model}_mod{args.modes}_wid{args.width}_lay{args.n_layers}_ntrain{ntrain}_form{args.form}_loss{args.loss_type}_logscale{args.fourier_logscale}_warmup{args.warmup_epochs}'
            # comment = args.comment + '{}_{}_mod{}_wid{}_lay{}_ntrain{}_normalizer_{}_form_{}'.format(args.dataset, args.model, args.modes, args.width, args.n_layers, ntrain, args.normalize_strategy, args.form)
        else:
            comment = args.comment + '{}_{}_mod{}_wid{}_lay{}_ntrain{}_normalizer_{}'.format(args.dataset, args.model, args.modes, args.width, args.n_layers, ntrain, args.normalize_strategy)
        log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
        # model_path = log_path + '/model.pth'
        model_path = log_path + f'/model_epochs_{args.epochs}.pth' # I will test a longer training epoch
    else:
        comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
        log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
        # model_path = log_path + '/model.pth'
        model_path = log_path + f'/model_epochs_{args.epochs}.pth' # I will test a longer training epoch
    # model_path = log_path + '/model.pth' # for testing, note: to be deleted later
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
        if args.dataset == 'ns2d_dedalus_big':
            model = FNO2d(args.modes, args.modes, width=args.width,
                        img_size=train_dataset.res,
                        in_channels=train_dataset.n_channels_in[args.form],
                        out_channels=train_dataset.n_channels_out[args.form],
                        in_timesteps = args.T_in, out_timesteps=1, 
                        n_layers = args.n_layers,
                        use_ln=True,
                        ).to(device)
        else:
            model = FNO2d(args.modes, args.modes, width=args.width,
                        n_channels=train_dataset.n_channels,
                        in_timesteps = args.T_in, out_timesteps=1, 
                        n_layers = args.n_layers).to(device)
    elif args.model == 'UNO':
        model = UNO( width=args.width, n_channels=train_dataset.n_channels, in_timesteps = args.T_in,  out_timesteps=1).to(device)
    elif args.model == 'wavelet_transformer':
        model = CrossWaveletTransformer(wave='haar', n_channels=train_dataset.n_channels, in_timesteps = args.T_in, dim=512, depth=8).to(device)
    elif args.model == 'HFS':
        model =  ResUNet(in_c = train_dataset.n_channels * args.T_in + 2 ,out_c = train_dataset.n_channels, 
                 bottleneck_feature=512, 
                 device=device).to(device)
    elif args.model == 'wavelet_transformer_skip':
        model = CrossWaveletTransSkipConnection(wave='haar', n_channels=train_dataset.n_channels, in_timesteps = args.T_in, dim=512, depth=8).to(device)
    elif args.model == 'WaveletTransV2':
        model = WaveletTransformer(in_timesteps = args.T_in, 
        in_chans=train_dataset.n_channels, out_chans=train_dataset.n_channels
        ,output_size=(train_dataset.res[0], train_dataset.res[1])).to(device)
    else:
        print("model not implemented", args.model)
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

def autoregressive_prediction(model=None, test_loader=None, max_steps=None, load=True):
    if load:
        data_path = f'{log_path}/test_data_prediction_{args.dataset_type}.pth'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        save_data = torch.load(data_path, map_location='cpu')
        pred = save_data['pred']
        target = save_data['output']
        return pred, target
    with torch.no_grad():
        model.eval()
        test_dataset = test_loader.dataset
        total_steps = test_dataset.n_size if max_steps is None else min(max_steps, test_dataset.n_size)
        pred = []
        target = []
        for i in tqdm(range(total_steps)):
            xx = test_loader.dataset[i][0] # (H, W, T_in, C_in) # ground truth input
            yy = test_loader.dataset[i][1] # (H, W, T_ar, C_ar) # ground truth output

            if i == 0:
                # load the initial state
                # xx shape: (H, W, T_in, C_in) where C_in includes forcing
                # Model expects: (B, H, W, 1, C_in), so we need to add batch dim and use last timestep if T_in > 1
                x_input = xx.to(device)  # (H, W, T_in, C_in)
                if x_input.shape[-2] > 1:
                    # Use the last timestep if T_in > 1
                    x_input = x_input[..., -1:, :]  # (H, W, 1, C_in)
                # Add batch dimension
                x_input = x_input.unsqueeze(0)  # (1, H, W, 1, C_in)
                # normalize it before the autoregressive predicting
                x_input = test_dataset.normalize_x(x_input)
                y_pred = model(x_input)  # (1, H, W, 1, C_out)
                # Remove batch dimension for denormalization
                y_pred = y_pred.squeeze(0)  # (H, W, 1, C_out)
                y_pred = test_dataset.denormalize_x(y_pred)  # (H, W, 1, C_out)
                pred.append(y_pred)
                target.append(yy)
                # Store predicted main variables (without forcing) for next step
                x_next = y_pred  # (H, W, 1, C_out) where C_out = main variables only
            else:
                # For subsequent steps: combine predicted main variables with ground truth forcing
                xx = xx.to(device)  # (H, W, T_in, C_in)
                # Get ground truth forcing from the last timestep of xx
                # xx[..., -1, -2:] selects last timestep and last 2 channels (forcing_x, forcing_y)
                forcing = xx[..., -1, -2:]  # (H, W, 2)
                # Add time dimension to forcing
                forcing = forcing.unsqueeze(-2)  # (H, W, 1, 2)
                
                # x_next is (H, W, 1, C_out) where C_out = main variables (e.g., 2 for vorticity form)
                # Concatenate predicted main variables with forcing along channel dimension
                # Result should be (H, W, 1, C_in) where C_in = C_out + 2
                x_input = torch.cat((x_next, forcing), dim=-1)  # (H, W, 1, C_in)
                # Add batch dimension
                x_input = x_input.unsqueeze(0)  # (1, H, W, 1, C_in)
                # Normalize before model call
                x_input = test_dataset.normalize_x(x_input)
                y_pred = model(x_input)  # (1, H, W, 1, C_out)
                # Remove batch dimension
                y_pred = y_pred.squeeze(0)  # (H, W, 1, C_out)
                y_pred = test_dataset.denormalize_x(y_pred)  # (H, W, 1, C_out)
                pred.append(y_pred)
                target.append(yy)
                x_next = y_pred  # Update for next iteration
        
        pred = torch.stack(pred, dim=0).cpu()
        target = torch.stack(target, dim=0).cpu()
        save_data = {
            'pred': pred,
            'output': target,
        }
        torch.save(save_data, f'{log_path}/test_data_prediction_{args.dataset_type}.pth')
        print(f"Saved prediction data and metrics to {log_path}/test_data_prediction_{args.dataset_type}.pth")
        return pred, target


def predict_and_save(model=None, test_loader=None, save=False, log_path=None, max_steps=None):
    """
    test_error(model, test_loader, save=False)
    Args:
        model: the model to test
        test_loader: the test loader
        save: whether to save the results to a pthfile
    Returns:
        save_data = {'input': torch.Tensor, 'output': torch.Tensor, 'pred': torch.Tensor} for computing the error
    """
    pred, target = autoregressive_prediction(model, test_loader, max_steps, load=False)  # unnormalized prediction and target
    print("pred shape", pred.shape, "target shape", target.shape)
    
    # Domain size (default to 2*pi, can be adjusted if needed)
    Lx = 2 * np.pi
    Ly = 2 * np.pi
    
    # Get dataset form to determine how to extract velocity
    # dataset_form = getattr(test_dataset, 'form', 'vorticity')
    dataset_form = 'velocity'
    # Initialize metrics lists
    rel_l2_loss_list = []
    spectral_data_list = []
    
    # Initialize loss function, comment if you don't need it
    rel_l2_loss_fn = RelL2Norm()
    
    # Compute metrics for each step
    # pred and target shape: (N, H, W, T, C) where N is number of steps
    # For each step, we have one timestep (T=1) in the output
    num_steps = pred.shape[0] # 1 for batch size
    
    for step_idx in range(num_steps):
        # Get prediction and target for this step
        # pred[step_idx]: (H, W, T, C), target[step_idx]: (H, W, T, C)
        pred_step = pred[step_idx]  # (H, W, T, C)
        target_step = target[step_idx]  # (H, W, T, C)
        
        # Compute rel_l2_loss for this step
        # Add batch dimension for loss computation: (1, H, W, T, C)
        pred_step_batch = pred_step.unsqueeze(0)
        target_step_batch = target_step.unsqueeze(0)
        rel_l2_err_step = rel_l2_loss_fn(pred_step_batch, target_step_batch)
        rel_l2_loss_list.append({
            'time_step': step_idx,
            'rel_l2_error': rel_l2_err_step.item()
        })
        
        # Compute spectral energy and enstrophy
        # Extract velocity components based on form
        # For each timestep in the output (usually T=1)
        for t_idx in tqdm(range(pred_step.shape[-2]), desc="Computing spectral metrics"):  # T dimension
            pred_t = pred_step[..., t_idx, :].numpy()  # (H, W, C)
            target_t = target_step[..., t_idx, :].numpy()  # (H, W, C)
            
            # Get velocity components
            if dataset_form == 'vorticity':
                # For vorticity form: channels are [vorticity, streamfunction]
                # Extract streamfunction (channel 1) and compute velocity
                if pred_t.shape[-1] >= 2:
                    psi_pred = pred_t[..., 1]  # streamfunction
                    psi_target = target_t[..., 1]
                    ux_pred, uy_pred = streamfunction_to_velocity(psi_pred, Lx, Ly)
                    ux_target, uy_target = streamfunction_to_velocity(psi_target, Lx, Ly)
                else:
                    continue
            elif dataset_form == 'velocity':
                # For velocity form: channels are [pressure, velocity_x, velocity_y]
                if pred_t.shape[-1] >= 3:
                    ux_pred = pred_t[..., 1]  # velocity_x
                    uy_pred = pred_t[..., 2]  # velocity_y
                    ux_target = target_t[..., 1]
                    uy_target = target_t[..., 2]
                else:
                    continue
            else:
                # Unknown form, skip spectral computation
                continue
            
            # compute the energy
            # Energy: E = (1/2) ∫ u² dx / Area
            Nx, Ny = ux_pred.shape[0], ux_pred.shape[1]
            area = Lx * Ly
            energy_pred = 0.5 * np.sum(ux_pred**2 + uy_pred**2) * area / (Nx * Ny)
            energy_target = 0.5 * np.sum(ux_target**2 + uy_target**2) * area / (Nx * Ny)

            # Compute spectra
            try:
                k_bins, Ek_pred, Zk_pred = compute_spectra(ux_pred, uy_pred, Lx, Ly)
                _, Ek_target, Zk_target = compute_spectra(ux_target, uy_target, Lx, Ly)
                

                # Get grid resolution H from velocity field shape
                H = ux_pred.shape[0]  # Grid resolution in y direction
                
                spectral_data_list.append({
                    'time_step': step_idx,
                    'timestep_in_output': t_idx,
                    'k_bins': k_bins,
                    'Ek_pred': Ek_pred,
                    'Ek_target': Ek_target,
                    'Zk_pred': Zk_pred,
                    'Zk_target': Zk_target,
                    'energy_pred': energy_pred,
                    'energy_target': energy_target,
                    'H': H,  # Store grid resolution for Nyquist truncation
                    'Lx': Lx  # Store domain size for Nyquist truncation
                })
            except Exception as e:
                print(f"Warning: Failed to compute spectra at step {step_idx}, timestep {t_idx}: {e}")
                import traceback
                traceback.print_exc()
    
    # Compute overall error
    pred_batch = pred.unsqueeze(0)  # (1, N, H, W, T, C)
    target_batch = target.unsqueeze(0)  # (1, N, H, W, T, C)
    rel_l2_err = rel_l2_loss_fn(pred_batch, target_batch)
    print("overall rel_l2_error", rel_l2_err.item())
    
    # Create save_data dictionary with all metrics
    save_data = {
        'pred': pred,  # (N, H, W, T, C)
        'output': target,  # (N, H, W, T, C)
        'rel_l2_loss_by_step': rel_l2_loss_list,  # List of dicts with time_step and rel_l2_error
        'spectral_data_by_step': spectral_data_list,  # List of dicts with spectral data
        'dataset_form': dataset_form,  # Store form for later use
        'domain_size': {'Lx': Lx, 'Ly': Ly},  # Store domain size
    }
    
    # Save to file if requested
    if save and log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        torch.save(save_data, f'{log_path}/test_data_prediction_{args.dataset_type}.pth')
        print(f"Saved prediction data and metrics to {log_path}/test_data_prediction_{args.dataset_type}.pth")

    # save the data to npz
    if save and log_path is not None:
        save_data_np = {
            'pred': save_data['pred'].numpy(),
            'output': save_data['output'].numpy(),
            'rel_l2_loss_by_step': save_data['rel_l2_loss_by_step'],
            'spectral_data_by_step': save_data['spectral_data_by_step'],
            'dataset_form': save_data['dataset_form'],
            'domain_size': save_data['domain_size'],
        }
        os.makedirs(log_path, exist_ok=True)
        np.savez(f'{log_path}/test_data_prediction_{args.dataset_type}.npz', **save_data_np)
        print(f"Saved prediction data and metrics to {log_path}/test_data_prediction_{args.dataset_type}.npz")
        
        return save_data


################################################################
# Animation Functions
################################################################
def animate_predictions(save_data, log_path=None, save_animation=True, fps=10, num_steps=None, num_animation_frames=None):
    """
    Create animation showing target, prediction, and error for each channel.
    
    Args:
        save_data: Dictionary containing 'pred', 'output' tensors and other metadata
        log_path: Path to save animation
        save_animation: Whether to save animation file
        fps: Frames per second for animation
        num_steps: Maximum number of steps to use (None = use all)
        num_animation_frames: Desired number of frames in animation (None = use all steps)
                             If specified, calculates step interval automatically
    """
    pred = save_data['pred']  # (N, H, W, T, C)
    target = save_data['output']  # (N, H, W, T, C)
    
    # Get total available steps
    total_steps = pred.shape[0]
    max_steps = total_steps if num_steps is None else min(num_steps, total_steps)
    
    # Calculate step interval if num_animation_frames is specified
    if num_animation_frames is not None and num_animation_frames > 0:
        step_interval = max(1, max_steps // num_animation_frames)
        num_steps = min(num_animation_frames, max_steps // step_interval)
        step_indices = np.arange(0, max_steps, step_interval)[:num_steps]
        print(f"Using {num_steps} frames with step interval {step_interval} (from {max_steps} total steps)")
    else:
        num_steps = max_steps
        step_indices = np.arange(num_steps)
        print(f"Using all {num_steps} steps for animation")
    
    H, W = pred.shape[1], pred.shape[2]
    num_channels = pred.shape[-1]
    
    # Select steps based on step_indices
    pred = pred[step_indices]  # (num_steps, H, W, T, C)
    target = target[step_indices]  # (num_steps, H, W, T, C)
    
    # Use first timestep in output (usually T=1)
    pred = pred[..., 0, :]  # (N, H, W, C)
    target = target[..., 0, :]  # (N, H, W, C)
    
    # Convert to numpy
    pred = pred.numpy()
    target = target.numpy()
    
    # Compute error
    error = pred - target  # (N, H, W, C)
    
    # Find min/max per channel for consistent colorbar across all frames
    # Use same range for target and prediction for better comparison within each channel
    vmin_common = []
    vmax_common = []
    vmin_error = []
    vmax_error = []
    
    for col in range(num_channels):
        # Common range for target and prediction (per channel)
        vmin_ch = min(target[:, :, :, col].min(), pred[:, :, :, col].min())
        vmax_ch = max(target[:, :, :, col].max(), pred[:, :, :, col].max())
        vmin_common.append(vmin_ch)
        vmax_common.append(vmax_ch)
        
        # Symmetric range for error (per channel)
        vmin_err_ch = error[:, :, :, col].min()
        vmax_err_ch = error[:, :, :, col].max()
        vmax_error_abs_ch = max(abs(vmin_err_ch), abs(vmax_err_ch))
        vmin_error.append(-vmax_error_abs_ch)
        vmax_error.append(vmax_error_abs_ch)
    
    # Create figure with 3 rows (target, pred, error) and num_channels columns
    fig, axes = plt.subplots(3, num_channels, figsize=(5*num_channels, 12))
    if num_channels == 1:
        axes = axes.reshape(-1, 1)
    
    # Channel names based on dataset form
    dataset_form = save_data.get('dataset_form', 'vorticity')
    if dataset_form == 'vorticity':
        channel_names = ['Vorticity', 'Streamfunction']
    elif dataset_form == 'velocity':
        channel_names = ['Pressure', 'Velocity X', 'Velocity Y']
    else:
        channel_names = [f'Channel {i}' for i in range(num_channels)]
    
    # Initialize images
    imgs = []
    for row in range(3):
        row_imgs = []
        for col in range(num_channels):
            ax = axes[row, col]
            if row == 0:  # Target
                im = ax.imshow(target[0, :, :, col], cmap='RdBu_r', 
                              vmin=vmin_common[col], vmax=vmax_common[col], origin='lower')
                ax.set_title(f'{channel_names[col] if col < len(channel_names) else f"Channel {col}"}\nTarget')
            elif row == 1:  # Prediction
                im = ax.imshow(pred[0, :, :, col], cmap='RdBu_r',
                              vmin=vmin_common[col], vmax=vmax_common[col], origin='lower')
                ax.set_title(f'Prediction')
            else:  # Error
                im = ax.imshow(error[0, :, :, col], cmap='RdBu_r',
                              vmin=vmin_error[col], vmax=vmax_error[col], origin='lower')
                ax.set_title(f'Error')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
            row_imgs.append(im)
        imgs.append(row_imgs)
    
    # Add step counter text
    step_text = fig.suptitle('Step: 0', fontsize=16, y=0.98)
    
    def animate(frame):
        # Get actual step index (for display purposes)
        actual_step = step_indices[frame] if frame < len(step_indices) else frame
        step_text.set_text(f'Step: {actual_step} (frame {frame})')
        for row in range(3):
            for col in range(num_channels):
                if row == 0:  # Target
                    imgs[row][col].set_data(target[frame, :, :, col])
                elif row == 1:  # Prediction
                    imgs[row][col].set_data(pred[frame, :, :, col])
                else:  # Error
                    imgs[row][col].set_data(error[frame, :, :, col])
        return [img for row in imgs for img in row] + [step_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=num_steps, 
                                  interval=1000/fps, blit=False, repeat=True)
    
    # Save animation
    if save_animation and log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        # Include actual step range in filename
        if num_animation_frames is not None:
            step_range = f"steps{step_indices[0]}-{step_indices[-1]}_interval{step_interval}_frames{num_steps}"
        else:
            step_range = f"totalsteps{num_steps}"
        anim_path = f'{log_path}/prediction_animation_{step_range}.mp4'
        print(f"Saving animation to {anim_path}...")
        print(f"  Animation details: {num_steps} frames, {fps} fps, bitrate=1800")
        start_time = time.time()
        anim.save(anim_path, writer='ffmpeg', fps=fps, bitrate=1800)
        elapsed_time = time.time() - start_time
        print(f"Animation saved to {anim_path} (took {elapsed_time:.2f} seconds, ~{elapsed_time/60:.2f} minutes)")
    
    return anim, fig


def animate_spectral_comparison(save_data, log_path=None, save_animation=True, 
                                fps=10, k_zoom_threshold=20, num_steps=None, num_animation_frames=None):
    """
    Create animation comparing spectral energy and enstrophy between target and prediction.
    Includes zoomed view for high frequency components (k > k_zoom_threshold).
    
    Args:
        save_data: Dictionary containing spectral data
        log_path: Path to save animation
        save_animation: Whether to save animation file
        fps: Frames per second for animation
        k_zoom_threshold: Wavenumber threshold for zoomed view
        num_steps: Maximum number of steps to use (None = use all)
        num_animation_frames: Desired number of frames in animation (None = use all steps)
                             If specified, calculates step interval automatically
    """
    spectral_data_list = save_data['spectral_data_by_step']
    
    if not spectral_data_list:
        print("Warning: No spectral data found in save_data")
        return None, None
    
    # Get total available steps
    total_steps = len(spectral_data_list)
    max_steps = total_steps if num_steps is None else min(num_steps, total_steps)
    
    # Calculate step interval if num_animation_frames is specified
    step_interval = None
    if num_animation_frames is not None and num_animation_frames > 0:
        step_interval = max(1, max_steps // num_animation_frames)
        num_steps = min(num_animation_frames, max_steps // step_interval)
        step_indices = np.arange(0, max_steps, step_interval)[:num_steps]
        spectral_data_list = [spectral_data_list[i] for i in step_indices]
        print(f"Using {num_steps} frames with step interval {step_interval} (from {max_steps} total steps)")
    else:
        num_steps = max_steps
        spectral_data_list = spectral_data_list[:num_steps]
        print(f"Using all {num_steps} steps for animation")
    
    # Extract all k_bins (should be the same for all steps)
    k_bins = spectral_data_list[0]['k_bins']
    
    # Get H and Lx for Nyquist truncation (should be the same for all steps)
    H = spectral_data_list[0].get('H', k_bins.shape[0])  # Fallback to k_bins length if not stored
    Lx = spectral_data_list[0].get('Lx', 2 * np.pi)  # Default to 2*pi if not stored
    
    # Compute Nyquist truncation index (same as utilities.py)
    k_nyquist = int((np.pi * H) // Lx)
    start_truth = 1  # Skip k_bins[0] as in utilities.py
    
    # Find global min/max for consistent y-axis (using truncated range)
    all_Ek_target = [data['Ek_target'][start_truth:k_nyquist] for data in spectral_data_list]
    all_Ek_pred = [data['Ek_pred'][start_truth:k_nyquist] for data in spectral_data_list]
    all_Zk_target = [data['Zk_target'][start_truth:k_nyquist] for data in spectral_data_list]
    all_Zk_pred = [data['Zk_pred'][start_truth:k_nyquist] for data in spectral_data_list]
    
    Ek_max = max([np.max(Ek) for Ek in all_Ek_target + all_Ek_pred if len(Ek) > 0])
    Ek_min = min([np.min(Ek[Ek > 0]) for Ek in all_Ek_target + all_Ek_pred if len(Ek) > 0 and np.any(Ek > 0)])
    
    Zk_max = max([np.max(Zk) for Zk in all_Zk_target + all_Zk_pred if len(Zk) > 0])
    Zk_min = min([np.min(Zk[Zk > 0]) for Zk in all_Zk_target + all_Zk_pred if len(Zk) > 0 and np.any(Zk > 0)])
    
    # Find index for zoom threshold (within truncated range)
    k_zoom_idx = np.where(k_bins[start_truth:k_nyquist] > k_zoom_threshold)[0]
    if len(k_zoom_idx) > 0:
        zoom_start_idx = k_zoom_idx[0] + start_truth  # Adjust for start_truth offset
    else:
        zoom_start_idx = min(k_nyquist - 10, len(k_bins) - 10)  # Fallback within truncated range
    
    # Create figure with subplots
    # Top row: Full spectrum for energy and enstrophy
    # Bottom row: Zoomed view for high frequencies
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Full energy spectrum
    ax_energy_full = fig.add_subplot(gs[0, 0])
    ax_energy_full.set_xlabel('Wavenumber k')
    ax_energy_full.set_ylabel('Energy E(k)')
    ax_energy_full.set_title('Energy Spectrum (Full)')
    ax_energy_full.set_xscale('log')
    ax_energy_full.set_yscale('log')
    ax_energy_full.grid(True, alpha=0.3)
    line_Ek_target_full, = ax_energy_full.plot([], [], 'b-', label='Target', linewidth=2)
    line_Ek_pred_full, = ax_energy_full.plot([], [], 'r--', label='Prediction', linewidth=2)
    ax_energy_full.legend()
    
    # Zoomed energy spectrum
    ax_energy_zoom = fig.add_subplot(gs[1, 0])
    ax_energy_zoom.set_xlabel('Wavenumber k')
    ax_energy_zoom.set_ylabel('Energy E(k)')
    ax_energy_zoom.set_title(f'Energy Spectrum (Zoom: k > {k_zoom_threshold})')
    ax_energy_zoom.set_xscale('log')
    ax_energy_zoom.set_yscale('log')
    ax_energy_zoom.grid(True, alpha=0.3)
    line_Ek_target_zoom, = ax_energy_zoom.plot([], [], 'b-', label='Target', linewidth=2)
    line_Ek_pred_zoom, = ax_energy_zoom.plot([], [], 'r--', label='Prediction', linewidth=2)
    ax_energy_zoom.legend()
    
    # Full enstrophy spectrum
    ax_enstrophy_full = fig.add_subplot(gs[0, 1])
    ax_enstrophy_full.set_xlabel('Wavenumber k')
    ax_enstrophy_full.set_ylabel('Enstrophy Z(k)')
    ax_enstrophy_full.set_title('Enstrophy Spectrum (Full)')
    ax_enstrophy_full.set_xscale('log')
    ax_enstrophy_full.set_yscale('log')
    ax_enstrophy_full.grid(True, alpha=0.3)
    line_Zk_target_full, = ax_enstrophy_full.plot([], [], 'b-', label='Target', linewidth=2)
    line_Zk_pred_full, = ax_enstrophy_full.plot([], [], 'r--', label='Prediction', linewidth=2)
    ax_enstrophy_full.legend()
    
    # Zoomed enstrophy spectrum
    ax_enstrophy_zoom = fig.add_subplot(gs[1, 1])
    ax_enstrophy_zoom.set_xlabel('Wavenumber k')
    ax_enstrophy_zoom.set_ylabel('Enstrophy Z(k)')
    ax_enstrophy_zoom.set_title(f'Enstrophy Spectrum (Zoom: k > {k_zoom_threshold})')
    ax_enstrophy_zoom.set_xscale('log')
    ax_enstrophy_zoom.set_yscale('log')
    ax_enstrophy_zoom.grid(True, alpha=0.3)
    line_Zk_target_zoom, = ax_enstrophy_zoom.plot([], [], 'b-', label='Target', linewidth=2)
    line_Zk_pred_zoom, = ax_enstrophy_zoom.plot([], [], 'r--', label='Prediction', linewidth=2)
    ax_enstrophy_zoom.legend()
    
    # Step counter
    step_text = fig.suptitle('Step: 0', fontsize=16, y=0.98)
    
    # Set axis limits (using truncated range matching utilities.py)
    k_min_full = k_bins[start_truth]  # Start from index 1 (skip k=0)
    k_max_full = k_bins[k_nyquist - 1] if k_nyquist < len(k_bins) else k_bins[-1]  # Truncate at Nyquist
    k_min_zoom = k_bins[zoom_start_idx]
    k_max_zoom = k_bins[k_nyquist - 1] if k_nyquist < len(k_bins) else k_bins[-1]  # Also truncate zoom at Nyquist
    
    ax_energy_full.set_xlim(k_min_full, k_max_full)
    ax_energy_full.set_ylim(Ek_min, Ek_max)
    ax_energy_zoom.set_xlim(k_min_zoom, k_max_zoom)
    ax_energy_zoom.set_ylim(Ek_min, Ek_max)
    
    ax_enstrophy_full.set_xlim(k_min_full, k_max_full)
    ax_enstrophy_full.set_ylim(Zk_min, Zk_max)
    ax_enstrophy_zoom.set_xlim(k_min_zoom, k_max_zoom)
    ax_enstrophy_zoom.set_ylim(Zk_min, Zk_max)
    
    def animate(frame):
        if frame >= len(spectral_data_list):
            return []
        
        data = spectral_data_list[frame]
        k_bins = data['k_bins']
        Ek_target = data['Ek_target']
        Ek_pred = data['Ek_pred']
        Zk_target = data['Zk_target']
        Zk_pred = data['Zk_pred']
        
        # Apply truncation matching utilities.py: start from index 1, truncate at Nyquist
        H_frame = data.get('H', H)
        Lx_frame = data.get('Lx', Lx)
        k_nyquist_frame = int((np.pi * H_frame) // Lx_frame)
        start_idx = start_truth
        end_idx = min(k_nyquist_frame, len(k_bins))
        
        # Use truncated range
        k_plot = k_bins[start_idx:end_idx]
        Ek_target_plot = Ek_target[start_idx:end_idx]
        Ek_pred_plot = Ek_pred[start_idx:end_idx]
        Zk_target_plot = Zk_target[start_idx:end_idx]
        Zk_pred_plot = Zk_pred[start_idx:end_idx]
        
        # Skip zero values for log scale
        mask = k_plot > 0
        k_plot = k_plot[mask]
        Ek_target_plot = Ek_target_plot[mask]
        Ek_pred_plot = Ek_pred_plot[mask]
        Zk_target_plot = Zk_target_plot[mask]
        Zk_pred_plot = Zk_pred_plot[mask]
        
        # Remove zero energy/enstrophy values for log scale
        Ek_target_mask = Ek_target_plot > 0
        Ek_pred_mask = Ek_pred_plot > 0
        Zk_target_mask = Zk_target_plot > 0
        Zk_pred_mask = Zk_pred_plot > 0
        
        # Full spectrum
        line_Ek_target_full.set_data(k_plot[Ek_target_mask], Ek_target_plot[Ek_target_mask])
        line_Ek_pred_full.set_data(k_plot[Ek_pred_mask], Ek_pred_plot[Ek_pred_mask])
        line_Zk_target_full.set_data(k_plot[Zk_target_mask], Zk_target_plot[Zk_target_mask])
        line_Zk_pred_full.set_data(k_plot[Zk_pred_mask], Zk_pred_plot[Zk_pred_mask])
        
        # Zoomed spectrum - filter for high frequencies
        zoom_mask = k_plot >= k_min_zoom
        
        # Energy spectrum zoom
        Ek_target_zoom_mask = zoom_mask & Ek_target_mask
        Ek_pred_zoom_mask = zoom_mask & Ek_pred_mask
        k_zoom_E_target = k_plot[Ek_target_zoom_mask]
        k_zoom_E_pred = k_plot[Ek_pred_zoom_mask]
        Ek_target_zoom = Ek_target_plot[Ek_target_zoom_mask]
        Ek_pred_zoom = Ek_pred_plot[Ek_pred_zoom_mask]
        
        # Enstrophy spectrum zoom
        Zk_target_zoom_mask = zoom_mask & Zk_target_mask
        Zk_pred_zoom_mask = zoom_mask & Zk_pred_mask
        k_zoom_Z_target = k_plot[Zk_target_zoom_mask]
        k_zoom_Z_pred = k_plot[Zk_pred_zoom_mask]
        Zk_target_zoom = Zk_target_plot[Zk_target_zoom_mask]
        Zk_pred_zoom = Zk_pred_plot[Zk_pred_zoom_mask]
        
        line_Ek_target_zoom.set_data(k_zoom_E_target, Ek_target_zoom)
        line_Ek_pred_zoom.set_data(k_zoom_E_pred, Ek_pred_zoom)
        line_Zk_target_zoom.set_data(k_zoom_Z_target, Zk_target_zoom)
        line_Zk_pred_zoom.set_data(k_zoom_Z_pred, Zk_pred_zoom)
        
        step_text.set_text(f'Step: {data["time_step"]}')
        
        return [line_Ek_target_full, line_Ek_pred_full, line_Zk_target_full, line_Zk_pred_full,
                line_Ek_target_zoom, line_Ek_pred_zoom, line_Zk_target_zoom, line_Zk_pred_zoom,
                step_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=num_steps,
                                  interval=1000/fps, blit=False, repeat=True)
    
    # Save animation
    if save_animation and log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        # Include actual step range in filename if using frame sampling
        if step_interval is not None and len(spectral_data_list) > 0:
            first_step = spectral_data_list[0].get('time_step', 0)
            last_step = spectral_data_list[-1].get('time_step', num_steps - 1)
            step_range = f"steps{first_step}-{last_step}_interval{step_interval}_frames{num_steps}"
        else:
            step_range = f"totalsteps{num_steps}"
        anim_path = f'{log_path}/spectral_comparison_animation_{step_range}.mp4'
        print(f"Saving spectral animation to {anim_path}...")
        print(f"  Animation details: {num_steps} frames, {fps} fps, bitrate=1800")
        start_time = time.time()
        anim.save(anim_path, writer='ffmpeg', fps=fps, bitrate=1800)
        elapsed_time = time.time() - start_time
        print(f"Spectral animation saved to {anim_path} (took {elapsed_time:.2f} seconds, ~{elapsed_time/60:.2f} minutes)")
    
    return anim, fig


def load_and_animate_predictions(log_path, save_animation=True, fps=10, k_zoom_threshold=20, num_steps=None, num_animation_frames=None):
    """
    Load saved data and create both animations.
    
    Args:
        log_path: Path to directory containing test_data_prediction.pth
        save_animation: Whether to save animation files
        fps: Frames per second for animations
        k_zoom_threshold: Wavenumber threshold for zoomed view in spectral animation
        num_steps: Maximum number of steps to use (None = use all)
        num_animation_frames: Desired number of frames in animation (None = use all steps)
                             If specified, calculates step interval automatically
    """
    # Load data
    data_path = f'{log_path}/test_data_prediction.pth'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    save_data = torch.load(data_path, map_location='cpu')
    print(f"Loaded data from {data_path}")
    print(f"Number of steps: {len(save_data['rel_l2_loss_by_step'])}")
    print(f"Dataset form: {save_data.get('dataset_form', 'unknown')}")
    
    # Create animations
    print("\nCreating prediction animation...")
    anim1, fig1 = animate_predictions(save_data, log_path, save_animation, fps, num_steps, num_animation_frames)
    
    print("\nCreating spectral comparison animation...")
    anim2, fig2 = animate_spectral_comparison(save_data, log_path, save_animation, 
                                              fps, k_zoom_threshold, num_steps, num_animation_frames)
    
    return anim1, anim2, fig1, fig2


if __name__ == '__main__':
    
    #### 1. predict and save the data
    #if you dont have the dataloader, comment this line
    model, test_loader, log_path = load_data_model(just_load_path=False)
    
    save_data = predict_and_save(model, test_loader, save=True, log_path=log_path)
    
    # #### 2. load the save_data and create animations
    # save_data = torch.load(f'{log_path}/test_data_prediction.pth', map_location='cpu')
    anim1, anim2, fig1, fig2 = load_and_animate_predictions(log_path, save_animation=True, fps=10, k_zoom_threshold=20, num_steps=args.num_steps, num_animation_frames=args.num_animation_frames)
    
    
    
    
    
    # #### 3. compute different types of metrics
    # compute_evalutation_metrics(save_data, model_name=args.model, log_path=log_path)

    #### 4. postprocessing save the data for diffusion training
    # no_postprocessing_pred_save_data(args)

    #### 5. plot the spectral error
    # plot_spectral_error(save_data, model_name=args.model, log_path=log_path)
    
    #### 6. create animations (uncomment to generate animations)
    # anim1, anim2, fig1, fig2 = load_and_animate_predictions(log_path, save_animation=True, fps=10, k_zoom_threshold=20)