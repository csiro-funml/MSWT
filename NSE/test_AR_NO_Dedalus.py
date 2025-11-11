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

parser.add_argument('--model', type=str, default='FNO') # FNO, ViT, UNO, CNO, Oformer, Transolver, DPOT, Crossformer, wavelet_transformer
parser.add_argument('--dataset',type=str, default='ns2d_pda') # ['ns2d_fno_1e-3', 'ns2d_pda', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 'sw2d_pda'], note: pdb is the pde bench
parser.add_argument('--resume_path',type=int, default=0 if not torch.cuda.is_available() else 1) # use random weights if not cuda available
parser.add_argument('--use_writer', action='store_true',default=False)
parser.add_argument('--form',type=str, default='vorticity', choices=['vorticity', 'velocity'])



# ### dataset details
parser.add_argument('--T_in', type=int, default=1)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--pad', type=int, default=0)
parser.add_argument('--normalize',type=int, default=1)
parser.add_argument('--normalize_strategy',type=str, default='zscore')

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
parser.add_argument('--epochs', type=int, default=3000)
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
    if not torch.cuda.is_available() and args.dataset != 'ns2d_dedalus_big':
        train_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, n_channels=3, normalize=args.normalize, train='train')
        test_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels=3, normalize=args.normalize, train='test')
    elif args.dataset == 'ns2d_dedalus_big':
        # load data and dataloader for big dedalus dataset
        train_dataset = MemmapDedalusBigDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form=args.form, normalize=args.normalize, train='train', strategy=args.normalize_strategy)
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

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
    # test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=0 if not torch.cuda.is_available() else 8, pin_memory=torch.cuda.is_available()) # TODO: removed later, jsut to test the traning set results
    # val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)

    # ntrain, ntest = len(train_dataset), len(test_dataset)

    if not args.pad:
        args.res = train_dataset.res  # use original dataset  resolution to train the model

    testing_mode = 'FNO_testing'
    if testing_mode == 'FNO_testing':
        if args.dataset == 'ns2d_dedalus_big':
            comment = args.comment + '{}_{}_mod{}_wid{}_lay{}_ntrain{}_normalizer_{}_form_{}'.format(args.dataset, args.model, args.modes, args.width, args.n_layers, ntrain, args.normalize_strategy, args.form)
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
# def get_initial_input_and_forcing_loader(test_dataset):



def predict_and_save(model, test_loader, save=False, log_path=None, max_steps=None):
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
        total_steps = min(max_steps, test_dataset.n_size)
        pred = []
        target = []
        for i in range(total_steps):
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
        
        pred = torch.cat(pred, dim=0)
        target = torch.cat(target, dim=0)
        print("pred shape", pred.shape, "target shape", target.shape)
        
        # Create save_data dictionary in expected format
        # Note: pred and target are concatenated along dim=0 (sample dimension)
        # They should have shape (N, H, W, T, C) where N is number of samples
        save_data = {'pred': pred, 'output': target}
        
        # Optionally save input data if needed
        # save_data['input'] = ...  # Can be added if needed
        
        return save_data
        
        # save_data = {'input': [], 'output': [], 'pred': []}
        # # autoregressive computing  
        # xx = x.to(device)  # (1, H, W, T_in, C_in) - original input (not normalized)
        
        # # Store original input for saving (before normalization)
        # x_original = xx.clone()
        
        # # Normalize input for model prediction
        # xx = test_dataset.normalize_x(xx)  # normalize the input before the autoregressive predicting
        
        # # Store targets as we compute them (for final output saving)
        # target_outputs = []
        
        # # Store predictions and metrics
        # pred = []
        # rel_l2_errors = []  # Store RelL2Norm for each time step
        # spectral_data = []  # Store spectral data for each time step
        
        # # Initialize loss functions
        # rel_l2_loss = RelL2Norm()
        # energy_enstropy_spectrum_error = Energy_Enstropy_SpectrumError(
        #     model_name=args.model, 
        #     save_path=None
        #     # save_path=log_path if save else None
        # )
        
        # # Domain size (default to 2*pi, can be adjusted if needed)
        # Lx = 2 * np.pi
        # Ly = 2 * np.pi
        
        # # Create directory for spectral plots if saving
        # if save:
        #     os.makedirs(os.path.join(log_path, 'spectral_error'), exist_ok=True)
        #     os.makedirs(os.path.join(log_path, 'rollout_metrics'), exist_ok=True)
        
        # for t in tqdm(range(0, max_steps, args.T_bundle), desc="Predicting and saving"):
        #     # Predict next step (outputs only main variables, no forcing)
        #     im = model(xx)  # (1, H, W, T_bundle, C_out) where C_out doesn't include forcing
            
        #     # Store prediction (denormalized main variables only)
        #     im_denorm = test_dataset.denormalize_x(im)
        #     if t == 0:
        #         pred = im_denorm
        #     else:
        #         pred = torch.cat((pred, im_denorm), -2)
            
        #     # Compute metrics for each timestep in the bundle
        #     for bundle_idx in range(args.T_bundle):
        #         time_step_actual = t + bundle_idx
        #         if time_step_actual >= max_steps:
        #             break
                
        #         # Get prediction for this timestep
        #         pred_step = im_denorm[..., bundle_idx, :]  # (1, H, W, C_out)
                
        #         # Load target on-demand (lazy loading)
        #         target_step_tensor = get_target_at_timestep(time_step_actual).to(device)  # (1, H, W, 1, C_out)
        #         target_step = target_step_tensor[..., 0, :]  # (1, H, W, C_out)
                
        #         # Store target for final output saving
        #         if time_step_actual == 0:
        #             target_outputs = target_step_tensor
        #         else:
        #             target_outputs = torch.cat([target_outputs, target_step_tensor], dim=-2)
                
        #         # Compute RelL2Norm for this timestep
        #         pred_step_expanded = pred_step.unsqueeze(-2)  # (1, H, W, 1, C_out)
        #         rel_l2_err = rel_l2_loss(pred_step_expanded, target_step_tensor)
        #         rel_l2_errors.append({
        #             'time_step': time_step_actual,
        #             'rel_l2_error': rel_l2_err.item()
        #         })
                
        #         # Compute spectral error for this timestep
        #         if args.dataset == 'ns2d_dedalus_big' or 'ns2d' in args.dataset:
        #             try:
                        
        #                 # Extract spectral data for saving
        #                 if save and hasattr(test_dataset, 'form'):
        #                     # Get velocity components based on form
        #                     if test_dataset.form == 'vorticity':
        #                         # For vorticity form: compute velocity from streamfunction
        #                         psi_pred = pred_step[0, ..., 1].detach().cpu().numpy()  # streamfunction
        #                         psi_target = target_step[0, ..., 1].detach().cpu().numpy()
                                
        #                         ux_pred, uy_pred = streamfunction_to_velocity(psi_pred, Lx, Ly)
        #                         ux_target, uy_target = streamfunction_to_velocity(psi_target, Lx, Ly)
        #                     elif test_dataset.form == 'velocity':
        #                         # For velocity form: use velocity components directly
        #                         ux_pred = pred_step[0, ..., 1].detach().cpu().numpy()  # velocity_x
        #                         uy_pred = pred_step[0, ..., 2].detach().cpu().numpy()  # velocity_y
        #                         ux_target = target_step[0, ..., 1].detach().cpu().numpy()
        #                         uy_target = target_step[0, ..., 2].detach().cpu().numpy()
        #                     else:
        #                         ux_pred = uy_pred = ux_target = uy_target = None
                            
        #                     if ux_pred is not None:
        #                         # Compute spectra
        #                         k_bins, Ek_pred, Zk_pred = compute_spectra(ux_pred, uy_pred, Lx, Ly)
        #                         _, Ek_target, Zk_target = compute_spectra(ux_target, uy_target, Lx, Ly)
                                
        #                         spectral_data.append({
        #                             'time_step': time_step_actual,
        #                             'k_bins': k_bins,
        #                             'Ek_pred': Ek_pred,
        #                             'Ek_target': Ek_target,
        #                             'Zk_pred': Zk_pred,
        #                             'Zk_target': Zk_target
        #                         })
        #             except Exception as e:
        #                 print(f"Warning: Failed to compute spectral error at time step {time_step_actual}: {e}")
        #                 import traceback
        #                 traceback.print_exc()
            
        #     # Prepare input for next step: combine predicted vars with ground truth forcing
        #     if has_forcing:
        #         # Load forcing on-demand for the timesteps we just predicted
        #         forcing_list = []
        #         for bundle_idx in range(args.T_bundle):
        #             time_step_actual = t + bundle_idx
        #             if time_step_actual >= max_steps:
        #                 # Use last available forcing if beyond bounds
        #                 time_step_actual = max_steps - 1
        #             gt_forcing_single = get_forcing_at_timestep(time_step_actual)  # (H, W, 1, 2)
        #             forcing_list.append(gt_forcing_single)
                
        #         # Concatenate forcing for all timesteps in bundle
        #         gt_forcing = torch.cat(forcing_list, dim=2)  # (H, W, T_bundle, 2)
        #         gt_forcing = gt_forcing.unsqueeze(0).to(device)  # (1, H, W, T_bundle, 2)
                
        #         # Normalize forcing using the same stats as input (forcing is in input channels)
        #         forcing_mean = test_dataset.norm_mean[forcing_in_input_idx].to(device)
        #         forcing_std = test_dataset.norm_std[forcing_in_input_idx].to(device)
        #         gt_forcing_norm = (gt_forcing - forcing_mean) / (forcing_std + 1e-6)
                
        #         # The model output `im` is already normalized
        #         im_norm = im  # Already normalized, no need to normalize again
                
        #         # Concatenate predicted main vars (normalized) with ground truth forcing (normalized)
        #         im_with_forcing = torch.cat([im_norm, gt_forcing_norm], dim=-1)  # (1, H, W, T_bundle, C_in)
                
        #         # Update xx for next step: shift window and add new prediction with forcing
        #         xx = torch.cat((xx[..., args.T_bundle:, :], im_with_forcing), dim=-2)
        #     else:
        #         # No forcing: just use predicted values (already normalized)
        #         xx = torch.cat((xx[..., args.T_bundle:, :], im), dim=-2)
        
        # # Store data (use original input, not the modified xx)
        # save_data['input'] = x_original  # Original input (already denormalized)
        # save_data['output'] = target_outputs  # Target outputs (loaded on-demand, denormalized) - (1, H, W, T_out, C_out)
        # save_data['pred'] = pred  # Predictions (denormalized) - (1, H, W, T_out, C_out)

        # # Store metrics
        # save_data['rel_l2_errors'] = rel_l2_errors
        # save_data['spectral_data'] = spectral_data

        # # print the shape of the np_data
        # print("save_data shape", save_data['input'].shape, save_data['output'].shape, save_data['pred'].shape)
        # print(f"Computed metrics for {len(rel_l2_errors)} time steps")
        # print(f"Computed spectral data for {len(spectral_data)} time steps")
        
        # # Print summary of errors
        # if rel_l2_errors:
        #     errors_array = np.array([e['rel_l2_error'] for e in rel_l2_errors])
        #     print(f"RelL2Norm - Mean: {errors_array.mean():.6f}, Std: {errors_array.std():.6f}, Min: {errors_array.min():.6f}, Max: {errors_array.max():.6f}")

        # # save to npz
        # if save:
        #     os.makedirs(log_path, exist_ok=True)
        #     torch.save(save_data, f'{log_path}/test_data_prediction.pth')
            
        #     # Save metrics to CSV
        #     if rel_l2_errors:
        #         metrics_df = pd.DataFrame(rel_l2_errors)
        #         metrics_df.to_csv(f'{log_path}/rollout_metrics/rel_l2_errors_by_timestep.csv', index=False)
        #         print(f"Saved RelL2Norm errors to {log_path}/rollout_metrics/rel_l2_errors_by_timestep.csv")
            
        #     # Save spectral data summary
        #     spectral_summary = []  # Initialize outside if block
        #     if spectral_data:
        #         # Create a summary of spectral errors
        #         for spec_data in spectral_data:
        #             t = spec_data['time_step']
        #             Ek_pred = spec_data['Ek_pred']
        #             Ek_target = spec_data['Ek_target']
        #             Zk_pred = spec_data['Zk_pred']
        #             Zk_target = spec_data['Zk_target']
                    
        #             # Compute relative errors in energy and enstrophy spectra
        #             # Use a reasonable cutoff (Nyquist frequency)
        #             H = Ek_pred.shape[0]
        #             k_nyquist = int((np.pi * H) // Lx)
        #             start_idx = 1
        #             end_idx = min(k_nyquist, len(Ek_pred))
                    
        #             if end_idx > start_idx:
        #                 Ek_pred_trimmed = Ek_pred[start_idx:end_idx]
        #                 Ek_target_trimmed = Ek_target[start_idx:end_idx]
        #                 Zk_pred_trimmed = Zk_pred[start_idx:end_idx]
        #                 Zk_target_trimmed = Zk_target[start_idx:end_idx]
                        
        #                 # Compute relative L2 error in spectra
        #                 Ek_rel_error = np.sqrt(np.mean((Ek_pred_trimmed - Ek_target_trimmed)**2)) / (np.sqrt(np.mean(Ek_target_trimmed**2)) + 1e-10)
        #                 Zk_rel_error = np.sqrt(np.mean((Zk_pred_trimmed - Zk_target_trimmed)**2)) / (np.sqrt(np.mean(Zk_target_trimmed**2)) + 1e-10)
                        
        #                 spectral_summary.append({
        #                     'time_step': t,
        #                     'energy_spectrum_rel_error': Ek_rel_error,
        #                     'enstrophy_spectrum_rel_error': Zk_rel_error
        #                 })
                
        #         if spectral_summary:
        #             spectral_df = pd.DataFrame(spectral_summary)
        #             spectral_df.to_csv(f'{log_path}/rollout_metrics/spectral_errors_by_timestep.csv', index=False)
        #             print(f"Saved spectral errors to {log_path}/rollout_metrics/spectral_errors_by_timestep.csv")
                    
        #             # Print summary
        #             ek_errors = np.array([s['energy_spectrum_rel_error'] for s in spectral_summary])
        #             zk_errors = np.array([s['enstrophy_spectrum_rel_error'] for s in spectral_summary])
        #             print(f"Energy Spectrum Rel Error - Mean: {ek_errors.mean():.6f}, Std: {ek_errors.std():.6f}")
        #             print(f"Enstrophy Spectrum Rel Error - Mean: {zk_errors.mean():.6f}, Std: {zk_errors.std():.6f}")
            
        #     # Create a combined metrics plot
        #     if rel_l2_errors:
        #         try:
        #             n_plots = 1 if not spectral_summary else 2
        #             fig, axes = plt.subplots(n_plots, 1, figsize=(12, 6 * n_plots))
        #             if n_plots == 1:
        #                 axes = [axes]  # Make it a list for consistent indexing
                    
        #             # Plot RelL2Norm over time
        #             time_steps = [e['time_step'] for e in rel_l2_errors]
        #             rel_l2_values = [e['rel_l2_error'] for e in rel_l2_errors]
        #             axes[0].plot(time_steps, rel_l2_values, 'b-', linewidth=2, marker='o', markersize=4)
        #             axes[0].set_xlabel('Time Step', fontsize=12)
        #             axes[0].set_ylabel('RelL2Norm Error', fontsize=12)
        #             axes[0].set_title('Relative L2 Norm Error vs Time Step', fontsize=14)
        #             axes[0].grid(True)
        #             axes[0].set_yscale('log')
                    
        #             # Plot spectral errors over time if available
        #             if spectral_summary:
        #                 spec_time_steps = [s['time_step'] for s in spectral_summary]
        #                 ek_errors = [s['energy_spectrum_rel_error'] for s in spectral_summary]
        #                 zk_errors = [s['enstrophy_spectrum_rel_error'] for s in spectral_summary]
        #                 axes[1].plot(spec_time_steps, ek_errors, 'r-', linewidth=2, marker='s', markersize=4, label='Energy Spectrum')
        #                 axes[1].plot(spec_time_steps, zk_errors, 'g-', linewidth=2, marker='^', markersize=4, label='Enstrophy Spectrum')
        #                 axes[1].set_xlabel('Time Step', fontsize=12)
        #                 axes[1].set_ylabel('Relative Spectrum Error', fontsize=12)
        #                 axes[1].set_title('Spectral Error vs Time Step', fontsize=14)
        #                 axes[1].legend(fontsize=10)
        #                 axes[1].grid(True)
        #                 axes[1].set_yscale('log')
                    
        #             plt.tight_layout()
        #             plt.savefig(f'{log_path}/rollout_metrics/metrics_vs_timestep.png', dpi=150, bbox_inches='tight')
        #             plt.close()
        #             print(f"Saved metrics plot to {log_path}/rollout_metrics/metrics_vs_timestep.png")
        #         except Exception as e:
        #             print(f"Warning: Failed to create metrics plot: {e}")
        #             import traceback
        #             traceback.print_exc()
        #     # save another version into numpy
        #     if args.dataset == 'ns2d_dedalus_big':
        #         # Convert output from (1, H, W, T, C) to (T, H, W) for each channel
        #         if test_dataset.form == 'vorticity':
        #             save_data_numpy = {
        #                 'input': {
        #                     'vorticity': save_data['input'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),  # (T_in, H, W)
        #                     'streamfunction': save_data['input'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
        #                     'forcing_x': save_data['input'][0, ..., 2].permute(2, 0, 1).cpu().numpy(),
        #                     'forcing_y': save_data['input'][0, ..., 3].permute(2, 0, 1).cpu().numpy(),
        #                 },
        #                 'output': {
        #                     'vorticity': save_data['output'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),  # (T_out, H, W)
        #                     'streamfunction': save_data['output'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
        #                 },
        #                 'pred': {
        #                     'vorticity': save_data['pred'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),  # (T_out, H, W)
        #                     'streamfunction': save_data['pred'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
        #                 }
        #             }
        #         else:  # velocity
        #             save_data_numpy = {
        #                 'input': {
        #                     'pressure': save_data['input'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
        #                     'velocity_x': save_data['input'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
        #                     'velocity_y': save_data['input'][0, ..., 2].permute(2, 0, 1).cpu().numpy(),
        #                     'forcing_x': save_data['input'][0, ..., 3].permute(2, 0, 1).cpu().numpy(),
        #                     'forcing_y': save_data['input'][0, ..., 4].permute(2, 0, 1).cpu().numpy(),
        #                 },
        #                 'output': {
        #                     'pressure': save_data['output'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
        #                     'velocity_x': save_data['output'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
        #                     'velocity_y': save_data['output'][0, ..., 2].permute(2, 0, 1).cpu().numpy(),
        #                 },
        #                 'pred': {
        #                     'pressure': save_data['pred'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
        #                     'velocity_x': save_data['pred'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
        #                     'velocity_y': save_data['pred'][0, ..., 2].permute(2, 0, 1).cpu().numpy(),
        #                 }
        #             }
        #         print("save_data_numpy keys:", save_data_numpy.keys())
        #         if test_dataset.form == 'vorticity':
        #             print("save_data_numpy shape", save_data_numpy['input']['vorticity'].shape, 
        #                   save_data_numpy['output']['vorticity'].shape, 
        #                   save_data_numpy['pred']['vorticity'].shape)
        #         else:
        #             print("save_data_numpy shape", save_data_numpy['input']['pressure'].shape, 
        #                   save_data_numpy['output']['pressure'].shape, 
        #                   save_data_numpy['pred']['pressure'].shape)
        #     else:
        #         # Original format for other datasets
        #         save_data_numpy = {
        #             'input': {'vorticity': save_data['input'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
        #                       'streamfunction': save_data['input'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
        #                       },
        #             'output': {'vorticity': save_data['output'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
        #                         'streamfunction': save_data['output'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
        #                         },
        #             'pred': {'vorticity': save_data['pred'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
        #                      'streamfunction': save_data['pred'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
        #                      }
        #         }
        #         print("save_data_numpy shape", save_data_numpy['input']['vorticity'].shape, 
        #               save_data_numpy['output']['vorticity'].shape, 
        #               save_data_numpy['pred']['vorticity'].shape)
        return save_data



################################################################
# Function 2 Compute evaluation metrics
################################################################
def compute_evalutation_metrics(save_data, model_name='', log_path=''):
    pred, target = save_data['pred'], save_data['output'] # shape: (B, H, W, T, C)
    
    # for step in [0, -1]: # first step and last step
        # for c in range(pred.shape[-1]):
            # low_err, mid_err, high_err = SpectralError()(pred[..., step, c][:, :, :, None, None], target[..., step, c][:, :, :, None, None])
            

    loss_dict = {}
    loss_dict['rel_l2_loss'] = RelL2Norm() # rel L2 loss
    # loss_dict['rmse'] = RMSE()
    # loss_dict['boundary_rmse'] = BoundaryRMSE()
    # loss_dict['max_avg'] = MaxAbsError()
    # loss_dict['max_global'] = GlobalMaxAbsError()
    # loss_dict['spectral_error_radial'] = SpectralError(model_name=model_name, save_path=log_path, low_percentile=0.70, high_percentile=0.97, method='radial')
    # loss_dict['spectral_error_square'] = SpectralError(model_name=model_name, save_path=log_path, low_percentile=0.70, high_percentile=0.97, method='square approximation')
    # loss_dict['spectral_error_cfd'] = SpectralError(model_name=model_name, save_path=log_path, low_percentile=0.70, high_percentile=0.97, method='cfd')
    loss_dict['energy_enstropy_spectrum_error'] = Energy_Enstropy_SpectrumError(model_name=model_name, save_path=log_path)
    if 'ns2d' in log_path and 'torchcf' not in log_path: # NS equation
        step_dict = {0: "t=1", -1: "t=T"} # just plot two steps
    else:
        total_steps_to_compute = 5
        step_dict = {t: f"t={t+1}" for t in range(0, pred.shape[-2], pred.shape[-2]//total_steps_to_compute)}
        print("steps to compute",step_dict.keys())
    # Standard error metrics
    print("\n=== Channel-wise Error Metrics ===")
    save_df = pd.DataFrame(columns=["step", "channel", "metric", f"{model_name}"])
    
    for step in step_dict.keys(): # first step and last step
        print("evaluating step .....", step)
        print("pred shape", pred.shape, "target shape", target.shape)
        if 'energy_enstropy_spectrum_error' in loss_dict.keys():
            loss_metric = loss_dict['energy_enstropy_spectrum_error'](pred[..., step, :], target[..., step, :], save_plot=True, time_step=step)
            continue
        for c in range(pred.shape[-1]):
            # evaluate different metrics per channel
            for key, loss_func in loss_dict.items():
                if 'spectral_error' in key:
                    # (B, H, W, T, C)
                    loss_metric = loss_func(pred[..., step, c][:, :, :, None, None], target[..., step, c][:, :, :, None, None],
                                             channel=c, time_step=step, save_plot=True)
                    # loss metric is a dict with keys: 'low_err', 'mid_err', 'high_err', 'k_low', 'k_high'
                    print("frequency bands", loss_metric['k_low'], loss_metric['k_high'])
                    for band_key, val in loss_metric.items():
                        if band_key == 'k_low' or band_key == 'k_high':
                            continue
                        print(f"Channel {c} {step_dict[step]} {band_key}: {val:.6f}")
                        new_row = pd.Series({"step": step_dict[step], "channel": c, "metric": band_key, f"{model_name}": val,
                                             "k_low": loss_metric['k_low'], "k_high": loss_metric['k_high']}).to_frame().T
                        save_df = pd.concat([save_df, new_row], ignore_index=True)
                else:
                    
                    loss_metric = loss_func(pred[..., step, c][:, :, :, None, None], target[..., step, c][:, :, :, None, None])
                    print(f"Channel {c} {step_dict[step]} {key}: {loss_metric.item():.6f}")   
                    new_row = pd.Series({"step": step_dict[step], "channel": c, "metric": key, f"{model_name}": loss_metric.item()}).to_frame().T
                    save_df = pd.concat([save_df, new_row], ignore_index=True)
    print(save_df.head(n=16))
    save_df.to_csv(f"{log_path}/evalutation_metrics_{model_name}_epochs_{args.epochs}.csv", index=False)
    return loss_dict
    

################################################################
# Function 3  save the data for diffusion training
################################################################
def no_postprocessing_pred_save_data(args):

    ################################################################
    # load some toy data to run locally
    if not torch.cuda.is_available():
        train_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, n_channels=3, normalize=args.normalize, train='train')
        test_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels=3, normalize=args.normalize, train='test')
        val_dataset = test_dataset
    else: 
        # load data and dataloader
        train_dataset = TemporalDataset2D(args.dataset, t_in = args.T_in, t_ar = args.T_ar, train='train', normalize=args.normalize)
        val_dataset =  TemporalDataset2D(args.dataset, n_train=260, t_in = args.T_in, t_ar =-1, train='val', normalize=args.normalize)
        test_dataset = TemporalDataset2D(args.dataset, n_train=260, t_in=args.T_in, t_ar=-1, n_channels = train_dataset.n_channels, train='test', normalize=args.normalize)


    
    ntrain, ntest = len(train_dataset), len(test_dataset)
    ntrain = 5200 if args.dataset == 'ns2d_pda' else ntrain # for testing


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
    val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)

    loaderdict = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    if not args.pad:
        args.res = train_dataset.res  # use original dataset  resolution to train the model

    comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
    log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
    if not os.path.exists(log_path):# running tests locallt
        log_path = './logs/' + comment
    model_path = log_path + f'/model_epochs_{args.epochs}.pth'
    print(model_path)
    
    # Load pretrained neural operator
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

    # load weights
    model, optimizer, scheduler, start_epoch = resume_training_from_checkpoint(model, model_path, device, optimizer=None, scheduler=None)
    print(f"Loaded {args.model} neural operator")


    # iterate throught train/val/test loader and save the predictions
    loaderdict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    for key, loader in loaderdict.items():
        model.eval()
        with torch.no_grad():
            save_data = {'output': [], 'pred': []}
            # autoregressive computing  
            for xx, yy in tqdm(loader):
                xx = xx.to(device)
                yy = yy.to(device)
                xx = loader.dataset.normalize_x(xx) # normalize the input before the autoregressive predicting
                for t in range(0, yy.shape[-2], args.T_bundle):
                    im = model(xx)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -2)
                    xx = torch.cat((xx[..., args.T_bundle:,:], im), dim=-2)
                # denormalize the pred at the final step (get better results)   
                pred = loader.dataset.denormalize_x(pred)    

                # # save the data to np_data
                save_data['output'].append(yy)
                save_data['pred'].append(pred)

            # organzie np_data
            save_data['output'] = torch.cat(save_data['output'], axis=0).cpu().numpy()
            save_data['pred'] = torch.cat(save_data['pred'], axis=0).cpu().numpy()
            
            print("save_data shape", save_data['output'].shape, save_data['pred'].shape)
            if not os.path.exists(f'{log_path}/diffusion'):
                os.makedirs(f'{log_path}/diffusion')
            np.savez(f'{log_path}/diffusion/{key}_pred.npz', **save_data)
            print(f"Saved {key} predictions to {log_path}/diffusion/{key}_pred.npz")

    print("Successfully saved the predictions for diffusion training")


################################################################
# Function 4 plot the spectral error
################################################################
def plot_spectral_error(save_data, model_name='', log_path=''):
    pred, target = save_data['pred'], save_data['output'] # shape: (B, H, W, T, C)
    idx = 0
    channel_id = 0
    t = 0
    n_test = target.shape[1]
    if 'ns2d' in log_path and 'torchcf' not in log_path: # NS equation
        step_dict = {0: "t=1", -1: "t=T"} # just plot two steps
    else:
        total_steps_to_compute = 5
        step_dict = {t: f"t={t+1}" for t in range(0, pred.shape[-2], pred.shape[-2]//total_steps_to_compute)}
        print("steps to compute",step_dict.keys())
    
    for channel_id in range(pred.shape[-1]):
        for step in step_dict.keys(): # first step and last step
            print("evaluating step .....", step, "channel .....", channel_id)
            # plot_enstrophy_spectrum(
            #     [target[idx, ..., step, channel_id].cpu(),
            #     pred[idx, ..., step, channel_id].cpu()],
            #     h=2 * np.pi / n_test,
            #     labels=["Ground Truth", "Prediction"],
            #     title=f"t={step},c={channel_id}",
            #     factor=1,
            #     slope=5/3,
            #     log_path=log_path,
            #     model_name=model_name
            # )
            E_spectrum_fno = spectrum_2d(target[...,step, channel_id].cpu(), n_test)
            E_spectrum_pred = spectrum_2d(pred[..., step, channel_id].cpu(), n_test)
            cutoff = n_test//2+1
            E_spectrum_fno = E_spectrum_fno[:cutoff]
            E_spectrum_pred = E_spectrum_pred[:cutoff]
            plt.loglog(E_spectrum_fno, label = 'FNO Spectrum')
            plt.loglog(E_spectrum_pred, label = 'Prediction Spectrum')
            plt.legend()
            plt.savefig(f'{log_path}/spectral_error/{model_name}_{step}_{channel_id}_fno_spectrum.png')
            plt.show()
            plt.clf()




if __name__ == '__main__':
    
    #### 1. predict and save the data
    model, test_loader, log_path = load_data_model(just_load_path=False)
    save_data = predict_and_save(model, test_loader, save=True, log_path=log_path, max_steps=1000)
    
    # #### 2. load the save_data
    # save_data = torch.load(f'{log_path}/test_data_prediction.pth', map_location=device)
    
    # #### 3. compute different types of metrics
    # compute_evalutation_metrics(save_data, model_name=args.model, log_path=log_path)

    #### 4. postprocessing save the data for diffusion training
    # no_postprocessing_pred_save_data(args)


    #### 5. plot the spectral error
    # plot_spectral_error(save_data, model_name=args.model, log_path=log_path)