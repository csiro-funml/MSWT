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

    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=0 if not torch.cuda.is_available() else 8, pin_memory=torch.cuda.is_available()) # TODO: removed later, jsut to test the traning set results
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
def get_full_sequence_with_forcing(test_dataset):
    """
    Get the full sequence including forcing channels for autoregressive prediction.
    For MemmapDedalusBigDataset2D, we need to get the full data with forcing channels.
    """
    if hasattr(test_dataset, '_read_window') and hasattr(test_dataset, 'channel_indices'):
        # This is MemmapDedalusBigDataset2D
        t0_global = test_dataset.start_idx
        t1_global = test_dataset.end_idx
        data = test_dataset._read_window(t0_global, t1_global - t0_global)
        if test_dataset.temporal_downsample == 1:
            data_dowmsample = data
        else:
            data_dowmsample = data[::test_dataset.temporal_downsample]
        # Get all channels including forcing (not just channel_indices)
        # Channel order: [vorticity, streamfunction, velocity_x, velocity_y, pressure, forcing_x, forcing_y]
        data = data_dowmsample  # (T, C_all, H, W) where C_all=7
        data = torch.from_numpy(data.astype(np.float32))
        if test_dataset.downsample != (1, 1):
            target_N = test_dataset.H // test_dataset.downsample[0]
            data = test_dataset.downsample_x(data, target_N)
        data = data.permute(2, 3, 0, 1)  # (H, W, T, C_all)
        
        # Extract input and output based on form
        t_in = test_dataset.t_in
        x_full = data[..., :t_in, :].unsqueeze(0)  # (1, H, W, T_in, C_all)
        y_full = data[..., t_in:, :].unsqueeze(0)  # (1, H, W, T_out, C_all)
        
        # Extract the channels we need based on form
        if test_dataset.form == 'vorticity':
            # Input: vorticity, streamfunction, forcing_x, forcing_y (indices 0, 1, 5, 6)
            # Output: vorticity, streamfunction (indices 0, 1)
            channel_indices_in = [0, 1, 5, 6]
            channel_indices_out = [0, 1]
        else:  # velocity
            # Input: pressure, velocity_x, velocity_y, forcing_x, forcing_y (indices 4, 2, 3, 5, 6)
            # Output: pressure, velocity_x, velocity_y (indices 4, 2, 3)
            channel_indices_in = [4, 2, 3, 5, 6]
            channel_indices_out = [4, 2, 3]
        
        x = x_full[..., channel_indices_in]  # (1, H, W, T_in, C_in)
        y = y_full[..., channel_indices_out]  # (1, H, W, T_out, C_out)
        # Also return full y with forcing for ground truth forcing
        y_full_with_forcing = y_full  # (1, H, W, T_out, C_all=7)
        
        return x, y, y_full_with_forcing, channel_indices_in, channel_indices_out
    else:
        # For other datasets, use standard get_all_sequence
        x, y = test_dataset.get_all_sequence()
        return x, y, None, None, None

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
        
        # Get full sequence with forcing if available
        if args.dataset == 'ns2d_dedalus_big':
            x, y, y_full_with_forcing, channel_indices_in, channel_indices_out = get_full_sequence_with_forcing(test_dataset)
            has_forcing = True
            # Channel mapping for forcing: forcing_x=5, forcing_y=6 in full data
            if test_dataset.form == 'vorticity':
                forcing_indices = [5, 6]  # forcing_x, forcing_y
                main_var_indices = [0, 1]  # vorticity, streamfunction
            else:  # velocity
                forcing_indices = [5, 6]  # forcing_x, forcing_y
                main_var_indices = [4, 2, 3]  # pressure, velocity_x, velocity_y
        else:
            x, y = test_dataset.get_all_sequence()
            y_full_with_forcing = None
            has_forcing = False
            channel_indices_in = None
            channel_indices_out = None
        
        max_steps = y.shape[-2]
        
        save_data = {'input': [], 'output': [], 'pred': []}
        # autoregressive computing  
        xx = x.to(device)  # (1, H, W, T_in, C_in) - original input (not normalized)
        yy = y.to(device)  # (1, H, W, T_out, C_out) - original output (not normalized)
        
        # Store original input for saving (before normalization)
        x_original = xx.clone()
        
        # Normalize input for model prediction
        xx = test_dataset.normalize_x(xx)  # normalize the input before the autoregressive predicting
        
        # Store predictions
        pred = []
        
        for t in range(0, max_steps, args.T_bundle):
            # Predict next step (outputs only main variables, no forcing)
            im = model(xx)  # (1, H, W, T_bundle, C_out) where C_out doesn't include forcing
            
            # Store prediction (denormalized main variables only)
            im_denorm = test_dataset.denormalize_x(im)
            if t == 0:
                pred = im_denorm
            else:
                pred = torch.cat((pred, im_denorm), -2)
            
            # Prepare input for next step: combine predicted vars with ground truth forcing
            if has_forcing and y_full_with_forcing is not None:
                # Get ground truth forcing for the NEXT step (the step we just predicted)
                # t is the current prediction step index, so we use forcing at step t
                if t < y_full_with_forcing.shape[-2]:
                    # Get forcing for the timestep we just predicted
                    gt_forcing = y_full_with_forcing[..., t:t+args.T_bundle, forcing_indices].to(device)
                else:
                    # If we're beyond available data, use last available forcing
                    gt_forcing = y_full_with_forcing[..., -1:, forcing_indices].to(device)
                    if args.T_bundle > 1:
                        gt_forcing = gt_forcing.expand(-1, -1, -1, args.T_bundle, -1)
                
                # Normalize forcing using the same stats as input (forcing is in input channels)
                # Get the forcing channel stats from input normalization
                if test_dataset.form == 'vorticity':
                    # Input channels are [vorticity, streamfunction, forcing_x, forcing_y]
                    # So forcing is at indices 2, 3 in the input
                    forcing_in_input_idx = [2, 3]
                else:  # velocity
                    # Input channels are [pressure, velocity_x, velocity_y, forcing_x, forcing_y]
                    # So forcing is at indices 3, 4 in the input
                    forcing_in_input_idx = [3, 4]
                
                # Normalize forcing (it should use the same normalization as input forcing channels)
                forcing_mean = test_dataset.norm_mean[forcing_in_input_idx].to(device)
                forcing_std = test_dataset.norm_std[forcing_in_input_idx].to(device)
                gt_forcing_norm = (gt_forcing - forcing_mean) / (forcing_std + 1e-6)
                
                # The model output `im` is already normalized (model trained with normalized inputs/outputs)
                # The model outputs normalized predictions that match the output normalization stats
                # Since norm_mean_out = norm_mean[:n_out] for main variables, the normalization is consistent
                # We can use `im` directly - it's already in the normalized output space
                # which matches the normalized input space for the main variables
                im_norm = im  # Already normalized, no need to normalize again
                
                # Concatenate predicted main vars (normalized) with ground truth forcing (normalized)
                # This creates the input format: [main_vars (normalized), forcing (normalized)]
                im_with_forcing = torch.cat([im_norm, gt_forcing_norm], dim=-1)  # (1, H, W, T_bundle, C_in)
                
                # Update xx for next step: shift window and add new prediction with forcing
                xx = torch.cat((xx[..., args.T_bundle:, :], im_with_forcing), dim=-2)
            else:
                # No forcing: just use predicted values (already normalized)
                xx = torch.cat((xx[..., args.T_bundle:, :], im), dim=-2)
        
        # Store data (use original input, not the modified xx)
        save_data['input'] = x_original  # Original input (already denormalized)
        save_data['output'] = yy  # Original output (already denormalized)
        save_data['pred'] = pred  # Predictions (denormalized)

        # print the shape of the np_data
        print("save_data shape", save_data['input'].shape, save_data['output'].shape, save_data['pred'].shape)

        # save to npz
        if save:
            os.makedirs(log_path, exist_ok=True)
            torch.save(save_data, f'{log_path}/test_data_prediction.pth')
            # save another version into numpy
            if args.dataset == 'ns2d_dedalus_big':
                if test_dataset.form == 'vorticity':
                    save_data_numpy = {
                        'input': {
                            'vorticity': save_data['input'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
                            'streamfunction': save_data['input'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
                            'forcing_x': save_data['input'][0, ..., 2].permute(2, 0, 1).cpu().numpy(),
                            'forcing_y': save_data['input'][0, ..., 3].permute(2, 0, 1).cpu().numpy(),
                        },
                        'output': {
                            'vorticity': save_data['output'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
                            'streamfunction': save_data['output'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
                        },
                        'pred': {
                            'vorticity': save_data['pred'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
                            'streamfunction': save_data['pred'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
                        }
                    }
                else:  # velocity
                    save_data_numpy = {
                        'input': {
                            'pressure': save_data['input'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
                            'velocity_x': save_data['input'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
                            'velocity_y': save_data['input'][0, ..., 2].permute(2, 0, 1).cpu().numpy(),
                            'forcing_x': save_data['input'][0, ..., 3].permute(2, 0, 1).cpu().numpy(),
                            'forcing_y': save_data['input'][0, ..., 4].permute(2, 0, 1).cpu().numpy(),
                        },
                        'output': {
                            'pressure': save_data['output'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
                            'velocity_x': save_data['output'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
                            'velocity_y': save_data['output'][0, ..., 2].permute(2, 0, 1).cpu().numpy(),
                        },
                        'pred': {
                            'pressure': save_data['pred'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
                            'velocity_x': save_data['pred'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
                            'velocity_y': save_data['pred'][0, ..., 2].permute(2, 0, 1).cpu().numpy(),
                        }
                    }
                print("save_data_numpy keys:", save_data_numpy.keys())
                if test_dataset.form == 'vorticity':
                    print("save_data_numpy shape", save_data_numpy['input']['vorticity'].shape, 
                          save_data_numpy['output']['vorticity'].shape, 
                          save_data_numpy['pred']['vorticity'].shape)
                else:
                    print("save_data_numpy shape", save_data_numpy['input']['pressure'].shape, 
                          save_data_numpy['output']['pressure'].shape, 
                          save_data_numpy['pred']['pressure'].shape)
            else:
                # Original format for other datasets
                save_data_numpy = {
                    'input': {'vorticity': save_data['input'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
                              'streamfunction': save_data['input'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
                              },
                    'output': {'vorticity': save_data['output'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
                                'streamfunction': save_data['output'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
                                },
                    'pred': {'vorticity': save_data['pred'][0, ..., 0].permute(2, 0, 1).cpu().numpy(),
                             'streamfunction': save_data['pred'][0, ..., 1].permute(2, 0, 1).cpu().numpy(),
                             }
                }
                print("save_data_numpy shape", save_data_numpy['input']['vorticity'].shape, 
                      save_data_numpy['output']['vorticity'].shape, 
                      save_data_numpy['pred']['vorticity'].shape)
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
    save_data = predict_and_save(model, test_loader, save=True, log_path=log_path)
    
    # #### 2. load the save_data
    # save_data = torch.load(f'{log_path}/test_data_prediction.pth', map_location=device)
    
    # #### 3. compute different types of metrics
    # compute_evalutation_metrics(save_data, model_name=args.model, log_path=log_path)

    #### 4. postprocessing save the data for diffusion training
    # no_postprocessing_pred_save_data(args)


    #### 5. plot the spectral error
    # plot_spectral_error(save_data, model_name=args.model, log_path=log_path)