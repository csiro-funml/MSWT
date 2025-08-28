# TEST THE Predictions of the model

import sys
import os
# sys.path.append(['.','./../'])
# os.environ['OMP_NUM_THREADS'] = '16'

import json
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from timeit import default_timer
from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR,  CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer import Adam, Lamb
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint, resume_training_from_checkpoint

from utils.griddataset import MixedTemporalDataset, TemporalDataset2D, TemporalDataset2D_multiscale
from utils.make_master_file import DATASET_DICT
from models.fno import FNO2d
from models.uno import UNO

from models.wavelet_transform import CrossWaveletTransformer
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm


class SimpleLpLoss(nn.Module):
    def __init__(self):
        super(SimpleLpLoss, self).__init__()
        
    
    def forward(self, pred, yy):
        """
        pred: (B, N, C)
        yy: (B, N, C)
        loss_bc = norm(pred - yy, dim=1) / norm(yy, dim=1)
        loss = torch.mean(loss_bc) # averge over batch size and number of channels
        """
        # compute the average step loss
        # diff_norm = torch.norm(pred - yy,p=2,dim=1)
        # yy_norm = torch.norm(yy,p=2,dim=1) + 1e-8
        diff_norm = torch.sum((pred - yy)**2, dim=1)
        yy_norm = torch.sum(yy**2, dim=1) + 1e-8
        loss_bc = torch.sqrt(diff_norm / yy_norm)
        return torch.mean(loss_bc)


class FourierLoss(nn.Module):
    def __init__(self):
        super(FourierLoss, self).__init__()
        self.lp_loss = SimpleLpLoss()
        self.beta = 0.1
        
    def forward(self, pred, target):
        pred_loss = self.lp_loss(pred, target)
        print("pred_loss", pred_loss.item())

        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        fft_loss = self.lp_loss(pred_fft, target_fft)
        print("fft_loss", fft_loss.item())
        
        loss = pred_loss + self.beta * fft_loss
        return loss

################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='FNO') # FNO, ViT, UNO, CNO, Oformer, Transolver, DPOT, Crossformer, wavelet_transformer
parser.add_argument('--dataset',type=str, default='ns2d_pda') # ['ns2d_fno_1e-3', 'ns2d_pda', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 'sw2d_pda'], note: pdb is the pde bench
parser.add_argument('--resume_path',type=int, default=1)
parser.add_argument('--use_writer', action='store_true',default=False)



# ### dataset details
parser.add_argument('--T_in', type=int, default=7)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--pad', type=int, default=0)
parser.add_argument('--normalize',type=int, default=1)


# ### FNO/UNO params 
parser.add_argument('--n_layers',type=int, default=4)
parser.add_argument('--modes', type=int, default=12)
# parser.add_argument('--modes', type=int, default=32)
parser.add_argument('--width', type=int, default=32)
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


def load_data_model():
    ################################################################
    # load data and dataloader
    train_dataset = TemporalDataset2D(args.dataset,  t_in=args.T_in, t_ar=-1, train='train', normalize=args.normalize ) # just need this to get the num of training samples, and n_channels
    test_dataset = TemporalDataset2D(args.dataset, n_train=260, t_in=args.T_in, t_ar=-1, train='test', normalize=args.normalize)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
    
    ntrain, ntest = len(train_dataset), len(test_dataset)
    print(args.dataset)
    print('Train num {}, Test num {}'.format(ntrain, ntest))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
    # val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)

    # ntrain, ntest = len(train_dataset), len(test_dataset)

    if not args.pad:
        args.res = train_dataset.res  # use original dataset  resolution to train the model

    comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
    log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
    model_path = log_path + '/model.pth'
    print(model_path)
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
        model = CrossWaveletTransformer(wave='haar', n_channels=train_dataset.n_channels, in_timesteps = args.T_in, dim=512).to(device)
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

    myloss = SimpleLpLoss() # L2 loss
    return model, test_loader, myloss, log_path

################################################################
# Function 1 Report Average step, step-wise, and full prediction relative l2 norm
################################################################
def test_error(model, test_loader, myloss, save=False, log_path=None):
    """
    test_error(model, test_loader, myloss, save=False)
    Args:
        model: the model to test
        test_loader: the test loader
        myloss: the loss function
        save: whether to save the results to a csv file
    Returns:
        pd_data: a pandas dataframe containing the test error
        np_data = {'input': np.ndarray, 'output': np.ndarray, 'pred': np.ndarray} for later visualization 
    """
    with torch.no_grad():
        model.eval()
        test_dataset = test_loader.dataset
        max_steps = test_dataset[0][1].shape[-2]

        test_l2_per_step = {t: [] for t in range(0, max_steps)}
        test_l2_per_step['full'] = []    
        test_l2_per_step['step'] = []

        np_data = {'input': [], 'output': [], 'pred': []}
        # autoregressive computing  
        for xx, yy in test_loader:
            np_data['input'].append(xx.cpu().numpy())
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
            np_data['output'].append(yy.cpu().numpy())
            np_data['pred'].append(pred.cpu().numpy())

            # compute the step and total loss here
            #1. compute the average step loss
            # reshape the pred and yy to (B*T, H*W, C)
            pred_step = rearrange(pred, 'b h w t c -> (b t) (h w) c')
            yy_step = rearrange(yy, 'b h w t c -> (b t) (h w) c') # (B* T_out, H*W, C)
            avg_step_loss = myloss(pred_step, yy_step)
            test_l2_per_step['step'].append(avg_step_loss.item())

            #2. compute the total loss for the entire prediction
            pred_full = rearrange(pred, 'b h w t c -> b (t h w) c')
            yy_full = rearrange(yy, 'b h w t c -> b (t h w) c')
            full_loss = myloss(pred_full, yy_full)
            test_l2_per_step['full'].append(full_loss.item())

            #3. compute the step-wise loss (no averaging)
            for t in range(0, max_steps):
                pred_step = rearrange(pred[..., t:t+1, :], 'b h w t c -> (b t) (h w) c')
                yy_step = rearrange(yy[..., t:t+1, :], 'b h w t c -> (b t) (h w) c')
                step_loss = myloss(pred_step, yy_step)
                test_l2_per_step[t].append(step_loss.item())
           
        

        # Initialize a pandas dataframe to store everything
        pd_data = {}
        # print average loss per step
        for t in range(0, yy.shape[-2]):
            print("average loss per step", t, np.mean(test_l2_per_step[t]))            
            pd_data[f'step_{t}'] = np.mean(test_l2_per_step[t])

        # compute the average loss acrss all steps
        test_l2_per_step_avg = np.mean(test_l2_per_step['step'])
        print("average loss across all steps", test_l2_per_step_avg)
        pd_data['step_avg'] = test_l2_per_step_avg

        # compute the average loss for the entire prediction
        test_l2_full_pred_avg = np.mean(test_l2_per_step['full'])
        print("average loss for the entire prediction", test_l2_full_pred_avg)
        pd_data['full_avg'] = test_l2_full_pred_avg

        # save to csv
        if save:
            pd.Series(pd_data).to_csv(f'{log_path}/test_l2_norm_step_full.csv')
        

        # organzie np_data
        np_data['input'] = np.concatenate(np_data['input'], axis=0)
        np_data['output'] = np.concatenate(np_data['output'], axis=0)
        np_data['pred'] = np.concatenate(np_data['pred'], axis=0)

        # print the shape of the np_data
        print("np_data shape", np_data['input'].shape, np_data['output'].shape, np_data['pred'].shape)

        # save to npz
        if save:
            np.savez(f'{log_path}/test_data.npz', input=np_data['input'], output=np_data['output'], pred=np_data['pred'])
    return pd_data, np_data


################################################################
# Function 2 draw_rollout_error
################################################################
   
    # compute the power energy
def compute_power(x, device="cpu"):
    # print("input shape",x.shape)
    H, W = x.shape
    DTYPE = torch.float32

    # image = x.permute(2, 0, 1)     # (C, H, W)
    image = x
    nx, ny = image.shape[-2:] 
    L = 2*np.pi
    N = H
    delta_x = L/N
    k_nq = np.pi/delta_x
    
    # Compute the Fourier transform and get the amplitude squared
    fourier_image = torch.fft.fft2(image, dim=(-2, -1)) # (C, H, W//2)
    fourier_amplitudes = torch.sum(torch.abs(fourier_image), dim=0) if len(fourier_image.shape) == 3 else torch.abs(fourier_image)
    # print("fourier_amplitudes shape",fourier_amplitudes.shape)

    # Create the k-frequency grids
    kfreq_y = torch.fft.fftfreq(ny) * ny
    kfreq_x = torch.fft.fftfreq(nx) * nx
    kfreq2D_x, kfreq2D_y = torch.meshgrid(kfreq_x, kfreq_y, indexing='ij')
    
    # Compute the wavenumber grid
    knrm = torch.sqrt(kfreq2D_x ** 2 + kfreq2D_y ** 2).to(x.device)
    
    # Flatten the arrays to use in binning
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    # print("fourier_amplitudes shape",fourier_amplitudes.shape)
    # print("knrm shape", knrm.shape)

    # Define the bins for the wavenumber
    kbins = np.arange(0.5, ny//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    kvals = kvals / (0.5*nx) * k_nq
    
    # Bin the data
    Abins, _, _ = stats.binned_statistic(knrm.detach().cpu().numpy(), fourier_amplitudes.detach().cpu().numpy(),
                                            statistic="mean",
                                            bins=kbins)
    
    # Scale the binned amplitudes
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    # print(Abins.shape)
    return kvals, Abins


def compute_energy(x, N, k_nq):
    """
    compute_energy(x, N, k_nq)
    Args:
        x: the input image, shape (H, W)
        N: the number of pixels in the image
        k_nq: the number of wavenumbers in the image
    Returns:
        kvals: the wavenumbers
    """
    #######################
    image = x # (H, W)
    H, W = image.shape
    
    # Compute the Fourier transform and get the amplitude squared
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)#**2
    
    # Create the k-frequency grid for rectangular image
    kfreq_x = np.fft.fftfreq(W) * W
    kfreq_y = np.fft.fftfreq(H) * H
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    # knrm = np.abs(kfreq2D[0]) + np.abs(kfreq2D[1])
    
    # Flatten the arrays to use in binning
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    
    # Define the bins for the wavenumber - use the minimum dimension for binning
    min_dim = min(H, W)
    kbins = np.arange(0.5, min_dim//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    kvals = kvals / (0.5*N) * k_nq
    
    # Bin the data
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                            statistic="mean",
                                            bins=kbins)
    
    # Scale the binned amplitudes
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    

    return kvals, Abins, kbins



def draw_rollout_error(np_data, channel_id=0, log_path='./logs'):
    """
    draw_rollout_error(np_data)
    Args:
        np_data: a dictionary containing the input, output, and pred
        shape of input, output, pred: (B, H, W, T, C)
    Returns:
        None
    """
    # load the data from np_data (three channels: representing the concentrationn field (d) and the velocities (vx, vy))
    x_true = np_data['input'][...,channel_id].transpose(0, 3, 1, 2)  # (B, T_in, H, W)
    y_true = np_data['output'][...,channel_id].transpose(0, 3, 1, 2) # (B, T_out, H, W)
    y_pred = np_data['pred'][...,channel_id].transpose(0, 3, 1, 2)  
    y_error = y_true - y_pred # (B, T_out, H, W)
    print("input_data shape", x_true.shape, "ground_truth shape", y_true.shape, "pred shape", y_pred.shape, "error shape", y_error.shape)

    L = 2*np.pi
    N = np.min(y_true.shape[2:])
    delta_x = L/N
    k_nq = np.pi/delta_x
    print(f"k_nq: {k_nq}")

    # for p in range(10):
    for p in [10]:
        print("sample_id", p)
        sample_id = p

        # Set 4 snapshots spanning the entire time range
        total_time_steps = y_true.shape[1]  # T_out
        t_idx = np.linspace(0, total_time_steps-1, 4, dtype=int)  # 4 evenly spaced indices
        
        init = x_true[sample_id, 3] # the last time step of the initial condition
        true = y_true[sample_id] # (T_out, H, W)
        pred = y_pred[sample_id] # (T_out, H, W)
        error = y_error[sample_id]
        
        print("true shape for channel", channel_id, true.shape)

        # Use RdBu_r colormap - white will naturally be at zero with symmetrical vmin/vmax
        cmap = 'RdBu_r'

        
        # Create the figure and axis
        fig, axes = plt.subplots(4, len(t_idx)+1, figsize=(18, 11))
        # add a colorbar to the right hand side of the figure
        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # Colorbar axis on the right side


        # Find the min and max values for colorbar and make them symmetrical around zero
        data_min = min(true.min(), pred.min())
        data_max = max(true.max(), pred.max())
        abs_max = max(abs(data_min), abs(data_max))
        vmin = -abs_max
        vmax = abs_max

        for ax in axes[:, 0]:  # This loops over the first column from the second row onwards
            ax.axis('off')  # Turn these axes off as they are not used
        
        # Plot the images
        im = axes[0, 0].imshow(init, vmin=init.min(), vmax=init.max(), cmap=cmap) # plot intial condition
        axes[0, 0].set_title(f'Initial Concentration', fontsize=16)
        axes[0, 0].axis('off') 
        
        
        for i in range(0, len(t_idx)): # one extra column for the initial condition
            col_idx = i + 1
            time_idx = t_idx[i]

            # Ground truth row
            im = axes[0, col_idx].imshow(true[time_idx], vmin=vmin, vmax=vmax, cmap=cmap)
            axes[0, col_idx].set_title(f't: {time_idx+1}', fontsize=16)
            axes[0, col_idx].axis('off')
        
            # Prediction 1 row
            im = axes[1, col_idx].imshow(pred[time_idx], vmin=vmin, vmax=vmax, cmap=cmap)
            axes[1, col_idx].set_title(f'Pred', fontsize=16) if i ==0 else None
            # axes[1, i].set_title(f'rel L2: {err:.2e}', fontsize=12)
            axes[1, col_idx].axis('off')

            # Error row
            im = axes[2, col_idx].imshow(error[time_idx], vmin=vmin, vmax=vmax, cmap=cmap)
            axes[2, col_idx].set_title(f'Error', fontsize=16) if i ==0 else None
            axes[2, col_idx].axis('off')


            # compute the energy
            kvals, Abins, kbins = compute_energy(true[time_idx], N, k_nq)
            axes[3, col_idx].loglog(kvals, Abins, c='black', label="Ground Truth", linewidth=2)


            # compute the energy
            kvals, Abins, _ = compute_energy(pred[time_idx], N, k_nq)
            axes[3, col_idx].loglog(kvals, Abins, c='blue', label="Wavelet Pred", linewidth=2)
            axes[3, col_idx].set_xlabel('$k$', fontsize=10) 
            axes[3, col_idx].set_ylabel('Energy', fontsize=14) if i ==0 else None
            axes[3, col_idx].set_title(f'Energy Spectrum', fontsize=14) if i ==0 else None
            # axes[3, col_idx].axis('off')
            # axes[3, col_idx].legend()

            

        leg = axes[3, col_idx].legend()
        # axes[3, i1].legend(loc='upper \left', bbox_to_anchor=(1.05, 1), frameon=False)
        handles, labels = axes[3, col_idx].get_legend_handles_labels()

        # Remove the original legend from the axes
        leg.remove()
        
        # Define custom coordinates for the new legend (adjust these values to your liking)
        custom_x = 0.89  # Adjust this value to move left-right
        custom_y = 0.1  # Adjust this value to move up-down
        
        # Create a new standalone legend at the custom coordinates on the figure
        fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(custom_x, custom_y), fontsize=10)




    # fig.tight_layout(rect=[0, 0, 0.9, 1], pad=1.5)
    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Leave 10% space on the right for colorbar
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f'{log_path}/test_rollout_error.png')
    plt.show()
    print("done")
    
    # For each column/ time step
    # true, pred, error, shape (H, W, T_out, 3)



def draw_rollout_error_comparison(np_data1, np_data2, channel_id=0, log_path1=None, log_path2=None, model_name1=None, model_name2=None, sample_ids=None, dataset=None):
    """
    draw_rollout_error_comparison(np_data1, np_data2)
    Args:
        np_data1: a dictionary containing the input, output, and pred
        np_data2: a dictionary containing the input, output, and pred
        shape of input, output, pred: (B, H, W, T, C)
    Returns:
        None
    """
    # Use RdBu_r colormap - white will naturally be at zero with symmetrical vmin/vmax
    custom_cmap = 'RdBu_r'
    # load the data from np_data (three channels: representing the concentrationn field (d) and the velocities (vx, vy))
    x_true = np_data1['input'][...,channel_id].transpose(0, 3, 1, 2)  # (B, T_in, H, W)
    y_true = np_data1['output'][...,channel_id].transpose(0, 3, 1, 2) # (B, T_out, H, W)
    y_pred1 = np_data1['pred'][...,channel_id].transpose(0, 3, 1, 2)  
    y_pred2 = np_data2['pred'][...,channel_id].transpose(0, 3, 1, 2)  
    if y_pred2.shape[0] > y_pred1.shape[0]:
        y_pred2 = y_pred2[:y_pred1.shape[0], :, :, :]
    

    # truncate the data to the last 20 time steps
    y_true = y_true[:, :-20, :, :] if y_true.shape[1] > 20 else y_true
    y_pred1 = y_pred1[:, :-20, :, :] if y_pred1.shape[1] > 20 else y_pred1
    y_pred2 = y_pred2[:, :-20, :, :] if y_pred2.shape[1] > 20 else y_pred2

    y_error1 = y_true - y_pred1 # (B, T_out, H, W)
    y_error2 = y_true - y_pred2 # (B, T_out, H, W)


    print("input_data shape", x_true.shape, "ground_truth shape", y_true.shape, "pred shape", y_pred1.shape, "error shape", y_error1.shape)

    L = 2*np.pi
    N = np.min(y_true.shape[2:])
    delta_x = L/N
    k_nq = np.pi/delta_x
    print(f"k_nq: {k_nq}")


    # for p in range(10):
    for p in sample_ids:
        print("sample_id", p)
        sample_id = p

        # Set 4 snapshots spanning the entire time range, starting from 1 th time step
        total_time_steps = y_true.shape[1]
        t_idx = np.linspace(1, total_time_steps-1, 4, dtype=int)  # 4 evenly spaced indices
        # check if len(t_idx) is 4

        init = x_true[sample_id, 3] # the last time step of the initial condition
        true = y_true[sample_id] # (T_out, H, W)
        pred1 = y_pred1[sample_id] # (T_out, H, W)
        pred2 = y_pred2[sample_id] # (T_out, H, W)
        error1 = y_error1[sample_id]
        error2 = y_error2[sample_id]
        
        print("true shape for channel", channel_id, true.shape)
        
        # Create the figure and axis
        fig, axes = plt.subplots(6, len(t_idx)+2, figsize=(14, 11)) if dataset == 'ns2d_pda' else plt.subplots(6, len(t_idx)+2, figsize=(14, 8))
       
        # add a colorbar to the right hand side of each figure 

        # Find the min and max values for colorbar and make them symmetrical around zero
        data_min_pred = min(true[-1].min(), pred1.min(), pred2.min()) # only use the last slice
        data_max_pred = max(true[-1].max(), pred1.max(), pred2.max()) # only use the last slice
        abs_max_pred = max(abs(data_min_pred), abs(data_max_pred))
        
        vmin_pred = data_min_pred - 0.01 * abs(data_min_pred)
        vmax_pred = abs_max_pred + 0.01 * abs(abs_max_pred)
        # vmax_pred = min(vmax_pred, 15)
        
        # Keep separate vmin/vmax for errors (not symmetrical)
        vmin_error = min(error1.min(), error2.min()) # only use the last slice
        vmax_error = max(error1.max(), error2.max()) # only use the last slice
        abs_max_error = max(abs(vmin_error), abs(vmax_error))
        abs_max_error = min(abs_max_error, 15) # limit the max error to 15
        vmin_error = -abs_max_error
        vmax_error = abs_max_error

        # verro_max = max(error1.max(), error2.max())
        # verro_min = min(error1.min(), error2.min())
        # print("verro_max", verro_max, "verro_min", verro_min)
        for ax in axes[:, 0]:  # This loops over the first column from the second row onwards
            ax.axis('off')  # Turn these axes off as they are not used
        
        # Plot the images
        im = axes[0, 0].imshow(init, vmin=vmin_pred, vmax=vmax_pred, cmap=custom_cmap) # plot intial condition
        axes[0, 0].set_title(f'Ground Truth', fontsize=20)
        axes[0, 0].axis('off') 
        
        # add text in the axes instead of the ylabel
        # axes[0, 0].text(0.5, 0.5, 'Ground Truth', fontsize=16, ha='center', va='center')
        axes[1, 0].text(0.5, 0.5, f'{model_name1} Pred', fontsize=16, ha='center', va='center')
        axes[2, 0].text(0.5, 0.5, f'{model_name1} Error', fontsize=16, ha='center', va='center')
        axes[3, 0].text(0.5, 0.5, f'{model_name2} Pred', fontsize=16, ha='center', va='center')
        axes[4, 0].text(0.5, 0.5, f'{model_name2} Error', fontsize=16, ha='center', va='center')
        axes[5, 0].text(0.5, 0.5, 'Energy Spectrum', fontsize=16, ha='center', va='center')
        

        for i in range(0, len(t_idx)): # one extra column for the initial condition
            col_idx = i + 1
            time_idx = t_idx[i]

            # Ground truth row
            im = axes[0, col_idx].imshow(true[time_idx], vmin=vmin_pred, vmax=vmax_pred, cmap=custom_cmap)
            axes[0, col_idx].set_title(f't: {time_idx}', fontsize=20)
            axes[0, col_idx].axis('off')
        

            # Prediction 1 row
            im = axes[1, col_idx].imshow(pred1[time_idx], vmin=vmin_pred, vmax=vmax_pred, cmap=custom_cmap)
            # axes[1, col_idx].set_title(f'FNO Pred', fontsize=16) if i ==0 else None
            # axes[1, i].set_title(f'rel L2: {err:.2e}', fontsize=12)
            axes[1, col_idx].axis('off')

            # Error row
            im = axes[2, col_idx].imshow(error1[time_idx], vmin=vmin_error, vmax=vmax_error, cmap=custom_cmap)
            # axes[2, col_idx].set_title(f'Error', fontsize=16) if i ==0 else None
            axes[2, col_idx].axis('off')

            # Prediction 2 row
            im = axes[3, col_idx].imshow(pred2[time_idx], vmin=vmin_pred, vmax=vmax_pred, cmap=custom_cmap)
            # axes[3, col_idx].set_title(f'Wavelet Pred', fontsize=16) if i ==0 else None
            axes[3, col_idx].axis('off')

            im = axes[4, col_idx].imshow(error2[time_idx], vmin=vmin_error, vmax=vmax_error, cmap=custom_cmap)
            # axes[4, col_idx].set_title(f'Wavelet Error', fontsize=16) if i ==0 else None
            axes[4, col_idx].axis('off')


            # compute the energy
            kvals, Abins, kbins = compute_energy(true[time_idx], N, k_nq)
            axes[5, col_idx].loglog(kvals, Abins, c='black', label="Ground Truth", linewidth=2)


            # compute the energy
            kvals, Abins, _ = compute_energy(pred1[time_idx], N, k_nq)
            axes[5, col_idx].loglog(kvals, Abins, c='blue', label=f"{model_name1} Pred", linewidth=2)
            axes[5, col_idx].set_xlabel('$k$', fontsize=10) 
            # axes[5, col_idx].set_ylabel('Energy', fontsize=14) if i ==0 else None
            # axes[5, col_idx].set_title(f'Energy Spectrum', fontsize=14) if i ==0 else None

            # compute the energy
            kvals, Abins, _ = compute_energy(pred2[time_idx], N, k_nq)
            axes[5, col_idx].loglog(kvals, Abins, c='red', label=f"{model_name2} Pred", linewidth=2)

            # axes[3, col_idx].axis('off')
            # axes[3, col_idx].legend()

        # Add colorbars in the last column
        last_col = len(t_idx) + 1
        
        # Colorbar for predictions (rows 0, 1, 3)
        for row in [0, 1, 3]:
            axes[row, last_col].axis('off')
            # Create a dummy image for the colorbar
            dummy_data = np.linspace(vmin_pred, vmax_pred, 100).reshape(10, 10)
            im_pred = axes[row, last_col].imshow(dummy_data, vmin=vmin_pred, vmax=vmax_pred, cmap=custom_cmap)
            im_pred.set_visible(False)  # Hide the dummy image
            axes[row, last_col].set_aspect('auto')
            # Add colorbar
            cbar = plt.colorbar(im_pred, ax=axes[row, last_col], fraction=1.0, shrink=1.0)
            cbar.ax.tick_params(labelsize=8)
        
        # Colorbar for errors (rows 2, 4)
        for row in [2, 4]:
            axes[row, last_col].axis('off')
            # Create a dummy image for the colorbar
            dummy_data = np.linspace(vmin_error, vmax_error, 100).reshape(10, 10)
            im_error = axes[row, last_col].imshow(dummy_data, vmin=vmin_error, vmax=vmax_error, cmap=custom_cmap)
            im_error.set_visible(False)  # Hide the dummy image
            
            # Add colorbar
            cbar = plt.colorbar(im_error, ax=axes[row, last_col], fraction=1.0, shrink=1.0)
            cbar.ax.tick_params(labelsize=8)
        
        # Empty last column for energy spectrum row
        axes[5, last_col].axis('off')
        
        # Add legend for energy spectrum
        axes[5, 1].legend(loc='lower left', fontsize=9)


        fig.tight_layout()
        plt.savefig(f'{log_path1}/{dataset}_test_rollout_error_comparison.png')
        plt.show()
    
    # For each column/ time step
    # true, pred, error, shape (H, W, T_out, 3)


 

################################################################
# Function 3 Compare Energy with ground truth PDEs (multiscale features)
################################################################


def compare_multiscale_features(np_data, channel_id=0, log_path=None, dataset=None):
    """
    decompose the input data via FFTs and compare the energy spectrum with the ground truth PDEs (2 dimensions)

    """
    input = np_data['input'][...,channel_id] 
    output = np_data['output'][...,channel_id] # (B, T_out, H, W)
    pred = np_data['pred'][...,channel_id] # (B, T_out, H, W)   

    def get_energy_spectrum(x):
        print("x shape", x.shape)
        # Compute the Fourier transform and get the amplitude squared
        fourier_image = np.fft.fftn(x)
        fourier_amplitudes = np.abs(fourier_image)#**2
        fourier_amplitudes = np.fft.fftshift(fourier_amplitudes) # shift the zero frequency to the center
        log_fourier_amplitudes = np.log(fourier_amplitudes)
        print("fourier_amplitudes shape", fourier_amplitudes.shape)
        H, W = x.shape[0], x.shape[1]
        # Create the k-frequency grid
        kfreqx = np.fft.fftfreq(W) * W
        kfreqy = np.fft.fftfreq(H) * H
        kfreqx = np.fft.fftshift(kfreqx) # shift the zero frequency to the center
        kfreqy = np.fft.fftshift(kfreqy) # shift the zero frequency to the center
        # Create 2D frequency grids
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreqx, kfreqy, indexing='xy')
        fourier_freq = np.stack([kfreq2D_x, kfreq2D_y], axis=0)
        print("fourier_freq shape", fourier_freq.shape)
        return fourier_freq, log_fourier_amplitudes
    

    data_idx = [10]
    total_steps = 5
    step_interval = output.shape[-1]//total_steps
    fig, axes = plt.subplots(2,total_steps, figsize=(18, 11))
    
    # Create custom colormap with specific transitions
    colors = ['blue', 'lightblue', 'white', 'pink', 'red']
    positions = np.linspace(-1, 1, len(colors))
    cmap = LinearSegmentedColormap.from_list('custom_diverging', list(zip(positions, colors)))
    
    # Set global min and max values
    vmin = -3
    vmax = 8
    
    
    # Define custom tick positions and labels based on image dimensions
    H, W = output.shape[1], output.shape[2]  # Get actual image dimensions
    max_freq_x = W // 2
    max_freq_y = H // 2
    
    # Create tick values for x and y axes based on actual dimensions
    tick_values_x = np.linspace(-max_freq_x, max_freq_x, 9, dtype=int)
    tick_values_y = np.linspace(-max_freq_y, max_freq_y, 9, dtype=int)
    
    # cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4]) # put it to right hand side of the figure # Colorbar axis

    for i in data_idx:
        for t in range(0, total_steps):
            output_spec = get_energy_spectrum(output[i, ... , t*step_interval])
            pred_spec = get_energy_spectrum(pred[i, ... , t*step_interval])

            # Convert frequency values to pixel positions for x and y axes separately
            freq_range_x = output_spec[0][0][0, :]  # x frequencies (first row)
            freq_range_y = output_spec[0][1][:, 0]  # y frequencies (first column)
            n_pixels_x = len(freq_range_x)
            n_pixels_y = len(freq_range_y)
            
            # Map frequency values to pixel positions for x and y axes
            tick_positions_x = [(val - freq_range_x[0]) * (n_pixels_x - 1) / (freq_range_x[-1] - freq_range_x[0]) if freq_range_x[-1] != freq_range_x[0] else 0 for val in tick_values_x]
            tick_positions_y = [(val - freq_range_y[0]) * (n_pixels_y - 1) / (freq_range_y[-1] - freq_range_y[0]) if freq_range_y[-1] != freq_range_y[0] else 0 for val in tick_values_y]


            # Plot ground truth with adjusted colormap settings
            im0 = axes[0, t].imshow(output_spec[1], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[0, t].set_xticks(tick_positions_x)
            axes[0, t].set_yticks(tick_positions_y)
            axes[0, t].set_xticklabels(tick_values_x)
            axes[0, t].set_yticklabels(tick_values_y)
            axes[0, t].set_title(f'Ground Truth t={t*step_interval}')

            # Plot prediction with the same colormap settings
            im1 = axes[1, t].imshow(pred_spec[1], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1, t].set_xticks(tick_positions_x)
            axes[1, t].set_yticks(tick_positions_y)
            axes[1, t].set_xticklabels(tick_values_x)
            axes[1, t].set_yticklabels(tick_values_y)
            axes[1, t].set_title(f'Prediction t={t*step_interval}')

            if t == 0:  # Only add labels to the first column
                axes[0, t].set_ylabel('Frequency')

                axes[1, t].set_ylabel('Frequency')
        
    


        # plt.colorbar(im1, ax=axes[1, t])
     # fig.tight_layout(rect=[0, 0, 0.9, 1], pad=1.5)
    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Leave 10% space on the right for colorbar
    # fig.colorbar(im1, cax=cbar_ax)
    # plt.savefig(f'{log_path}/{dataset}_test_multiscale_features.png')
    plt.show()           
    return



def compare_multiscale_features_comparison(np_data1, np_data2, channel_id=0, log_path1=None, log_path2=None, sample_ids=None, dataset=None):
    """
    decompose the input data via FFTs and compare the energy spectrum with the ground truth PDEs (2 dimensions)

    """
    input = np_data1['input'][...,channel_id] 
    output = np_data1['output'][...,channel_id] # (B, T_out, H, W)
    pred1 = np_data1['pred'][...,channel_id] # (B, T_out, H, W)   
    pred2 = np_data2['pred'][...,channel_id] # (B, T_out, H, W)   
    # truncate the data to the last 20 time steps
    output = output[:, :, :, :-20] if output.shape[-1] > 20 else output
    pred1 = pred1[:, :, :, :-20] if pred1.shape[-1] > 20 else pred1
    pred2 = pred2[:, :, :, :-20] if pred2.shape[-1] > 20 else pred2
    
    def get_energy_spectrum(x):
        print("x shape", x.shape)
        # Compute the Fourier transform and get the amplitude squared
        fourier_image = np.fft.fftn(x)
        fourier_amplitudes = np.abs(fourier_image)#**2
        fourier_amplitudes = np.fft.fftshift(fourier_amplitudes) # shift the zero frequency to the center
        log_fourier_amplitudes = np.log(fourier_amplitudes)
        print("fourier_amplitudes shape", fourier_amplitudes.shape)
        H, W = x.shape[0], x.shape[1]
        # Create the k-frequency grid
        kfreqx = np.fft.fftfreq(W) * W
        kfreqy = np.fft.fftfreq(H) * H
        kfreqx = np.fft.fftshift(kfreqx) # shift the zero frequency to the center
        kfreqy = np.fft.fftshift(kfreqy) # shift the zero frequency to the center
        # Create 2D frequency grids
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreqx, kfreqy, indexing='xy')
        fourier_freq = np.stack([kfreq2D_x, kfreq2D_y], axis=0)
        print("fourier_freq shape", fourier_freq.shape)
        return fourier_freq, log_fourier_amplitudes
    

    data_idx = sample_ids
    total_time_steps = output.shape[-1]
    t_idx = np.linspace(1, total_time_steps-1, 4, dtype=int)  # 4 evenly spaced indices
    
    # adjust the figure size based on the total_steps
    # fig, axes = plt.subplots(3,len(t_idx), figsize=(18, 11))    
    fig, axes = plt.subplots(3,len(t_idx), figsize=(18, 11)) if dataset == 'ns2d_pda' else plt.subplots(3,len(t_idx), figsize=(18, 8))
    # Create custom colormap with specific transitions
    cmap = plt.cm.PuOr_r # Red-Blue reversed (white in middle)
    

    # Set global min and max values
    vmin = -3
    vmax = 8
    
    
    # Define custom tick positions and labels based on image dimensions
    H, W = output.shape[1], output.shape[2]  # Get actual image dimensions
    max_freq_x = W // 2
    max_freq_y = H // 2
    
    # Create tick values for x and y axes based on actual dimensions
    tick_values_x = np.linspace(-max_freq_x, max_freq_x, 9, dtype=int)
    tick_values_y = np.linspace(-max_freq_y, max_freq_y, 9, dtype=int)
    
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4]) # put it to right hand side of the figure # Colorbar axis

    for i in data_idx:
        for t in range(0, len(t_idx)):
            output_spec = get_energy_spectrum(output[i, ... , t_idx[t]])
            pred_spec1 = get_energy_spectrum(pred1[i, ... , t_idx[t]])
            pred_spec2 = get_energy_spectrum(pred2[i, ... , t_idx[t]])

            # Convert frequency values to pixel positions for x and y axes separately
            # Get the 1D frequency ranges (first row/column of the 2D grids)
            freq_range_x = output_spec[0][0][0, :]  # x frequencies (first row)
            freq_range_y = output_spec[0][1][:, 0]  # y frequencies (first column)
            n_pixels_x = len(freq_range_x)
            n_pixels_y = len(freq_range_y)
            
            # Map frequency values to pixel positions for x and y axes
            tick_positions_x = [(val - freq_range_x[0]) * (n_pixels_x - 1) / (freq_range_x[-1] - freq_range_x[0]) if freq_range_x[-1] != freq_range_x[0] else 0 for val in tick_values_x]
            tick_positions_y = [(val - freq_range_y[0]) * (n_pixels_y - 1) / (freq_range_y[-1] - freq_range_y[0]) if freq_range_y[-1] != freq_range_y[0] else 0 for val in tick_values_y]


            # Plot ground truth with adjusted colormap settings
            im0 = axes[0, t].imshow(output_spec[1], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[0, t].set_xticks(tick_positions_x)
            axes[0, t].set_yticks(tick_positions_y)
            axes[0, t].set_xticklabels(tick_values_x)
            axes[0, t].set_yticklabels(tick_values_y)
            axes[0, t].set_title(f'Ground Truth t={t_idx[t]}')

            # Plot prediction with the same colormap settings
            im1 = axes[1, t].imshow(pred_spec1[1], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1, t].set_xticks(tick_positions_x)
            axes[1, t].set_yticks(tick_positions_y)
            axes[1, t].set_xticklabels(tick_values_x)
            axes[1, t].set_yticklabels(tick_values_y)
            axes[1, t].set_title(f'FNO Prediction t={t_idx[t]}')


            im2 = axes[2, t].imshow(pred_spec2[1], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[2, t].set_xticks(tick_positions_x)
            axes[2, t].set_yticks(tick_positions_y)
            axes[2, t].set_xticklabels(tick_values_x)
            axes[2, t].set_yticklabels(tick_values_y)
            axes[2, t].set_title(f'Wavelet Prediction t={t_idx[t]}')

            if t == 0:  # Only add labels to the first column
                axes[0, t].set_ylabel('Frequency')

                axes[1, t].set_ylabel('Frequency')
                axes[2, t].set_ylabel('Frequency')
    


        # plt.colorbar(im1, ax=axes[1, t])
        # fig.tight_layout(rect=[0, 0, 0.9, 1], pad=1.5)
        fig.tight_layout(rect=[0, 0, 0.9, 1])  # Leave 10% space on the right for colorbar
        fig.colorbar(im1, cax=cbar_ax)
        plt.savefig(f'{log_path1}/{dataset}_test_multiscale_features_comparison.png')
        plt.show()           
    return


def plot_resolution_invariant_results(log_path1, log_path2):
    x_grid  = np.arange(32, 129, 16)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # read the csv file
    df1 = pd.read_csv(f'{log_path1}/test_l2_step_full_avg.csv')
    df2 = pd.read_csv(f'{log_path2}/test_l2_step_full_avg.csv')
    
    # compare the full loss and step loss between FNO and Wavelet Transformer
    axes[0].plot(df1['res'], df1['test_l2_full_avg'], label='FNO full loss', marker='x')
    axes[0].plot(df2['res'], df2['test_l2_full_avg'], label='Wavelet Transformer full loss', marker='o')
    axes[0].scatter(64, df1[df1['res'] == 64]['test_l2_full_avg'].values[0], color='r', marker='x', label='Training Resolution', zorder=2)
    axes[0].scatter(64, df2[df2['res'] == 64]['test_l2_full_avg'].values[0], color='r', marker='o', zorder=2)

    # compare the step loss between FNO and Wavelet Transformer
    axes[1].plot(df1['res'], df1['test_l2_step_avg'], label='FNO step loss', marker='x')
    axes[1].plot(df2['res'], df2['test_l2_step_avg'], label='Wavelet Transformer step loss', marker='o')
    # marke the value as res=64 as it is the training resolution    
    axes[1].scatter(64, df1[df1['res'] == 64]['test_l2_step_avg'].values[0], color='r', marker='x', label='Training Resolution', zorder=2)
    axes[1].scatter(64, df2[df2['res'] == 64]['test_l2_step_avg'].values[0], color='r', marker='o', zorder=2)

    axes[0].set_xlabel('Resolution')
    axes[0].set_ylabel('Rollout relative L2 loss')
    axes[0].set_title('Resolution Invariant Results')
    axes[0].set_xticks(x_grid)
    axes[0].legend()
    axes[0].grid(True)
    axes[1].set_xlabel('Resolution')
    axes[1].set_ylabel('Step relative L2 loss')
    axes[1].set_title('Resolution Invariant Results')
    axes[1].set_xticks(x_grid)
    axes[1].legend()
    axes[1].grid(True)

    plt.savefig(f'{log_path1}/test_l2_step_full_avg.png')
    plt.show()


def compute_average_pooling(x, kW=None, kH=None):
    """
    input x: (B, C, H, W), 
    return an output: (B, C, H, W)
    where we use a kernel with (kW, kH) to average the input data,  use torch.nn.functional.avg_pool2d to do the average pooling
    and the stride is the same as the kernel size
    """
    if kW is None:
        kW = x.shape[-1]
    if kH is None:
        kH = x.shape[-2]
    H, W = x.shape[-2], x.shape[-1]
    # compute the average pooling
    print('x shape before pooling', x.shape)
    x = torch.nn.functional.avg_pool2d(x, kernel_size=(kH, kW), stride=(kH, kW))
    print('x shape after pooling', x.shape)

    if H != x.shape[1] or W != x.shape[2]:
        x = F.interpolate(x, scale_factor=(kH, kW), mode='nearest')
        x = x[..., :H, :W]
    return x
    


def plot_average_pooling_multiscale_features(data, channel_id=0, log_path=None) :
    """
    plot the average pooling multiscale features
    """
    print("data shape", data.shape) # (H, W, T_in, C)
    vmax, vmin = np.max(data[..., channel_id]), np.min(data[..., channel_id])
    custom_cmap = 'RdBu_r'
    
    # fig, axes = plt.subplots(figsize=(10, 5))
    # print the input data
    x_input = data[..., 0, channel_id] # (H, W, C)
    plt.imshow(x_input, vmin=vmin, vmax=vmax, cmap=custom_cmap) # plot intial condition
    plt.axis('off')
    plt.title('Input')
    plt.tight_layout()
    plt.savefig(f'{log_path}/input.png')
    plt.show()

    x_output = data[..., 3, channel_id] # (H, W, C)
    plt.imshow(x_output, vmin=vmin, vmax=vmax, cmap=custom_cmap)
    plt.axis('off')
    plt.title('Output')
    plt.tight_layout()
    plt.savefig(f'{log_path}/output.png')
    plt.show()

    # sample a few kernel sizes
    kernel_sizes = [1, 2, 4, 8, 10, 12, 14, 16, 20, 32, 64, 128]
    x_output_torch = torch.from_numpy(x_output)[None, None, ...]  # (B, C, H, W)
    for kernel_size in kernel_sizes:
        x_pooled = compute_average_pooling(x_output_torch, kernel_size, kernel_size)
        x_pooled = x_pooled.squeeze().numpy()
        print("x_pooled shape", x_pooled.shape)
        plt.imshow(x_pooled, vmin=vmin, vmax=vmax, cmap=custom_cmap)
        plt.axis('off')
        plt.title(f'Pooled with kernel size {kernel_size}')
        plt.tight_layout()
        plt.savefig(f'{log_path}/pooled_kernel_{kernel_size}.png')
        plt.show()


def plot_fourier_transform_multiscale_features(data, channel_id=0, log_path=None) :
    """
    plot the fourier transform multiscale features
    """
    print("data shape", data.shape) # (H, W, T_in, C)
    vmax, vmin = np.max(data[..., channel_id]), np.min(data[..., channel_id])
    
    x_input = data[..., 0, channel_id] # (H, W, C)



def compute_output_magnitude(y):
    # y shape (B, H, W, T, C)
    B, C = y.shape[0], y.shape[-1]
    y_reshape = y.reshape(B, -1, C)
    y_magnitude = torch.sqrt(torch.sum(y_reshape**2, dim=1)) # (B,C)
    return y_magnitude

def plot_megnitude_hist(loss_magnitude):
    # loss_magnitude shape (N, C)
    # plot the histogram of the loss magnitude
    # plot the histogram of the loss magnitude
    fig, axs = plt.subplots(loss_magnitude.shape[1], 1, figsize=(5, 5))
    for c in range(loss_magnitude.shape[1]):
        axs[c].hist(loss_magnitude[:, c].flatten(), bins=100)
        axs[c].set_title(f'Channel {c} ')
        axs[c].set_xlabel('Sample Magnitude')
        axs[c].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('logs/loss_magnitude_hist.png', dpi=300)
    plt.close()


def plot_prediction_gt_abserror(pred_data, sample_id=0, channel_id=0, model_name='FNO', log_path=None):
    print("saved_data shape", pred_data['pred'].shape, "pred_data.keys()", pred_data.keys())
    # keys are (input, output, pred), shape of  (B, H, W, T_in/out, C)
    
    pred = pred_data['pred'][sample_id, ... , channel_id].detach().cpu().numpy() # (H, W, T_out)
    target = pred_data['output'][sample_id, ... , channel_id].detach().cpu().numpy() # (H, W, T_out)
    print("sample_id", sample_id, "channel_id", channel_id, "pred shape", pred.shape, "target shape", target.shape)
    error = pred - target # (H, W, T_out)    
    
    # axes has two columns and four rows: 
   
    # use target min and max as vmin and vmax
    vmin = np.min(target)
    vmax = np.max(target)
    error_vmin = -0.23
    error_vmax = 0.23
    cmap = 'RdBu_r'
    fig, axes = plt.subplots(4, 2, figsize=(8, 10))
    # axes[0, 0] is the target at the first time step,
    axes[0, 0].imshow(target[..., 0], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0, 0].set_ylabel('GT T+1')
    # just turn off the ticks but not lables
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    #  axes[2, 0] is the target at the last time step,
    axes[2, 0].imshow(target[..., -1], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[2, 0].set_ylabel('GT T+T_out')
    axes[2, 0].set_xticks([])
    axes[2, 0].set_yticks([])
    
    # turn off axes  ofr axes[1,0] and [3,0]
    axes[1, 0].axis('off')
    axes[3, 0].axis('off')
    
    
    # axes[0, 1] to axes[1, 1] is the prediction and the abs error at the first time step
    cm0 = axes[0, 1].imshow(pred[..., 0], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0, 1].set_ylabel('Pred T+1')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    fig.colorbar(cm0, ax=axes[0, 1], location='right', anchor=(0, 0.3), shrink=0.7) # add colorbar
    
    cm1 = axes[1, 1].imshow(error[..., 0], cmap=cmap, vmin=error_vmin, vmax=error_vmax)
    axes[1, 1].set_ylabel('Error T+1')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    fig.colorbar(cm1, ax=axes[1, 1], location='right', anchor=(0, 0.3), shrink=0.7)

     # axes[2, 1] to axes[3, 1] is the prediction and the abs error at the last time step
    cm2 = axes[2, 1].imshow(pred[..., -1], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[2, 1].set_ylabel('Pred T+T_out')
    axes[2, 1].set_xticks([])
    axes[2, 1].set_yticks([])
    fig.colorbar(cm2, ax=axes[2, 1], location='right', anchor=(0, 0.3), shrink=0.7)

    cm3 = axes[3, 1].imshow(error[..., -1], cmap=cmap, vmin=error_vmin, vmax=error_vmax)
    axes[3, 1].set_ylabel('Error T+T_out')
    axes[3, 1].set_xticks([])
    axes[3, 1].set_yticks([])
    fig.colorbar(cm3, ax=axes[3, 1], location='right', anchor=(0, 0.3), shrink=0.7)
    
    # tight layout
    fig.tight_layout()
    plt.savefig(f'{log_path}/{model_name}_prediction_gt_error.png')
    plt.show()
    
    return pred, target, error
    # plot the prediction and ground truth and abs error
    # plt.imshow(pred[0, 0, ...], vmin=vmin, vmax=vmax, cmap=custom_cmap)
    # plt.axis('off')
    # plt.title('Prediction')
    # plt.tight_layout()


if __name__ == '__main__':
    
    """
    Obsolete code, keep for reference

    ################################################################
    # Block 1: plot the single model results (no comparison with FNO)
    ################################################################
    # ntrain = 7000 if args.dataset == 'sw2d_pda' else 5200
    # comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
    # log_path = './logs/' + comment 
    # np_data = np.load(f'{log_path}/test_data.npz')
    # compare_multiscale_features(np_data, channel_id=0, log_path=log_path)
    # draw_rollout_error(np_data)

    ################################################################
    # Block 2 (Plots in the paper): plot the comparison between FNO and the model
    ################################################################
    
    ##  get ntrain from DATASET_DICT
    # ntrain = DATASET_DICT[args.dataset]['train_size']
    # sample_ids = [4] if args.dataset == 'ns2d_pda' else [4]
    # model_name1 = 'FNO'
    # model_name2 = 'wavelet_transformer'
    # log_path1 = 'logs/{}_{}_ntrain{}'.format(model_name1, args.dataset, ntrain)    
    # log_path2 = 'logs/{}_{}_ntrain{}'.format(model_name2, args.dataset, ntrain)
    # np_data1 = np.load(f'{log_path1}/test_data.npz')
    # np_data2 = np.load(f'{log_path2}/test_data.npz')
    # model_name2 = 'Our'
    # # # Function 1: plot rollout predictions, errors, and energy spectrum
    # draw_rollout_error_comparison(np_data1, np_data2, channel_id=0, log_path1=log_path1, log_path2=log_path2, model_name1=model_name1, model_name2=model_name2, sample_ids=sample_ids, dataset=args.dataset)

    # # # # Function 2: plot the energy spectrum of the ground truth and the prediction
    # compare_multiscale_features_comparison(np_data1, np_data2, channel_id=0, log_path1=log_path1, log_path2=log_path2, sample_ids=sample_ids, dataset=args.dataset)

    """
    

    ################################################################
    # Block 1: Plot the prediction and ground truth and abs error
    ntrain = 7000 if args.dataset == 'sw2d_pda' else 5200
    comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
    log_path = args.log_path + comment if len(args.log_path) > 0 else './logs/' + comment
    # FNO/Wavelet/HFS test data
    # pred_data = torch.load(f'{log_path}/test_data_prediction.pth', map_location=device)
    # plot_prediction_gt_abserror(pred_data, sample_id=0, channel_id=0, model_name=args.model, log_path=log_path)
    
    
    # FNO-Diffusion test data
    log_path = log_path + '/diffusion' 
    pred_data = torch.load(f'{log_path}/best_model_diffusion.pt', map_location=device)
    plot_prediction_gt_abserror(pred_data, sample_id=0, channel_id=0, model_name='FNO-Diffusion', log_path=log_path)


    
    
    
    
    
