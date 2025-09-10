"""
collected the idea from the paper: ANTI-OVERSMOOTHING IN DEEP VISION TRANSFORMERS VIA THE FOURIER DOMAIN ANALYSIS: FROM THEORY TO PRACTICE
testing each block of the neural operator model and see if it is a high/low/mid pass filter

Step 1: First sample a random sample from the dataset (NSE or SWE), and run FFT to get the frequency spectrum,
Step 2: Then, generate synthetic data with certain frequency components/ (or maybe use the original sample, need to test it out),
 similar to X = sum_i A_i*cos(w_i*t) where A_i is the amplitude and w_i is the frequency from step 1,
 and i is from the filter band
Step 3: Iteratively pass (many steps, say 20) the neural operator model to the synthetic data to get the prediction
Step 4: Run FFT to get the frequency spectrum of the prediction
Step 5: Compute the ratio the filter prequency vs pass frequency
Step 6: See if the prediction is a high/low/mid pass filter
"""


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

from utils.griddataset import MixedTemporalDataset, TemporalDataset2D, TemporalDataset2D_multiscale, LocalTemporalDataset2D
from utils.make_master_file import DATASET_DICT
from models.fno import FNO2d
from models.wavelet_transform import CrossWaveletTransformer
from models.high_frequency_scaling import ResUNet
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils.criterion import get_frequency_bands_from_cumulative_energy
from typing import Tuple
from scipy import ndimage
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='wavelet_transformer') # FNO, ViT, UNO, CNO, Oformer, Transolver, DPOT, Crossformer, wavelet_transformer
parser.add_argument('--dataset',type=str, default='ns2d_pda') # ['ns2d_fno_1e-3', 'ns2d_pda', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 'sw2d_pda'], note: pdb is the pde bench
parser.add_argument('--resume_path',type=int, default=1) # use random weights if not cuda available
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


def load_data_model(just_load_path=False):
    ################################################################
    # load some toy data to run locally
    if not torch.cuda.is_available():
        train_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, n_channels=3, normalize=args.normalize, train='train')
        test_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels=3, normalize=args.normalize, train='test')
    else:
        # load data and dataloader
        train_dataset = TemporalDataset2D(args.dataset, t_in = args.T_in, t_ar = args.T_ar, train='train', normalize=args.normalize)
        test_dataset = TemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels = train_dataset.n_channels, train='test', normalize=args.normalize)


    
    ntrain, ntest = len(train_dataset), len(test_dataset)
    ntrain = 5200 if args.dataset == 'ns2d_pda' else ntrain # for testing (with my local folder)
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
    # model_path = log_path + '/model.pth' # for testing, note: to be deleted later
    model_path = log_path + f'/model_epochs_{args.epochs}.pth'
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
    elif args.model == 'wavelet_transformer':
        model = CrossWaveletTransformer(wave='haar', n_channels=train_dataset.n_channels, in_timesteps = args.T_in, dim=512, depth=8).to(device)
    elif args.model == 'HFS':
        model =  ResUNet(in_c = train_dataset.n_channels * args.T_in + 2 ,out_c = train_dataset.n_channels, 
                 bottleneck_feature=512, 
                 device=device).to(device)
    else:
        raise NotImplementedError


    print(model)
    count_parameters(model)

    start_epoch = 0
    best_loss_epoch = 0
    if args.resume_path:
        print('Loading models and resume from {}'.format(model_path))
        args.resume_path = model_path
        if os.path.exists(args.resume_path):
            model, optimizer, scheduler, start_epoch = resume_training_from_checkpoint(model, args.resume_path, device, optimizer=None, scheduler=None)
            print("resume training from epoch:", start_epoch)
            best_loss_epoch = start_epoch
        else:
            print("resume path does not exist, using random weights")
            start_epoch = 0
            best_loss_epoch = 0

    return model, test_loader, log_path


def compute_energy_spectrum(z, f_low=None, f_mid=None, f_high=None):
    """ 
    input: z: (1, H, W, D), f_low a list of frequency bands, f_mid a list of frequency bands, f_high a list of frequency bands
    if f_low/mid/high is None, use the quantile to find the frequency bands,and return the normalized energy spetrum and the frequency bands
    else, compute the normalized energy of the frequency using the frequency bands
    """
    if len(z.shape) == 3: # a list of (B, N, D) transformer block from wavelet transformer
        E_freq_list = []
        img_size_list = [16**2, 8**2, 4**2, 2**2]
        total_img_size = sum(img_size_list)
        assert z.shape[1] == total_img_size, "the number of pixels in the input should be the sum of the pixels in the image at each scale"
        
        # split z into a list of (B, sqrt(img_size_list[i]), sqrt(img_size_list[i]), D)
        z_list = torch.split(z, img_size_list, dim=1)
        for z_i in z_list:
            H, W = int(np.sqrt(z_i.shape[1])), int(np.sqrt(z_i.shape[1]))
            # reshape z_i to (B, sqrt(img_size_list[i]), sqrt(img_size_list[i]), D)
            z_i = z_i.reshape(z_i.shape[0], H, W, 1, z_i.shape[2])
            #
            # run 2D FFT and then bin the frequencies from 2D to 1D
            k_low, k_high, k_freq, E_freq = get_frequency_bands_from_cumulative_energy(z_i, low_percentile=0.7, high_percentile=0.97)
            E_freq_list.append(E_freq)
        E_freq = np.concatenate(E_freq_list, axis=0) #(7+3+1+1)
    else:
        if z.shape[1] == z.shape[2]:
            # run 2D FFT and then bin the frequencies from 2D to 1D
            k_low, k_high, k_freq, E_freq = get_frequency_bands_from_cumulative_energy(z, low_percentile=0.7, high_percentile=0.97)
        else:
            k_low, k_high, k_freq, E_freq = get_frequency_bands_from_cumulative_energy_1D(z,low_percentile=0.7, high_percentile=0.97)

        # aggregate the energy with the frequency bands (normalized by the band width), k_freq is np.ndarry need to cast to int
        if f_low is None or f_mid is None or f_high is None:
            f_low = (k_freq[:k_low] -1).astype(int) # -1 because the frequency is 0-indexed
            f_mid = (k_freq[k_low:k_high] -1).astype(int)
            f_high = (k_freq[k_high:] - 1).astype(int)
    
    # E_low_raw = E_freq[f_low].mean()
    # E_mid_raw = E_freq[f_mid].mean()
    # E_high_raw = E_freq[f_high].mean()
    
    return  E_freq, (f_low, f_mid, f_high)


def f_lowpass(z, cutoff_freq=10):
    """Apply a proper low-pass filter using FFT"""
    N = z.shape[1]
    signal = z.squeeze().numpy()
    
    # FFT
    signal_fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/N)  # Frequency bins
    
    # Create low-pass mask
    mask = np.abs(freqs) <= cutoff_freq
    
    # Apply filter in frequency domain
    filtered_fft = signal_fft * mask
    
    # IFFT back to time domain
    filtered = np.real(np.fft.ifft(filtered_fft))
    
    return torch.from_numpy(filtered).reshape(1, N, 1, 1, 1)


def generate_synthetic_input(dim=1, N=128):
    """Generate synthetic input with specific frequency components"""
    # Define frequency components: low, mid, high
    w_components = np.array([3, 10, 40])  # frequencies in cycles per signal length
    a_components = np.array([1, 2, 1])
    # Create time axis normalized to [0, 1] to match the filter function
    t = torch.linspace(0, 1, N)
    x = torch.zeros_like(t)
    
    # Add frequency components
    for i, w in enumerate(w_components):
        x += a_components[i] * torch.cos(2 * np.pi * w * t)
    
    # Plot the synthetic input
    plt.figure(figsize=(10, 4))
    plt.plot(t.numpy(), x.numpy(), label='Synthetic input')
    plt.xlabel('Normalized time')
    plt.ylabel('Amplitude')
    plt.title(f'Synthetic signal with frequencies: {w_components}')
    plt.legend()
    plt.show()
    
    # Define frequency bands based on the FFT bin indices
    # For N=128, frequency bins are [0, 1, 2, ..., N/2]
    # Each bin k corresponds to frequency k cycles per signal length
    f_low = np.arange(0, 8)      # 0-7 cycles per signal (low freq)
    f_mid = np.arange(8, 25)     # 8-24 cycles per signal (mid freq) 
    f_high = np.arange(25, N//2) # 25+ cycles per signal (high freq)
    
    # Reshape to expected 5D format: (batch, H, W, T, C)
    x = x.reshape(1, N, 1, 1, 1)
    return x, (f_low, f_mid, f_high)


def get_frequency_bands_from_cumulative_energy_1D(
    y: torch.Tensor,
    low_percentile: float = 0.67,
    high_percentile: float = 0.99,
    max_freq: int = None,
    eps: float = 1e-12,
    ) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Analyze 1D signal frequency content using FFT.
    
    Args:
        y: Tensor of shape (1, N, 1, 1, 1) - 1D signal data.
        low_percentile: Cumulative energy fraction for low/mid boundary.
        high_percentile: Cumulative energy fraction for mid/high boundary.
        max_freq: Maximum frequency to consider (default = N//2).
        eps: Small number to avoid divide-by-zero.

    Returns:
        k_low: int, frequency bin for low/mid boundary.
        k_high: int, frequency bin for mid/high boundary.
        freq_bins: array of frequency bin indices [0, 1, 2, ..., max_freq].
        E_freq: array of energy at each frequency bin.
    """
    assert y.ndim == 5, "y must have shape (1,N,1,1,1)"
    _, N, _, _, _ = y.shape
    
    if max_freq is None:
        max_freq = N // 2

    # Extract the 1D signal and compute FFT
    signal_1d = y.squeeze()  # (N,)
    y_fft = torch.fft.fft(signal_1d)
    
    # Take magnitude and keep only positive frequencies
    fourier_amplitudes = torch.abs(y_fft)[:max_freq].detach().cpu().numpy()
    
    # Energy is proportional to amplitude squared
    E_freq = fourier_amplitudes ** 2
    
    # Frequency bins correspond to [0, 1, 2, ..., max_freq-1] cycles per signal length
    k_freq = np.arange(len(E_freq))
    
    # Plot frequency spectrum
    # plt.figure(figsize=(10, 4))
    # plt.plot(k_freq, E_freq, 'o-', markersize=4, label='Energy spectrum')
    # plt.xlabel('Frequency bin (cycles per signal length)')
    # plt.ylabel('Energy')
    # plt.title('1D FFT Energy Spectrum')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.show()
    
    # For debugging, use fixed boundaries that make sense for our test frequencies
    # Our synthetic signal has frequencies [3, 10, 40], so:
    k_low = 8   # boundary between low and mid
    k_high = 25 # boundary between mid and high

    return k_low, k_high, k_freq, E_freq


def compare_filter_types():
    """Compare different filter types to see which ones actually work as filters"""
    print("=== Comparing Different Filter Types ===")
    
    # Generate synthetic input
    z, (f_low, f_mid, f_high) = generate_synthetic_input(N=128)
    print(f"Original signal has frequencies: [3, 10, 40] cycles per signal length")
    
    # Analyze original spectrum
    E_freq_original = compute_energy_spectrum(z, f_low, f_mid, f_high)
    
    # Define filters to test
    filters = {
        'Low-pass (cutoff=15)': lambda x: f_lowpass(x, cutoff_freq=15),
    }
    
    fig, axes = plt.subplots(1, 1, figsize=(8,8))
    axes = [axes]
    
    for i, (filter_name, filter_func) in enumerate(filters.items()):
        try:
            # Apply filter
            z_filtered = filter_func(z.clone())
            
            # Analyze spectrum
            E_freq_filtered = compute_energy_spectrum(z_filtered, f_low, f_mid, f_high)
            
            # Plot comparison
            freq_bins = np.arange(len(E_freq_original))
            axes[i].plot(freq_bins, np.log(E_freq_original), 'b-', alpha=0.7, label='Original', linewidth=2)
            axes[i].plot(freq_bins, np.log(E_freq_filtered), 'r-', alpha=0.7, label='Filtered', linewidth=2)
            
            # Mark test frequencies
            for freq in [3, 10, 40]:
                if freq < len(E_freq_original):
                    axes[i].axvline(x=freq, color='green', linestyle='--', alpha=0.5)
            
            axes[i].set_title(filter_name)
            axes[i].set_xlabel('Frequency Bin')
            axes[i].set_ylabel('Log Energy')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Print energy changes
            energy_changes = []
            for freq in [3, 10, 40]:
                if freq < len(E_freq_original):
                    change = np.log(E_freq_filtered[freq]) - np.log(E_freq_original[freq])
                    energy_changes.append(f"Freq {freq}: {change:.2f}")
            print(f"{filter_name}: {', '.join(energy_changes)}")
            
        except Exception as e:
            # axes[i].text(0.5, 0.5, f"Error: {str(e)}", transform=axes[i].transAxes, 
            #             ha='center', va='center')
            axes[i].set_title(f"{filter_name} (Failed)")
    
    plt.tight_layout()
    plt.show()
    print("Filter comparison completed")


def pass_filter_simluation_testing(filter_type='lowpass', filter_params=None):
    """
    Test different types of filters on synthetic data
    
    Args:
        filter_type: 'lowpass', 'highpass', 'bandpass', 'gaussian', 'butterworth', 'modulation'
        filter_params: dict with filter-specific parameters
    """
    if filter_params is None:
        filter_params = {}
    
    with torch.no_grad():
        # Generate synthetic input
        z, (f_low, f_mid, f_high) = generate_synthetic_input(dim=1, N=128)
        E_freq_raw, _ = compute_energy_spectrum(z, f_low, f_mid, f_high)
        
        # Select filter function
        if filter_type == 'lowpass':
            cutoff = filter_params.get('cutoff_freq', 15)
            filter_func = lambda x: f_lowpass(x, cutoff_freq=cutoff)
            print(f"Testing Low-pass filter (cutoff={cutoff})")
            
        # Run iterative filtering
        total_steps = 50
        E_freq_time = np.zeros((total_steps+1, len(E_freq_raw)))
        E_freq_time[0] = 0

        for t in range(total_steps):
            z = filter_func(z)
            E_freq_t, _ = compute_energy_spectrum(z, f_low, f_mid, f_high) 
            # Store relative energy change
            # E_freq_time[t+1] = np.log(E_freq_t) - np.log(E_freq_raw)
            # try the absolute value of the energy change
            E_freq_time[t+1] = np.log(E_freq_t)

    return E_freq_time
    

def pass_filter_testing(test_loader=None, model=None, start_block_index=None):
    """
    Test the filter passing on the test loader
    """
    
    # get a random sample from the test loader
    x, y = next(iter(test_loader))
    x = x.to(device)
    y = y.to(device)


    with torch.no_grad():
        model.eval()
        
        start_block_index = 0 if start_block_index is None else start_block_index
        # Obtain the latent representation from the input to the start_block_iddex
        z = model.get_latent_by_index(x, start_block_index) # (B, H, W, T, C) -> (B, C, H, W)
        
        E_freq_raw, (f_low, f_mid, f_high) = compute_energy_spectrum(z)
        
        
        filter_func = model.get_testing_block_by_index
            
        # Run iterative filtering
        total_steps = 50
        E_freq_time = np.zeros((total_steps, len(E_freq_raw)))
        E_freq_time[0] = 0

        for t in range(total_steps):
            z = filter_func(start_block_index, z)
            E_freq_t, _ = compute_energy_spectrum(z, f_low, f_mid, f_high) 
            # Store relative energy change 
            # E_freq_time[t+1] = np.log(E_freq_t) - np.log(E_freq_raw)
            E_freq_time[t] = np.log(E_freq_t)

    return E_freq_time, start_block_index


def plot_filter_passing(E_freq_time, simulation=True, start_block_index=None):
    """
    Plot the filter passing showing how energy in each frequency bin evolves over time.
    E_freq_time shape: (n_steps, n_frequencies)
    """
    n_steps, n_frequencies = E_freq_time.shape
    
    # Apply smoothing if requested

    E_freq_smoothed = E_freq_time
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a colormap with different shades of blue
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.cm as cm
    
    #  Use the blue to green to red colormap
    cmap = plt.cm.gist_rainbow
    
    # Plot each frequency bin with a different shade of blue
    time_steps = np.arange(n_steps)
    denom = max(n_frequencies - 1, 1)

    # Plot all frequencies, but only add labels for every 5th frequency for cleaner legend
    for freq_idx in range(n_frequencies):
        color = cmap(freq_idx / denom)
        label = f'Freq {freq_idx}'
        if simulation:
            if freq_idx in [3, 10, 40]:  # Show our test frequencies
                
                # Highlight our test frequencies
                if freq_idx == 3:
                    label = f'Freq {freq_idx} (Low test)'
                    axes.plot(time_steps, E_freq_smoothed[:, freq_idx], color='red', linewidth=2, label=label, zorder=2, linestyle='--')
                elif freq_idx == 10:
                    label = f'Freq {freq_idx} (Mid test)'
                    axes.plot(time_steps, E_freq_smoothed[:, freq_idx], color='green', linewidth=2, label=label, zorder=2, linestyle='--')
                elif freq_idx == 40:
                    label = f'Freq {freq_idx} (High test)'
                    axes.plot(time_steps, E_freq_smoothed[:, freq_idx], color='orange', linewidth=2, label=label, zorder=2, linestyle='--')

        # Add label only for every 5th frequency (or important frequencies)
        if freq_idx % 5 == 0 or n_frequencies < 15:
            axes.plot(time_steps, E_freq_smoothed[:, freq_idx], color=color, linewidth=1, alpha=0.6 if simulation else 1, label=label, zorder=1)
        # else:
        #     # Plot without label for cleaner legend
        #     axes.plot(time_steps, E_freq_smoothed[:, freq_idx], color=color, linewidth=1, alpha=0.6, zorder=1)
    
    # Customize the plot
    # axes.set_xlim(-1, n_steps)
    axes.set_xlabel('Rollout time k', fontsize=12)
    axes.set_ylabel('Log Energy', fontsize=12)
    axes.set_title(f'Block Index {start_block_index}', fontsize=12)
    axes.grid(True, alpha=0.3)

    
    # Create a more organized legend
    legend = axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    legend.set_title("Frequency Bins", prop={'size': 11, 'weight': 'bold'})
    
    plt.tight_layout()
    if not os.path.exists(f'{log_path}/filter_passing'):
        os.makedirs(f'{log_path}/filter_passing')
    plt.savefig(f'{log_path}/filter_passing/block_index_{start_block_index}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    print("Plot saved and displayed")



if __name__ == "__main__":
    # Run filter testing
    # E_freq_time = pass_filter_simluation_testing(filter_type='lowpass', filter_params={'cutoff_freq': 8})
    # plot_filter_passing(E_freq_time)

   model, test_loader, log_path = load_data_model()
   for start_block_index in range(args.n_layers):
       print(f"Testing block index {start_block_index}")
       E_freq_time, start_block_index = pass_filter_testing(test_loader=test_loader, model=model, start_block_index=start_block_index)
       plot_filter_passing(E_freq_time, simulation=False, start_block_index=start_block_index)
    