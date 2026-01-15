import os
import yaml
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from data_utils.datasets import NS_Dedalus_Loader2D
from einops import rearrange
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.fno import FNO2d
from models.high_frequency_scaling import ResUNet
from models.wno import WNO2d
from models.saot import SAOTModel
from models.wavelet_transform import MultiscaleWaveletTransformer2D
from models.wavelet_transform_exploration import MultiscaleWaveletTransformer2DDecoderNoAttention, MultiscaleWaveletTransformer2DEfficient, MultiscaleWaveletDoubleAttention, MSWT_DeNoAttn_StackLayers
from models.pderefiner import PDERefiner
from models.pderefiner_unet import UNetRefiner
from tqdm import tqdm
from utils.criterion import LpLoss
from utils.utilities import _to_rgb_minmax, fig_to_tensorboard_image, count_parameters, save_checkpoint
from utils.compute_diagnostics import velocity_from_vorticity, compute_spectra
import numpy as np


def evaluate_3d(model, test_loader, device):
    """Run a quick L2 evaluation on a held-out set."""
    lploss = LpLoss(size_average=True)
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size, S, _, T, _ = x.shape
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S, S, T + 5)
            out = out[..., :-5]
            total += lploss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T)).item()
            batches += 1
    if batches == 0:
        return None
    return total / batches


def evaluate_step_ahead(model, test_loader, device, grid):
    """Evaluate one-step prediction u_t -> u_{t+1}."""
    lploss = LpLoss(size_average=True)

    model.eval()
    total = 0.0
    batches = 0
    pred_plot = None
    target_plot = None
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch = x.shape[0]
            x_in = torch.cat((x, grid.expand(batch, -1, -1, -1)), dim=-1)
            if isinstance(model, PDERefiner):
                if len(x.shape) == 3:
                    x = rearrange(x, 'b h w -> b 1 1 h w')
                pred = model.validation_step(x)
                pred = rearrange(pred, 'b 1 c h w -> b h w c')
            else:
                pred = model(x_in)
            total += lploss(pred, y).item()
            if pred_plot is None:
                pred_plot = pred.clone()
                target_plot = y.clone()
            batches += 1
    if batches == 0:
        return None
    return total / batches, pred_plot, target_plot


def _get_base_dataset(ds):
    """Return the underlying dataset (unwrap Subset/DataLoader)."""
    if isinstance(ds, DataLoader):
        ds = ds.dataset
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def get_fixed_test_pair(model, test_source, grid, device, sample_idx=0, t_idx=0):
    """
    Grab a deterministic (x_t, x_{t+1}) pair from the test data without relying on
    the test loader's random timestep selection.
    """
    base_ds = _get_base_dataset(test_source)
    if not hasattr(base_ds, 'data'):
        return None, None
    data = base_ds.data
    if sample_idx >= data.shape[0]:
        sample_idx = data.shape[0] - 1
    max_t = data.shape[-2] - 1
    if max_t <= 0:
        return None, None
    t_idx = min(t_idx, max_t - 1)

    x = data[sample_idx].to(device) # (X, Y, C)
    y = data[sample_idx + 1, ..., 0].to(device) # (X, Y, C)
    grid_b = grid.to(device)
    return x, y

def torch2dgrid(num_x, num_y, bot=(0,0), top=(1,1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y)
    xx, yy = torch.meshgrid(x_arr, y_arr, indexing='ij')
    mesh = torch.stack([xx, yy], dim=2)
    return mesh


def train_step_ahead(model, train_loader, optimizer, scheduler, config, device, grid, test_loader=None, eval_step=10,save_step=100, use_tqdm=True, writer=None, model_name='fno2d', start_ep=0):
    """Train on one-step pairs (u_t, u_{t+1})."""
    lploss = LpLoss(size_average=True)
    epochs = config['train']['epochs']
    grid = grid.to(device).unsqueeze(0)

    lambda_amp_final = 1e-2    # good starting point
    warmup_frac = 0.2          # first 20% epochs
    if start_ep >= epochs:
        print(f'start_ep ({start_ep}) >= epochs ({epochs}); skipping training loop.')
        return
    if use_tqdm:
        pbar = tqdm(range(start_ep, epochs), dynamic_ncols=True, smoothing=0.1)
    else:
        pbar = range(start_ep, epochs)
    best_loss = torch.inf
    for ep in pbar:
        model.train()
        running = 0.0
        batches = 0

        # linear warm-up
        t = ep / max(1, epochs - 1)
        if t < warmup_frac:
            lambda_amp = 0.0
        else:
            ramp = (t - warmup_frac) / (1.0 - warmup_frac)
            lambda_amp = lambda_amp_final * ramp
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch = x.shape[0]
            x_in = torch.cat((x, grid.expand(batch, -1, -1, -1)), dim=-1)
            if isinstance(model, PDERefiner): # for PDERefiner, the loss function is a denoising loss
                loss = model.training_step((x, y))
            else:
                pred = model(x_in)
                if isinstance(pred, tuple):
                    pred, x_reg = pred
                    # print("pred shape:", pred.shape, "x_reg shape:", x_reg.shape)
                else:
                    x_reg = None
                data_loss = lploss(pred, y)
                if x_reg is not None:
                    loss = data_loss + lambda_amp * x_reg
                else:
                    loss = data_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            batches += 1
        scheduler.step()
        avg = running / max(1, batches)
        print(f'Epoch {ep + 1}/{epochs}, train L2: {avg:.6f}')
        if writer is not None:
            writer.add_scalar('train/l2', avg, ep + 1)
        if use_tqdm:
            pbar.set_description((f'Train L2: {avg:.6f}'))

        if ep % eval_step == 0 and test_loader is not None:
            test_l2, _, _ = evaluate_step_ahead(model, test_loader, device, grid)
            print(f'Random test split relative L2: {test_l2:.6f}')
            if writer is not None:
                writer.add_scalar('eval/test_l2', test_l2, ep + 1)
                fixed_pred, fixed_target = get_fixed_test_pair(model, test_loader, grid, device, sample_idx=3000, t_idx=0)
                print("fixed_pred shape:", fixed_pred.shape, "fixed_target shape:", fixed_target.shape)
                
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'],
                                model, 
                                ep,
                                optimizer, scheduler)
                if fixed_pred is not None:
                    log_tensorboard_images_and_spectra(writer,
                                                       fixed_pred[..., None, None],
                                                       fixed_target[..., None, None],
                                                       ep + 1,
                                                       'vorticity',
                                                       model_name,
                                                       ) 

def build_synthetic_dataset(data_config, n_samples, step_ahead=False):
    """Create a random dataset that mimics NSLoader/NSLoader2D output."""
    sub = data_config.get('sub', 1)
    sub_t = data_config.get('sub_t', 1)
    nx = data_config.get('nx', 64)
    nt = data_config.get('nt', 64)
    nc = data_config.get('nc', 5)
    time_scale = data_config.get('time_interval', 1.0)
    S = nx // sub
    T = int(nt * time_scale) // sub_t + 1

    if step_ahead:
        data = torch.rand(n_samples, S, S, T, nc)

        class SyntheticStepDataset(Dataset):
            def __init__(self, arr):
                self.arr = arr
                self.max_t = arr.shape[-2] - 1

            def __len__(self):
                return self.arr.shape[0]

            def __getitem__(self, idx):
                sample = self.arr[idx]
                t = torch.randint(0, self.max_t, ()).item()
                return sample[..., t, :], sample[..., t + 1, :1]

        return SyntheticStepDataset(data), S, 1

    a0 = torch.rand(n_samples, S, S, 1, 1)
    a_data = a0.repeat(1, 1, 1, T, 1)
    gridx, gridy, gridt = get_grid3d(S, T, time_scale=time_scale)
    a_data = torch.cat((
        gridx.repeat([n_samples, 1, 1, 1, 1]),
        gridy.repeat([n_samples, 1, 1, 1, 1]),
        gridt.repeat([n_samples, 1, 1, 1, 1]),
        a_data
    ), dim=-1)
    u_data = torch.rand(n_samples, S, S, T)
    return TensorDataset(a_data, u_data), S, T



def genertate_images_and_spectra(
    pred_denorm: torch.Tensor,
    Lx: float = 2 * np.pi,
    Ly: float = 2 * np.pi,
    model_name: str = 'fno2d'
):
    """Log prediction, target, error images and energy/enstrophy spectra to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        pred_denorm: Denormalized predictions, shape (B, H, W, T, C)
        target_denorm: Denormalized targets, shape (B, H, W, T, C)
        epoch: Current epoch number
        form: Data form ('vorticity' or 'velocity')
        model_name: Name of the model (for plot labels)
        Lx: Domain size in x direction (default: 2*pi)
        Ly: Domain size in y direction (default: 2*pi)
    """
    import matplotlib.pyplot as plt
    B, H, W, T, C = pred_denorm.shape
    # Create enstrophy spectrum plot
    

    plt.figure(figsize=(10, 6))
    plt.imshow(pred_denorm[0, :, :, 0, 0], cmap='viridis')
    plt.savefig(f'{model_name}_ground_truth.png')

    ux_pred, uy_pred = velocity_from_vorticity(torch.from_numpy(pred_denorm))

    # Compute spectra for prediction and target
    k_bins, Ek_pred, Zk_pred = compute_spectra(ux_pred, uy_pred, Lx, Ly)

        
    # Create energy spectrum plot
    fig_energy, ax_energy = plt.subplots(figsize=(10, 6))
    k_nyquist = int((np.pi * H) // Lx)

    start_truth = 1

    ax_energy.loglog(k_bins[start_truth:k_nyquist], Ek_pred[start_truth:k_nyquist], 
                    'o-', markersize=1, label=f'{model_name} Prediction', linewidth=1, color='blue')
    ax_energy.set_xlabel('Wavenumber', fontsize=14)
    ax_energy.set_ylabel('Energy', fontsize=14)
    ax_energy.set_title('Energy Spectrum', fontsize=14)
    ax_energy.legend(fontsize=12)
    ax_energy.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_name}_energy_spectrum.png')
    
    

   

def train_2d(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    # prepare dataloader for training with data (real or synthetic)
    if args.synthetic_samples > 0:
        full_dataset, S_data, _ = build_synthetic_dataset(
            data_config, args.synthetic_samples, step_ahead=True)
    else:
        full_dataset = NS_Dedalus_Loader2D(datapath1=data_config['datapath'],
                                    nx=data_config['nx'], nt=data_config['nt'],
                                    sub=data_config['sub'], sub_t=data_config['sub_t'],
                                    N=data_config['total_num'],
                                    t_interval=data_config['time_interval'],
                                    n_samples=data_config.get('n_sample', data_config.get('n_samples', data_config['total_num'])),
                                    offset=data_config.get('offset', 0))
        S_data = full_dataset.S
    
    
    #
    x, _ = full_dataset[3000]
    print("x shape:", x.shape)
    genertate_images_and_spectra(x[...,-1][None, ..., None, None].cpu().numpy())




if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--test_ratio', type=float, default=0.0,
                        help='Hold out this fraction of samples for a random test split')
    parser.add_argument('--test_seed', type=int, default=42,
                        help='Seed for the random test split')
    parser.add_argument('--synthetic_samples', type=int, default=0,
                        help='Use random synthetic data with this many samples to sanity-check the 3D pipeline')
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the last checkpoint')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Specific checkpoint filename to resume from (in save_dir)')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    train_2d(args, config)
