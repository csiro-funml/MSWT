import os
import yaml
from argparse import ArgumentParser
import torch
import numpy as np
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
from models.mswt_res import PeriodicMSWT2D_Patching_Residual
from models.pderefiner import PDERefiner
from models.pderefiner_unet import UNetRefiner
from tqdm import tqdm
from utils.criterion import LpLoss
from utils.utilities import log_tensorboard_images_and_spectra, count_parameters, save_checkpoint
from utils.compute_diagnostics import velocity_from_vorticity, compute_spectra
import matplotlib.pyplot as plt




def verify_dataset(dataset, total_samples=5, interval=200, 
                   Lx=2 * np.pi,
                   Ly=2 * np.pi,
                   state='train'):
    """plot the resolution and the spectrum of some samples in the dataset."""
    fig, axes = plt.subplots(2, total_samples, figsize=(10, 20))
    for img_idx, sample_idx in enumerate(range(0, total_samples*interval, interval)):
        x, y = dataset[sample_idx]
        print("x shape: ", x.shape, "y shape: ", y.shape)
        ax = axes[0, img_idx]
        im = ax.imshow(y[..., -1].cpu().numpy(), cmap='RdBu_r', origin='lower')
        plt.colorbar(im)
        ax.set_title(f'Sample {sample_idx}')
        
        ux_pred, uy_pred = velocity_from_vorticity(y[..., -1].cpu())

        # Compute spectra for prediction and target
        k_bins, Ek_pred, Zk_pred = compute_spectra(ux_pred, uy_pred, Lx, Ly)

        k_nyquist = int((np.pi * x.shape[1]) // Lx)

        start_truth = 1
        ax_energy = axes[1, img_idx]
        ax_energy.loglog(k_bins[start_truth:k_nyquist], Ek_pred[start_truth:k_nyquist], 
                        'o-', markersize=1, label=f'Ground Truth', linewidth=1, color='blue')
        ax_energy.set_xlabel('Wavenumber', fontsize=14)
        ax_energy.set_ylabel('Energy', fontsize=14)
        ax_energy.set_title('Energy Spectrum', fontsize=14)
        ax_energy.legend(fontsize=12)
        ax_energy.grid(True)

    plt.savefig(f'dedalus_data_{state}.png')
    return True


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
    
    This function properly handles Subset wrappers from random_split by using
    the dataset's __getitem__ method, which respects the offset applied during
    dataset initialization.
    """
    # Unwrap DataLoader if needed, but preserve Subset to use its index mapping
    if isinstance(test_source, DataLoader):
        test_source = test_source.dataset
    
    # If it's a Subset, use it directly (Subset.__getitem__ handles index mapping)
    # Otherwise, use the base dataset
    if isinstance(test_source, Subset):
        dataset = test_source
        # sample_idx is relative to the Subset's indices
        if sample_idx >= len(dataset):
            sample_idx = len(dataset) - 1
        print(f"Accessing Subset: sample_idx={sample_idx}, subset_length={len(dataset)}")
    else:
        dataset = _get_base_dataset(test_source)
        if not hasattr(dataset, 'data'):
            return None, None
        # sample_idx is relative to the base dataset (which has already been offset-sliced)
        if sample_idx >= len(dataset):
            sample_idx = len(dataset) - 1
        # Log offset info if available for debugging
        if hasattr(dataset, 'original_offset'):
            print(f"Accessing base dataset: sample_idx={sample_idx}, dataset_length={len(dataset)}, "
                  f"original_offset={dataset.original_offset}")
        else:
            print(f"Accessing base dataset: sample_idx={sample_idx}, dataset_length={len(dataset)}")
    
    # Use __getitem__ to properly access the dataset, which handles the offset correctly
    # For NS_Dedalus_Loader2D, data shape is (T, X, Y, C) and __getitem__(idx) returns
    # (data[idx], data[idx+1, :, :, :1]) where idx is a timestep index after offset
    x, y = dataset[sample_idx]
    x = x.to(device)  # (X, Y, C)
    y = y.to(device)  # (X, Y, 1) or (X, Y) - vorticity only
    # Squeeze last dimension if it exists (y might be (X, Y, 1) from __getitem__)
    if y.dim() == 3 and y.shape[-1] == 1:
        y = y.squeeze(-1)  # (X, Y)
    grid_b = grid.to(device)
    # print("x shape:", x.shape, "y shape:", y.shape, "grid_b shape:", grid_b.shape)
    x_in = torch.cat((x.unsqueeze(0), grid_b), dim=-1) # (1, X, Y, 2)
    with torch.no_grad():
        if isinstance(model, PDERefiner):
            if len(x.shape) == 2:
                x = rearrange(x, 'h w -> 1 1 1 h w')
            pred = model.validation_step(x)
            pred = rearrange(pred, 'b 1 c h w -> b h w c')
        else:
            pred = model(x_in)
        if pred.dim() == 5:
            pred = pred.squeeze(-2)
        if pred.dim() == 4:
            pred = pred.squeeze(-1)
    return pred, y.unsqueeze(0)

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


def train_2d(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    # prepare dataloader for training with data (real or synthetic)
    if args.synthetic_samples > 0:
        full_dataset, S_data, _ = build_synthetic_dataset(
            data_config, args.synthetic_samples, step_ahead=True)
    else:
        print("offset: ", data_config.get('offset', 0))
        full_dataset = NS_Dedalus_Loader2D(datapath1=data_config['datapath'],
                                    nx=data_config['nx'], nt=data_config['nt'],
                                    sub=data_config['sub'], sub_t=data_config['sub_t'],
                                    N=data_config['total_num'],
                                    t_interval=data_config['time_interval'],
                                    n_samples=data_config.get('n_sample', data_config.get('n_samples', data_config['total_num'])),
                                    offset=data_config.get('offset', 0))
        S_data = full_dataset.S
        print("full_dataset shape: ", full_dataset.data.shape)
    
    
    # split dataset into training and validation sets by test_ratio
    # Note: random_split works correctly with offset because:
    # 1. The dataset's data has already been offset-sliced during __init__
    # 2. random_split creates Subset objects that map indices correctly
    # 3. Subset.__getitem__ calls the base dataset's __getitem__ with mapped indices
    if args.test_ratio > 0:
        test_size = max(1, int(len(full_dataset) * args.test_ratio))
        if len(full_dataset) - test_size <= 0:
            raise ValueError('test_ratio is too large; no samples left for training.')
        train_size = len(full_dataset) - test_size
        train_set, test_set = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(args.test_seed)
        )
        test_set.train = False # set test set to not train
        test_loader = DataLoader(test_set,
                                 batch_size=config['train']['batchsize'],
                                 shuffle=False)
        
        print("train set length: ", len(train_set), "test set length: ", len(test_set))
    else:
        train_set = full_dataset
        test_loader = None

    train_loader = DataLoader(train_set,
                              batch_size=config['train']['batchsize'],
                              shuffle=data_config['shuffle'])
    
    # todo: verify the the first 
    verify_dataset(full_dataset, total_samples=5, interval=200, state='full')
    verify_dataset(train_set, total_samples=5, interval=200, state='train')
    verify_dataset(test_set, total_samples=5, interval=20, state='test')
    exit(-1)
    # create model
    print("device: ", device)
    model_cfg = config['model']
    model_name = model_cfg.get('name', 'fno2d').lower()
    
    if model_name == 'fno2d':
        model = FNO2d(modes1=model_cfg['modes1'],
                      modes2=model_cfg['modes2'],
                      fc_dim=model_cfg['fc_dim'],
                      layers=model_cfg['layers'],
                      act=model_cfg['act'],
                      in_dim=model_cfg['in_dim'],
                      out_dim=model_cfg['out_dim'],
                    #   pad_ratio=model_cfg.get('pad_ratio', [0., 0.])
                      ).to(device)
    elif model_name == 'hfs':
        model = ResUNet(in_c=model_cfg.get('in_c', 3),
                        out_c=model_cfg.get('out_c', 1),
                        target_params=model_cfg.get('target_params', 'medium'),
                        device=device).to(device)
    elif model_name == 'pderefiner':
        model = PDERefiner(
                name=model_cfg.get('basemodel_name', 'Unetmod-64'),
                time_history=model_cfg.get('time_history', 1), # T_in
                time_future=model_cfg.get('time_future', 1), # T_ar
                time_gap=0,
                max_num_steps=model_cfg.get('max_num_steps', 1),  # T_ar, just one step ahead
                n_spatial_dim=model_cfg.get('n_spatial_dim', 2),
                in_channels=model_cfg.get('in_channels', 3), # input channels
                out_channels=model_cfg.get('out_channels', 1)   , # output channels
                trajlen=model_cfg.get('trajlen', 64), # T_max
                activation=model_cfg.get('activation', 'gelu'),
                criterion=model_cfg.get('criterion', 'mse'),
                hidden_channels=model_cfg.get('hidden_channels', 16),
                n_blocks=model_cfg.get('n_blocks', 3),
    ).to(device)
    elif model_name in ['refiner_unet']:
        model = UNetRefiner(
            input_channels=model_cfg.get('in_channels', 3),
            output_channels=model_cfg.get('out_channels', 1),
            time_history=model_cfg.get('time_history', 0),
            time_future=model_cfg.get('time_future', 0),
            hidden_channels=model_cfg.get('hidden_channels', 16),
            activation=model_cfg.get('activation', 'gelu'),
            n_blocks=model_cfg.get('n_blocks', 3),
        ).to(device)
    elif model_name in ['wno', 'wno2d']:
        dummy = torch.zeros(1, 1, S_data[0], S_data[1], device=device)
        model = WNO2d(in_channels=model_cfg.get('in_chans', 3),
                      out_channels=model_cfg.get('out_chans', 1),
                      width=model_cfg.get('width', 64),
                      level=model_cfg.get('level', 3),
                      dummy_data=dummy).to(device)
    elif model_name in ['saot', 'saot2d']:
        model = SAOTModel(space_dim=model_cfg.get('space_dim', 2),
                        n_layers=model_cfg.get('n_layers', 3),
                        n_hidden=model_cfg.get('n_hidden', 64)  ,
                        dropout=model_cfg.get('dropout', 0.0),
                        n_head=model_cfg.get('n_head', 4),
                        Time_Input=model_cfg.get('Time_Input', False),
                        mlp_ratio=model_cfg.get('mlp_ratio', 1),
                        fun_dim=model_cfg.get('fun_dim', 1),
                        out_dim=model_cfg.get('out_dim', 1),
                        H = S_data[0],
                        W = S_data[1],
                        slice_num=model_cfg.get('slice_num', 32),
                        ref=model_cfg.get('ref', 8),
                        unified_pos=model_cfg.get('unified_pos', 0),
                        is_filter=model_cfg.get('is_filter', True)).to(device)
    elif model_name in ['mswt_periodic_patching_residual', 'periodic_mswt_patching_residual']:
        model = PeriodicMSWT2D_Patching_Residual(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
            add_grid=model_cfg.get('add_grid', False),
            add_periodic_grid=model_cfg.get('add_periodic_grid', False),
            patch_size=model_cfg.get('patch_size', None),
            local_attention_size=model_cfg.get('local_attention_size', None),
            residual_connection=model_cfg.get('residual_connection', False),
        ).to(device)
    else:
        raise ValueError(f'Model {model_name} not supported')
    print('model structure: ', model)
    count_parameters(model)
    
    # create optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])

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
    
    
    save_dir = config['train']['save_dir'] if torch.cuda.is_available() else 'saved_models'
    tensorboard_dir = config['train'].get('tensorboard_dir')
    if tensorboard_dir is None:
        tensorboard_dir = os.path.join(save_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    grid = torch2dgrid(S_data, S_data)
    train_step_ahead(model,
                        train_loader,
                        optimizer,
                        scheduler,
                        config,
                        device,
                        grid,
                        test_loader=test_loader,
                        writer=writer,
                        model_name=model_name,
                        start_ep=start_ep)
    
    if test_loader is not None:
        test_l2, _, _ = evaluate_step_ahead(model, test_loader, device, grid.to(device).unsqueeze(0))
        print(f'Random test split relative L2: {test_l2:.6f}')
        if writer is not None:
            writer.add_scalar('eval/test_l2', test_l2, config['train']['epochs'])
    if writer is not None:
        writer.close()



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
