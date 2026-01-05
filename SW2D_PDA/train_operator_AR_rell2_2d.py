import os
import yaml
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from data_utils.datasets import SWLoader2D
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
from utils.utilities import log_tensorboard_images_and_spectra, count_parameters, save_checkpoint



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

    sample = data[sample_idx]
    x = sample[..., t_idx, :].to(device)
    y = sample[..., t_idx + 1, :].to(device)
    grid_b = grid.to(device)
    x_in = torch.cat((x.unsqueeze(0), grid_b), dim=-1)
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


def train_step_ahead(model, train_loader, optimizer, scheduler, config, device, grid, test_loader=None, eval_step=100,save_step=1000, use_tqdm=True, writer=None, model_name='fno2d', start_ep=0):
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
                fixed_pred, fixed_target = get_fixed_test_pair(model, test_loader, grid, device, sample_idx=0, t_idx=0)
                # print("fixed_pred shape:", fixed_pred.shape, "fixed_target shape:", fixed_target.shape)
                
                save_checkpoint(config['train']['save_dir'],
                                config['train']['save_name'],
                                model, 
                                ep,
                                optimizer, scheduler)
                if fixed_pred is not None:
                    log_tensorboard_images_and_spectra(writer,
                                                       fixed_pred.unsqueeze(-2)[...,[0]],  # the first channel is vorticity
                                                       fixed_target.unsqueeze(-2)[..., [0]],  # the first channel is vorticity
                                                       ep + 1,
                                                       'vorticity',
                                                       model_name,
                                                       ) 


def build_synthetic_dataset(data_config, n_samples, step_ahead=False):
    """Create a random dataset that mimics NSLoader/NSLoader2D output."""
    sub = data_config.get('sub', 1)
    sub_t = data_config.get('sub_t', 1)
    nx = data_config.get('nx', 96)
    ny = data_config.get('ny', 192)
    nt = data_config.get('nt', 87)
    nc = data_config.get('nc', 2)
    time_scale = data_config.get('time_interval', 1.0)
    S1 = nx // sub
    S2 = ny // sub
    T = ny // sub
    T = int(nt * time_scale) // sub_t + 1

    if step_ahead:
        data = torch.rand(n_samples, S1, S2, T, nc)

        class SyntheticStepDataset(Dataset):
            def __init__(self, arr):
                self.arr = arr
                self.max_t = arr.shape[-2] - 1

            def __len__(self):
                return self.arr.shape[0]

            def __getitem__(self, idx):
                sample = self.arr[idx]
                t = torch.randint(0, self.max_t, ()).item()
                return sample[..., t, :], sample[..., t + 1, :]

        return SyntheticStepDataset(data), (S1, S2), 1

    a0 = torch.rand(n_samples, S1, S2, 1, 1)
    a_data = a0.repeat(1, 1, 1, T, 1)
    gridx, gridy, gridt = get_grid3d(S1, S2, T, time_scale=time_scale)
    a_data = torch.cat((
        gridx.repeat([n_samples, 1, 1, 1, 1]),
        gridy.repeat([n_samples, 1, 1, 1, 1]),
        gridt.repeat([n_samples, 1, 1, 1, 1]),
        a_data
    ), dim=-1)
    u_data = torch.rand(n_samples, S1, S2, T)
    return TensorDataset(a_data, u_data), S, T


def train_2d(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    # prepare dataloader for training with data (real or synthetic)
    if args.synthetic_samples > 0:
        full_dataset, S_data, _ = build_synthetic_dataset(
            data_config, args.synthetic_samples, step_ahead=True)
    else:
        full_dataset = SWLoader2D(datapath1=data_config['datapath'],
                                    nx=data_config['nx'], 
                                    ny=data_config['ny'],
                                    nt=data_config['nt'],
                                    sub=data_config['sub'], sub_t=data_config['sub_t'],
                                    N=data_config['total_num'],
                                    t_interval=data_config['time_interval'],
                                    n_samples=data_config.get('n_sample', data_config.get('n_samples', data_config['total_num'])),
                                    offset=data_config.get('offset', 0),
                                    normalizer_path=data_config.get('normalizer_path', None))
        S_data = full_dataset.S
    
    
    # split dataset into training and validation sets by test_ratio
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
    else:
        train_set = full_dataset
        test_loader = None

    train_loader = DataLoader(train_set,
                              batch_size=config['train']['batchsize'],
                              shuffle=data_config['shuffle'])
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
        dummy = torch.zeros(1, 1, S_data, S_data, device=device)
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
                        H = S_data,
                        W = S_data,
                        slice_num=model_cfg.get('slice_num', 32),
                        ref=model_cfg.get('ref', 8),
                        unified_pos=model_cfg.get('unified_pos', 0),
                        is_filter=model_cfg.get('is_filter', True)).to(device)
    elif model_name in ['multiscale_wavelet', 'multiscale_wavelet2d', 'multiscale_wavelet_transformer2d']:
        model = MultiscaleWaveletTransformer2D(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            patch_size= model_cfg.get('patch_size', None),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
        ).to(device)
    elif model_name in ['multiscale_wavelet2d_nodecoderattn']:
        model = MultiscaleWaveletTransformer2DDecoderNoAttention(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            patch_size= model_cfg.get('patch_size', None),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
        ).to(device)
    elif model_name in ['multiscale_wavelet2d_attn05124_group4']:
        model = MultiscaleWaveletTransformer2DEfficient(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            patch_size= model_cfg.get('patch_size', None),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
        ).to(device)
    elif model_name in ['multiscale_wavelet2d_double_attn']:
        model = MultiscaleWaveletDoubleAttention(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
        ).to(device)
    elif model_name in ['multiscale_wavelet2d_denoattn_stacklayers3']:
        model = MSWT_DeNoAttn_StackLayers(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dims=model_cfg.get('dims', []),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
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
        ckpt_path = os.path.join(config['train']['save_dir'], args.resume_ckpt)
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

    grid = torch2dgrid(S_data[0], S_data[1])
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
