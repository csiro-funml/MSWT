import yaml
import torch
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_utils.datasets import SWLoader2D
from models.fno import FNO2d
from models.high_frequency_scaling import ResUNet
from models.wno import WNO2d
from models.saot import SAOTModel
from models.wavelet_transform import MultiscaleWaveletTransformer2D
from models.wavelet_transform_exploration import MultiscaleWaveletTransformer2DDecoderNoAttention, MultiscaleWaveletTransformer2DEfficient, MultiscaleWaveletDoubleAttention
from models.pderefiner import PDERefiner
from models.pderefiner_unet import UNetRefiner
from einops import rearrange
from utils.criterion import LpLoss, LogEnstropyEnergyLoss
from utils.compute_diagnostics import velocity_from_vorticity, compute_spectra_torch


def torch2dgrid(num_x, num_y, bot=(0,0), top=(1,1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y)
    xx, yy = torch.meshgrid(x_arr, y_arr, indexing='ij')
    mesh = torch.stack([xx, yy], dim=2)
    return mesh

def load_sw_sequences(data_config):
    """Load normalized (N, H, W, T, C) sequences using the same pipeline as training."""
    dataset = np.load(data_config['test_data']['datapath'])
    sequences = torch.tensor(dataset, dtype=torch.float)
    # normalize the data
    normalizer_path = data_config['data']['normalizer_path']
    normalizer = torch.load(normalizer_path)
    vars = ['vor', 'pres']
    mean = []
    std = []
    for var in vars:
        mean.append(normalizer[var]['mean'].squeeze()) # (1, H, W)
        std.append(normalizer[var]['std'].squeeze()) # (1, H, W)
    mean = torch.stack(mean, dim=0) # (C, H, W)
    std = torch.stack(std, dim=0) # (C, H, W)
    mean = mean.permute(1, 2, 0)[None, :, :, None, :] # (C, H, W) -> (1, H, W, 1, C)
    std = std.permute(1, 2, 0)[None, :, :, None, :] # (C, H, W) -> (1, H, W, 1, C)
    
    sequences = (sequences - mean) / std

    S1, S2 = data_config['test_data']['nx'], data_config['test_data']['ny']
    T = data_config['test_data']['nt']
    print("final data shape: ", sequences.shape)  # (N, H, W, T, C)
    return sequences, (S1, S2), T


def autoregressive_eval(model, sequences, device):
    """Run autoregressive rollout on full sequences."""
    lploss = LpLoss(size_average=True)
    log_en_err = LogEnstropyEnergyLoss()
    model.eval()
    S1, S2 = sequences.shape[1], sequences.shape[2]
    T = sequences.shape[-2]
    grid = torch2dgrid(S1, S2).to(device).unsqueeze(0)  # 1 x S1 x S2 x 2
    total_l2 = 0.0
    step_l2 = 0.0
    total_log_en_err = 0.0
    step_log_en_err = 0.0
    batches = 0
    example = {'truth': None, 'pred': None}
    loader = DataLoader(TensorDataset(sequences), batch_size=1, shuffle=False)
    with torch.no_grad():
        for (seq,) in loader:
            seq = seq.to(device)  # (1, S1, S2, T, C)
            preds = []  # predicted rollout
            prev = seq[..., 0, :]  # initial condition (B, S1, S2, C)
            for _ in range(T - 1):
                x_in = torch.cat((prev, grid.expand(prev.shape[0], -1, -1, -1)), dim=-1)
                if isinstance(model, PDERefiner):
                    prev_in = prev
                    if prev_in.dim() == 3:
                        prev_in = rearrange(prev_in, 'b h w -> b 1 1 h w')
                    elif prev_in.dim() == 4:
                        prev_in = rearrange(prev_in, 'b h w c -> b 1 c h w')
                    pred = model.validation_step(prev_in)
                    pred = rearrange(pred, 'b 1 c h w -> b h w c')
                else:
                    pred = model(x_in)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                if pred.dim() == 5:
                    pred = pred.squeeze(-2)
                if pred.dim() == 3:  # single-channel prediction without channel dim
                    pred = pred.unsqueeze(-1)
                if pred.dim() == 4:
                    pred = pred
                preds.append(pred)
                prev = pred
            pred_seq = torch.stack(preds, dim=-2)       # (1, S1, S2, T-1, C)
            truth_seq = seq[..., 1:, :]                 # align with predictions
            # print("pred_seq shape:", pred_seq.shape, "truth_seq shape:", truth_seq.shape)
            
            
            step_l2 += lploss(pred_seq[..., :1, :], truth_seq[..., :1, :]).item() # first step loss
            total_l2 += lploss(pred_seq, truth_seq).item() # overall step loss
            
            step_log_en_err += log_en_err(pred_seq[..., 0, 1], truth_seq[..., 0, 1]).item() # first step loss
            reshape_pred_seq = rearrange(pred_seq[..., 1], 'b h w t -> (b t) h w') # (B*T, H, W) 
            reshape_truth_seq = rearrange(truth_seq[..., 1], 'b h w t -> (b t) h w')
            total_log_en_err += log_en_err(reshape_pred_seq, reshape_truth_seq).item() # overall step loss
            
            batches += 1
            if example['truth'] is None:
                example['truth'] = truth_seq.detach().cpu()
                example['pred'] = pred_seq.detach().cpu()
    return total_l2 / max(1, batches), step_l2 / max(1, batches), total_log_en_err / max(1, batches), step_log_en_err / max(1, batches), example



def main():
    parser = ArgumentParser(description='Evaluate 2D operator autoregressively')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg = config['model']
    sequences, S_data, T_data = load_sw_sequences(config)

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
    else:
        raise ValueError(f'Model {model_name} not supported')
    print('model structure: ', model)

    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))


    ckpt_path = os.path.join(config.get('train', {}).get('save_dir'), config.get('train', {}).get('save_name'))
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f'Weights loaded from {ckpt_path}')
    else:
        print(f'Checkpoint not found at {ckpt_path}; evaluating with randomly initialized weights.')

    print(f'Evaluating on {sequences.shape[0]} samples at resolution {S_data[0]}x{S_data[1]} for {T_data} steps.')
    total_l2, step_l2, total_log_en_err, step_log_en_err, example = autoregressive_eval(model, sequences, device)
    print(f'Relative L2  rollout avg: {total_l2:.6f}')
    print(f'Relative L2 over first step: {step_l2:.6f}')
    print(f'Log energy error rollout avg: {total_log_en_err:.6f}')
    print(f'Log energy error over first step: {step_log_en_err:.6f}')


    
    # Save prediction and energy plots for the first example
    if example['truth'] is not None:
        plot_dir = config.get('train', {}).get('save_dir')
        pred_dir = os.path.join(plot_dir, 'saved_plots', 'predictions')
        spec_dir = os.path.join(plot_dir, 'saved_plots', 'energy')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(spec_dir, exist_ok=True)

        truth = example['truth'][0]  # (S1, S2, T-1, C)
        pred = example['pred'][0]
        T_pred = pred.shape[-2]
        time_indices = range(0, T_pred, 5)
        for t_raw in time_indices:
            pred_frame = pred[..., t_raw, 0]
            truth_frame = truth[..., t_raw, 0]
            err_frame = pred_frame - truth_frame
            truth_min = truth_frame.min().item()
            truth_max = truth_frame.max().item()
            abs_lim = max(abs(truth_min), abs(truth_max))
            vmin = -abs_lim
            vmax = abs_lim

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            titles = ['Truth', 'Prediction', 'Error']
            data_to_plot = [truth_frame, pred_frame, err_frame]
            for ax, title, data in zip(axes, titles, data_to_plot):
                if title in ['Truth', 'Prediction']:
                    im = ax.imshow(data.numpy(), cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
                else:
                    err_abs = max(abs(data.min().item()), abs(data.max().item()), 1e-8)
                    im = ax.imshow(data.numpy(), cmap='RdBu_r', origin='lower', vmin=-err_abs, vmax=err_abs)
                ax.set_title(f'{title} (T={t_raw})')
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            pred_plot_path = os.path.join(pred_dir, f'ns_prediction_t{t_raw}.png')
            fig.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Spectral energy comparison
            
            ux_pred, uy_pred = velocity_from_vorticity(pred_frame.float())
            ux_true, uy_true = velocity_from_vorticity(truth_frame.float())
            k_bins, Ek_pred = compute_spectra_torch(ux_pred, uy_pred, 2 * math.pi, 2 * math.pi)
            _, Ek_true = compute_spectra_torch(ux_true, uy_true, 2 * math.pi, 2 * math.pi)

            k_np = k_bins.cpu().numpy()
            Ek_pred_np = Ek_pred.cpu().numpy()
            Ek_true_np = Ek_true.cpu().numpy()

            valid_mask = range(1, min(len(k_np), min(S_data) // 2))
            fig_spec, ax_spec = plt.subplots(1, 1, figsize=(6, 4))
            ax_spec.loglog(k_np[valid_mask], Ek_true_np[valid_mask], label='Truth', linewidth=1)
            ax_spec.loglog(k_np[valid_mask], Ek_pred_np[valid_mask], '--', label='Prediction', linewidth=1)
            ax_spec.set_xlabel('Wavenumber k')
            ax_spec.set_ylabel('Energy E(k)')
            ax_spec.set_title(f'Spectral Energy (T={t_raw})')
            ax_spec.grid(True, which='both', alpha=0.3)
            ax_spec.legend()
            spec_plot_path = os.path.join(spec_dir, f'ns_spectral_energy_t{t_raw}.png')
            fig_spec.savefig(spec_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig_spec)
            # except Exception as exc:  # noqa: BLE001
            #     print(f'Warning: failed to create spectral energy plot at T={t_raw}: {exc}')


if __name__ == '__main__':
    main()
