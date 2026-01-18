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
from models.periodic_mswt import  PeriodicMSWT2D_Patching
from models.pderefiner import PDERefiner
from models.pderefiner_unet import UNetRefiner
from einops import rearrange
from utils.criterion import LpLoss, LogEnstropyEnergyLoss
from utils.compute_diagnostics import velocity_from_vorticity, compute_spectra_torch
from utils.utilities import torch2dgrid_2d
from data_utils.datasets import SWLoader2D


def autoregressive_predict(model, test_loader, device, grid):
    """Run autoregressive rollout on full sequences."""
    
    model.eval()
    
    total_pred = []
    total_truth = []
    initial_condition = []

    with torch.no_grad():
        for (seq, truth) in test_loader:
            seq = seq.to(device)  # (N, T, H, W, C) 
            truth = truth.to(device) # (N, T, H, W, C)
            preds = []  # predicted rollout
            trues = []
            prev = seq[..., 0, :]  # initial condition (B, S1, S2, C)
            initial_condition.append(prev)
            T = seq.shape[-2]
            for t in range(T):
                if grid is not None:
                    x_in = torch.cat((prev, grid.expand(prev.shape[0], -1, -1, -1)), dim=-1)
                else:
                    x_in = prev
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
                trues.append(truth[..., t, :])
                prev = pred
                
            pred_seq = torch.stack(preds, dim=-2)       # (1, S1, S2, T-1, C)
            truth_seq = torch.stack(trues, dim=-2)               # align with predictions
            total_pred.append(pred_seq)
            total_truth.append(truth_seq)
            print("pred_seq shape:", pred_seq.shape, "truth_seq shape:", truth_seq.shape)

    total_pred = torch.cat(total_pred, dim=0)[..., 0] # (N, H, W, T)
    total_truth = torch.cat(total_truth, dim=0)[..., 0] # (N, H, W, T)
    initial_condition = torch.cat(initial_condition, dim=0)[..., 0] # (N, H, W)
    print("total_pred shape:", total_pred.shape, "total_truth shape:", total_truth.shape, "initial_condition shape:", initial_condition.shape)        
    return initial_condition, total_pred, total_truth


def main():
    parser = ArgumentParser(description='Evaluate 2D operator autoregressively')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--test_seed', type=int, help='Seed for the test set')
    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
        config['train']['save_name'] = config['train']['save_name'].replace('.pt', f'_seed{args.test_seed}.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg = config['model']
    
    data_config = config['data']
    test_set = SWLoader2D(datapath=data_config['datapath'],
                                state='test',
                                train=False,
                                normalizer_path=data_config.get('normalizer_path', None))
                                
    test_set.transform_rollout(T=data_config['nt']) # convert (N*T, H, W, C) to (N, T, H, W, C) for autoregressive rollout
    test_loader = DataLoader(test_set,
                                 batch_size=config['train']['batchsize'],
                                 shuffle=False,
                                 num_workers=config['train'].get('num_workers', 1))
    S_data = test_set.S # (H, W)
    T_data = test_set.T # T
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
    elif model_name in ['multiscale_wavelet2d_periodic_patching', 'mswt_periodic_patching', 'periodic_mswt_patching']:
        model = PeriodicMSWT2D_Patching(
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

    print(f'Evaluating on {len(test_set)} samples at resolution {S_data[0]}x{S_data[1]} for {T_data} steps.')
    use_external_grid = model_cfg.get('external_grid', True)
    grid = torch2dgrid_2d(S_data[0], S_data[1], form=config['data']['grid_form'], device=device, dtype=torch.float32)
    
    
    # total_l2, step_l2, total_log_en_err, step_log_en_err, example = autoregressive_eval(model, sequences, device, grid)
    initial_condition, pred_seq, truth_seq = autoregressive_predict(model, test_loader, device, grid)
    
    plot_dir = config.get('train', {}).get('save_dir')
    pred_dir = os.path.join(plot_dir, 'saved_plots', 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    time_indices = range(0, pred_seq.shape[-1], 10)
    for t_raw in time_indices:
        # print("t_raw:", t_raw)
        pred_frame = pred_seq[0, ..., t_raw].cpu()
        truth_frame = truth_seq[0, ..., t_raw].cpu()
        # print("pred_frame shape:", pred_frame.shape, "truth_frame shape:", truth_frame.shape, "pred_seq shape:", pred_seq.shape, "truth_seq shape:", truth_seq.shape)
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
    exit(-1)  
    
    

    evaluate_model(truth_seq, pred_seq, model_name, seed=args.test_seed, save_dir=save_dir)
    
    
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
