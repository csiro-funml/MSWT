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
from models.fno import FNO2d
from models.high_frequency_scaling import ResUNet
from models.wno import WNO2d
from models.saot import SAOTModel
from models.wavelet_transform import MultiscaleWaveletTransformer2D
from models.periodic_mswt import PeriodicMSWT2D_Patching
from models.pderefiner import PDERefiner
from models.pderefiner_unet import UNetRefiner
from einops import rearrange
from utils.criterion import LpLoss, LogEnstropyEnergyLoss, MeanEnergyAbsolutePercentageError, MeanEnergyLogRatioError, compute_2d_spectral_energy, compute_2d_enstropy_spectrum, PDEResidualLoss
from utils.compute_diagnostics import velocity_from_vorticity, compute_spectra_torch
from utils.utilities import torch2dgrid_2d
import pandas as pd


def load_ns_sequences(data_config):
    """Load full (N, X, Y, T) sequences for evaluation."""
    sub = data_config.get('sub', 1)
    sub_t = data_config.get('sub_t', 1)
    nx = data_config['nx']
    nt = data_config['nt']
    t_interval = data_config.get('time_interval', 1.0)
    datapath1 = data_config['datapath']

    S = nx // sub
    T = int(nt * t_interval) // sub_t + 1

    data1 = np.load(datapath1)
    data1 = torch.tensor(data1, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
    # print("data1 shape: ", data1.shape)
    if t_interval == 0.5:
        # subselect time to 1s 
        # data1 = NSLoader2D.extract(data1)
        sub_t = int(1//t_interval)
        data1 = data1[:, ::sub_t, ...]
        # print("data1 shape: ", data1.shape)
        
    part1 = data1.permute(0, 2, 3, 1)  # (N, X, Y, T)
    data = part1
    # print("data shape: ", data.shape)

    offset = data_config.get('offset', 0)
    n_sample = data_config.get('n_sample', data_config.get('n_samples', data_config.get('total_num', data.shape[0])))
    end = min(data.shape[0], offset + n_sample)
    data = data[offset:end]
    print("final data shape: ", data.shape) # (N, X, Y, T)
    # exit(-1)
    return data, S, T


def autoregressive_predict(model, sequences, device, grid):
    """Run autoregressive rollout on full sequences.
    
    sequences: (B, H, W, T)
    """
    model.eval()
    S = sequences.shape[1]
    T = sequences.shape[-1]
    total_pred = []
    loader = DataLoader(TensorDataset(sequences), batch_size=1, shuffle=False)
    with torch.no_grad():
        for (seq,) in loader:
            seq = seq.to(device)  # (1, S, S, T)
            preds = []  # predicted rollout
            prev = seq[..., 0]  # initial condition
            for t in range(T - 1):
                x_in = torch.cat((prev.unsqueeze(-1), grid.unsqueeze(0).expand(prev.shape[0], -1, -1, -1)), dim=-1)
                if isinstance(model, PDERefiner):
                    if len(prev.shape) == 3:
                        prev = rearrange(prev, 'b h w -> b 1 1 h w')
                    pred = model.validation_step(prev)
                    pred = rearrange(pred, 'b 1 c h w -> b h w c')
                else:
                    pred = model(x_in)
                if pred.dim() == 5:
                    pred = pred.squeeze(-2)
                if pred.dim() == 4:
                    pred = pred.squeeze(-1)
                preds.append(pred)
                prev = pred
        
            pred_seq = torch.stack(preds, dim=-1)       # (1, S, S, T-1)
            total_pred.append(pred_seq)
        total_pred = torch.stack(total_pred, dim=0)
    print("total_pred shape: ", total_pred.shape, "sequences.shape: ", sequences.shape)
    return total_pred.squeeze(1), sequences[..., 1:].to(device)


def evaluate_model(truth_seq, pred_seq, model_name, seed, save_dir):
    """ 
    truth_seq: (B, H, W, T)
    pred_seq: (B, H, W, T)
    """
    lploss = LpLoss(size_average=True)
    
    # log_en_err = LogEnstropyEnergyLoss()
    meape = MeanEnergyAbsolutePercentageError()
    melr = MeanEnergyLogRatioError()
    # pderesidual = PDEResidualLoss()


    time_idx = [0, 29, truth_seq.shape[-1] - 1]
    metrics_name = ['l2', 'SMLR', 'EMLR', 'SMAE', 'EMAE']
    metrics_dict = {}
    
    # Initialize metrics_dict in desired column order:
    # First all metrics (l2, spectral_meape, etc.) grouped by metric, then seed and model
    for metric in metrics_name:
       for t in time_idx:
           metrics_dict[metric+f'_step{t+1}'] = 0
    metrics_dict['seed'] = seed
    metrics_dict['model'] = model_name
    
    # Compute actual metric values
    for t in time_idx:
        truth_seq_t = truth_seq[..., t]
        pred_seq_t = pred_seq[..., t]
        # convert the vorcitity to velocity
        ux_true, uy_true = velocity_from_vorticity(truth_seq_t)
        ux_pred, uy_pred = velocity_from_vorticity(pred_seq_t)
        # print("ux_true shape: ", ux_true.shape, "uy_true shape: ", uy_true.shape, "ux_pred shape: ", ux_pred.shape, "uy_pred shape: ", uy_pred.shape)
        # (N, H, W)
        Ek_true = compute_2d_spectral_energy(ux_true, uy_true) #(N, H, W//2)
        Ek_pred = compute_2d_spectral_energy(ux_pred, uy_pred) 
        # print("Ek_true shape: ", Ek_true.shape, "Ek_pred shape: ", Ek_pred.shape)
        Zk_true = compute_2d_enstropy_spectrum(w_grid=truth_seq_t)  # (N, H, W)
        Zk_pred = compute_2d_enstropy_spectrum(w_grid=pred_seq_t) # (N, H, W)

        # step_pderesidual = pderesidual(ux_true, uy_true, truth_seq_t).item()
        # print("ground truth pderesidual: ", step_pderesidual)
        # step_pderesidual_pred = pderesidual(ux_pred, uy_pred, pred_seq_t).item()
        # print("predicted pderesidual: ", step_pderesidual_pred)
        # print("Zk_true shape: ", Zk_true.shape, "Zk_pred shape: ", Zk_pred.shape)
        # exit(-1)s
    
        step_l2 = lploss(pred_seq_t, truth_seq_t).item() # first step loss
        step_spectral_meape = meape(Ek_pred, Ek_true).item()
        step_spectral_melr = melr(Ek_pred, Ek_true).item()
        step_enstropy_meape = meape(Zk_pred, Zk_true).item()
        step_enstropy_melr = melr(Zk_pred, Zk_true).item()
        
        print(f"{model_name} seed: {seed}, step: {t}, step l2: {step_l2:.4f}, \
            step SMLR: {step_spectral_melr:.4f},\
            step EMLR: {step_enstropy_melr:.4f},\
            step EMAE: {step_spectral_meape:.4f},\
            step SMAE: {step_enstropy_meape:.4f}")
            
        # Use consistent f-string formatting
        metrics_dict[f'l2_step{t+1}'] = step_l2
        metrics_dict[f'SMLR_step{t+1}'] = step_spectral_melr
        metrics_dict[f'EMLR_step{t+1}'] = step_enstropy_melr
        metrics_dict[f'SMAE_step{t+1}'] = step_spectral_meape
        metrics_dict[f'EMAE_step{t+1}'] = step_enstropy_meape
        
    df_metric = pd.Series(metrics_dict).to_frame().T
    # want df_metric to have 2 level of columns: the first level is the metric name, the second level is the step number
    save_folder = os.path.join(save_dir, 'evaluation_metrics')
    os.makedirs(save_folder, exist_ok=True)
    df_metric.to_csv(os.path.join(save_folder, f'{model_name}_seed{seed}_metrics.csv'), index=False)
    return metrics_dict

def main():
    parser = ArgumentParser(description='Evaluate 2D operator autoregressively')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--test_seed', type=int, default=42, help='Seed for the random test split')
    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = config['test_data']
    model_cfg = config['model']
    sequences, S_data, T_data = load_ns_sequences(data_config)
    grid = torch2dgrid_2d(S_data, S_data, form=config['data']['grid_form'], device=device, dtype=torch.float32)
    model_name = model_cfg.get('name', 'fno2d').lower()
    
    if model_name == 'fno2d':
        model = FNO2d(in_dim=model_cfg.get('in_dim', 3),
                      out_dim=model_cfg.get('out_dim', 1),
                      modes1=model_cfg['modes1'],
                      modes2=model_cfg['modes2'],
                      fc_dim=model_cfg['fc_dim'],
                      layers=model_cfg['layers'],
                      act=model_cfg['act'],
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
    print('model structure: ', model)

    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))

    save_dir = config.get('train', {}).get('save_dir')
    save_name = config.get('train', {}).get('save_name')
    ckpt_path = os.path.join(save_dir, save_name).replace('.pt', f'_seed{args.test_seed}.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f'Weights loaded from {ckpt_path}')
    else:
        print(f'Checkpoint not found at {ckpt_path}; evaluating with randomly initialized weights.')

    print(f'Evaluating on {sequences.shape[0]} samples at resolution {S_data}x{S_data} for {T_data} steps.')
    # total_l2, step_l2, total_log_en_err, step_log_en_err, example = autoregressive_eval(model, sequences, device, grid)
    truth_seq, pred_seq = autoregressive_predict(model, sequences, device, grid)
    
    evaluate_model(truth_seq, pred_seq, model_name, seed=args.test_seed, save_dir=save_dir)
    

    example = {'truth': truth_seq.detach().cpu(), 'pred': pred_seq.detach().cpu()}
    # Save prediction and energy plots for the first example
    if example['truth'] is not None:
        plot_dir = config.get('train', {}).get('save_dir')
        pred_dir = os.path.join(plot_dir, 'saved_plots', 'predictions', f'seed{args.test_seed}')
        spec_dir = os.path.join(plot_dir, 'saved_plots', 'energy', f'seed{args.test_seed}')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(spec_dir, exist_ok=True)

        truth = example['truth'][0]  # (S, S, T-1)
        pred = example['pred'][0]
        T_pred = pred.shape[-1]
        time_indices = range(0, T_pred, 5)
        for t_raw in time_indices:
            pred_frame = pred[..., t_raw]
            truth_frame = truth[..., t_raw]
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

            valid_mask = range(1, min(len(k_np), S_data // 2))
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
