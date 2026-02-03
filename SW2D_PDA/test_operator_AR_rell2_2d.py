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
from models.mswt import PeriodicMSWT2D_Patching
from models.pderefiner import PDERefiner
from models.pderefiner_unet import UNetRefiner
from einops import rearrange
from utils.criterion import LpLoss, LogEnstropyEnergyLoss, MeanEnergyAbsolutePercentageError, MeanEnergyLogRatioError, compute_2d_spectral_energy, compute_2d_enstropy_spectrum, PDEResidualLoss
from utils.compute_diagnostics import velocity_from_vorticity, compute_spectra_torch, compute_enstropy_torch
from utils.utilities import torch2dgrid_2d
import pandas as pd



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

def evaluate_model(truth_seq, pred_seq, model_name, seed, save_dir, time_indices, save_csv=False):
    """ 
    truth_seq: (B, H, W, T, C)
    pred_seq: (B, H, W, T, C)
    """
    lploss = LpLoss(size_average=True)
    
    # log_en_err = LogEnstropyEnergyLoss()
    meape = MeanEnergyAbsolutePercentageError()
    melr = MeanEnergyLogRatioError()
    # pderesidual = PDEResidualLoss()


    metrics_name = ['l2', 'SMLR', 'EMLR', 'SMAE', 'EMAE']
    metrics_dict = {}
    
    # Initialize metrics_dict in desired column order:
    # First all metrics (l2, spectral_meape, etc.) grouped by metric, then seed and model
    for metric in metrics_name:
       for t in time_indices:
           metrics_dict[metric+f'_step{t+1}'] = 0
    metrics_dict['seed'] = seed
    metrics_dict['model'] = model_name
    
    # Compute actual metric values
    for t in time_indices:
        truth_seq_t = truth_seq[..., t, :] # the first channel for vorticity
        pred_seq_t = pred_seq[..., t, :] # the first channel for vorticity
        # convert the vorcitity to velocity
        ux_true, uy_true = velocity_from_vorticity(truth_seq_t[..., 0])
        ux_pred, uy_pred = velocity_from_vorticity(pred_seq_t[..., 0])
        # print("ux_true shape: ", ux_true.shape, "uy_true shape: ", uy_true.shape, "ux_pred shape: ", ux_pred.shape, "uy_pred shape: ", uy_pred.shape)
        # (N, H, W)
        # Ek_true = compute_2d_spectral_energy(ux_true, uy_true) #(N, H, W//2)
        # Ek_pred = compute_2d_spectral_energy(ux_pred, uy_pred) 
        # # print("Ek_true shape: ", Ek_true.shape, "Ek_pred shape: ", Ek_pred.shape)
        # Zk_true = compute_2d_enstropy_spectrum(w_grid=truth_seq_t[..., 0])  # (N, H, W)
        # Zk_pred = compute_2d_enstropy_spectrum(w_grid=pred_seq_t[..., 0]) # (N, H, W)

        k_bins, Ek_pred = compute_spectra_torch(ux_pred, uy_pred, 2 * math.pi, 2 * math.pi)
        _, Ek_true = compute_spectra_torch(ux_true, uy_true, 2 * math.pi, 2 * math.pi)
        
        _, Zk_true = compute_enstropy_torch(truth_seq_t[..., 0].float(), 2 * math.pi, 2 * math.pi)
        _, Zk_pred = compute_enstropy_torch(pred_seq_t[..., 0].float(), 2 * math.pi, 2 * math.pi)
        # print("Ek_true shape: ", Ek_true.shape, "Ek_pred shape: ", Ek_pred.shape, "Zk_true shape: ", Zk_true.shape, "Zk_pred shape: ", Zk_pred.shape)
        k_np = k_bins
        valid_mask = range(1, min(len(k_np), min(truth_seq_t.shape[1], truth_seq_t.shape[2]) // 2)) # important, because we need to truncate the energy spectra and enstropy spectrum to the same length
        # print("valid_mask: ", valid_mask)
        k_np = k_np[valid_mask]
        Ek_true_truncated = Ek_true[:, valid_mask]
        Ek_pred_truncated = Ek_pred[:, valid_mask]
        Zk_true_truncated = Zk_true[:, valid_mask]
        Zk_pred_truncated = Zk_pred[:, valid_mask]

        # step_pderesidual = pderesidual(ux_true, uy_true, truth_seq_t).item()
        # print("ground truth pderesidual: ", step_pderesidual)
        # step_pderesidual_pred = pderesidual(ux_pred, uy_pred, pred_seq_t).item()
        # print("predicted pderesidual: ", step_pderesidual_pred)
        # print("Zk_true shape: ", Zk_true.shape, "Zk_pred shape: ", Zk_pred.shape)
        # exit(-1)s
    
        step_l2 = lploss(pred_seq_t, truth_seq_t).item() # first step loss
        step_spectral_meape = meape(Ek_pred_truncated, Ek_true_truncated).item()
        step_spectral_melr = melr(Ek_pred_truncated, Ek_true_truncated).item()
        step_enstropy_meape = meape(Zk_pred_truncated, Zk_true_truncated).item()
        step_enstropy_melr = melr(Zk_pred_truncated, Zk_true_truncated).item()
        
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
    if save_csv:
        df_metric.to_csv(os.path.join(save_folder, f'{model_name}_seed{seed}_metrics.csv'), index=False)
    return metrics_dict


def save_ground_truth_and_predictions(initial_condition, truth_seq, pred_seq, time_indices, save_dir, model_name, seed):
    """
    truth_seq: (B, H, W, T)
    pred_seq: (B, H, W, T)
    time_indices: list of time indices
    save_dir: directory to save the ground truth and predictions
    """
    save_dir = os.path.join(save_dir, 'saved_plots')
    os.makedirs(save_dir, exist_ok=True)
    initial_condition = initial_condition.detach().cpu()
    for t in time_indices:
        truth_seq_t = truth_seq[..., t, 0].detach().cpu()
        pred_seq_t = pred_seq[..., t, 0].detach().cpu()
        error_seq_t = pred_seq_t - truth_seq_t
        save_path = os.path.join(save_dir, f'{model_name}_seed{seed}_prediction_t{t+1}.npz')
        np.savez(save_path, initial_condition=initial_condition, truth_seq_t=truth_seq_t, pred_seq_t=pred_seq_t, error_seq_t=error_seq_t)
        print(f"Saved ground truth and predictions to {save_path}")
    return save_path


def compute_save_energy_spectra(truth_seq, pred_seq, time_indices, save_dir, model_name, seed):
    """
    truth_seq: (B, H, W, T)
    pred_seq: (B, H, W, T)
    time_indices: list of time indices
    save_dir: directory to save the energy spectra
    model_name: name of the model
    seed: seed for the random test split
    """
    save_dir = os.path.join(save_dir, 'saved_plots')
    os.makedirs(save_dir, exist_ok=True)
    for t_raw in time_indices:
        pred_frame = pred_seq.detach().cpu()[..., t_raw, 0]
        truth_frame = truth_seq.detach().cpu()[..., t_raw, 0]
       
        # Spectral energy comparison 
        ux_pred, uy_pred = velocity_from_vorticity(pred_frame.float())
        ux_true, uy_true = velocity_from_vorticity(truth_frame.float())
        
        k_bins, Ek_pred = compute_spectra_torch(ux_pred, uy_pred, 2 * math.pi, 2 * math.pi)
        _, Ek_true = compute_spectra_torch(ux_true, uy_true, 2 * math.pi, 2 * math.pi)
        
        _, Zk_true = compute_enstropy_torch(truth_frame.float(), 2 * math.pi, 2 * math.pi)
        _, Zk_pred = compute_enstropy_torch(pred_frame.float(), 2 * math.pi, 2 * math.pi)
        k_np = k_bins.detach().cpu().numpy()
        valid_mask = range(1, min(len(k_np), min(truth_frame.shape[-1], truth_frame.shape[-2]) // 2))
        k_np = k_np[valid_mask]
        Ek_true_np = Ek_true.detach().cpu().numpy()[:, valid_mask]
        Ek_pred_np = Ek_pred.detach().cpu().numpy()[:, valid_mask]
        Zk_true_np = Zk_true[:, valid_mask]
        Zk_pred_np = Zk_pred[:, valid_mask]
        save_path = os.path.join(save_dir, f'{model_name}_seed{seed}_energy_spectra_t{t_raw+1}.npz')
        np.savez(save_path, k_np=k_np, Ek_true_np=Ek_true_np, Ek_pred_np=Ek_pred_np, Zk_true_np=Zk_true_np, Zk_pred_np=Zk_pred_np)
        print(f"Saved energy spectra to {save_path}")
    return save_path


def autoregressive_predict(model, sequences, device, grid=None):
    """Run autoregressive rollout on full sequences."""
    model.eval()
    T = sequences.shape[-2]
    loader = DataLoader(TensorDataset(sequences), batch_size=16, shuffle=False)
    initial_condition = []
    pred_seq_list = []
    truth_seq_list = []

    with torch.no_grad():
        for (seq,) in loader:
            seq = seq.to(device)  # (B, S1, S2, T, C) where B is batch size
            preds = []  # predicted rollout
            prev = seq[..., 0, :]  # initial condition (B, S1, S2, C)
            # Append the full batch tensor to the list (handles variable batch sizes)
            initial_condition.append(prev)
            for _ in range(T - 1):
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
                prev = pred
            pred_seq = torch.stack(preds, dim=-2)       # (B, S1, S2, T-1, C)
            truth_seq = seq[..., 1:, :]                 # (B, S1, S2, T-1, C) align with predictions
            # Append the full batch tensors to the list (handles variable batch sizes)
            pred_seq_list.append(pred_seq)
            truth_seq_list.append(truth_seq)
    # Concatenate all batches along the first dimension to get (N, H, W, ...)
    initial_condition = torch.cat(initial_condition, dim=0)  # (N, S1, S2, C)
    pred_seq = torch.cat(pred_seq_list, dim=0)              # (N, S1, S2, T-1, C)
    truth_seq = torch.cat(truth_seq_list, dim=0)            # (N, S1, S2, T-1, C)
    print("initial_condition shape:", initial_condition.shape, "pred_seq shape:", pred_seq.shape, "truth_seq shape:", truth_seq.shape)
            
    return initial_condition, pred_seq, truth_seq
    # return total_l2 / max(1, batches), step_l2 / max(1, batches), total_log_en_err / max(1, batches), step_log_en_err / max(1, batches), example



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

    save_dir = config.get('train', {}).get('save_dir')
    ckpt_path = os.path.join(config.get('train', {}).get('save_dir'), config.get('train', {}).get('save_name'))
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f'Weights loaded from {ckpt_path}')
    else:
        print(f'Checkpoint not found at {ckpt_path}; evaluating with randomly initialized weights.')

    print(f'Evaluating on {sequences.shape[0]} samples at resolution {S_data[0]}x{S_data[1]} for {T_data} steps.')
    use_external_grid = model_cfg.get('external_grid', True)
    grid = torch2dgrid_2d(S_data[0], S_data[1], form=config['data']['grid_form'], device=device, dtype=torch.float32)
    
    
    initial_condition, pred_seq, truth_seq = autoregressive_predict(model, sequences, device, grid)

    # time_indices = range(0, truth_seq.shape[-2], 10)
    time_indices = [0, 40, 80]
    evaluate_model(truth_seq, pred_seq, model_name, seed=args.test_seed, save_dir=save_dir, time_indices=time_indices,save_csv=False)
    exit(-1)
    # time_indices = [0, 40, truth_seq.shape[-2] - 1]
    # time_indices = range(0, truth_seq.shape[-2], 10)
    save_path = save_ground_truth_and_predictions(initial_condition, truth_seq, pred_seq, time_indices, save_dir, model_name, seed=args.test_seed)
    


if __name__ == '__main__':
    main()
