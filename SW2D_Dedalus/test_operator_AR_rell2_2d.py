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
# from torch_harmonics.examples.models import SphericalFourierNeuralOperatorNet as SFNO
from torch_harmonics.examples.models import SphericalFourierNeuralOperator as SFNO
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
                if isinstance(model, SFNO):
                    x_in = x_in.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
                    pred = model(x_in) 
                    pred = pred.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
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
    # total_pred = torch.rot90(total_pred, k=1, dims=[-2, -1])
    # total_truth = torch.rot90(total_truth, k=1, dims=[-2, -1])
    # initial_condition = torch.rot90(initial_condition, k=1, dims=[-2, -1])
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
    elif model_name == 'sfno':
        model = SFNO(img_size=(S_data[0], S_data[1]), 
                                                  in_chans=model_cfg.get('in_chans', 3),
                                                  out_chans=model_cfg.get('out_chans', 1),
                                                  num_layers=model_cfg.get('num_layers', 4),
                                                  scale_factor=model_cfg.get('scale_factor', 3),
                                                  embed_dim=model_cfg.get('embed_dim', 16),
                                                  pos_embed=model_cfg.get('pos_embed', 'spectral'),
                                                  use_mlp=model_cfg.get('use_mlp', True),
                                                  normalization_layer=model_cfg.get('normalization_layer', None)
        ).to(device)
    else:
        raise ValueError(f'Model {model_name} not supported')
    print('model structure: ', model)

    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))

    save_dir = config['train']['save_dir'] if torch.cuda.is_available() else 'saved_models'
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
    
    # time_indices = range(0, truth_seq.shape[-2], 10)
    time_indices = [0, 40, 80]
    evaluate_model(truth_seq, pred_seq, model_name, seed=args.test_seed, save_dir=save_dir, time_indices=time_indices,save_csv=False)
    # exit(-1)
    # time_indices = [0, 40, truth_seq.shape[-2] - 1]
    # time_indices = range(0, truth_seq.shape[-2], 10)
    save_path = save_ground_truth_and_predictions(initial_condition, truth_seq, pred_seq, time_indices, save_dir, model_name, seed=args.test_seed)
    

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
