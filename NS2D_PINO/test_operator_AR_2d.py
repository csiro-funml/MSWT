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
from einops import rearrange
from utils.criterion import LpLoss, MeanEnergyAbsolutePercentageError, MeanEnergyLogRatioError, get_forcing_vel, PINO_loss3d_vel
from utils.compute_diagnostics import velocity_from_vorticity, compute_spectra_torch, compute_enstropy_torch
from utils.utilities import torch2dgrid_2d
from data_utils.datasets_dedalus import NSLoader2D
import pandas as pd





def autoregressive_predict(model, test_loader, device, grid):
    """Run autoregressive rollout on full sequences.
    
    test_loader: DataLoader
    """
    model.eval()
    total_pred = []
    initial_condition = []
    total_ground_truth = []
    with torch.no_grad():
        for (seq, truth) in test_loader:
            seq = seq.to(device)  # (B, T, S, S, C)
            truth = truth.to(device) # (B, T, S, S, C)
            T = seq.shape[1]
            preds = []  # predicted rollout
            prev = seq[..., 0]  # initial condition (B, S, S, C)
            initial_condition.append(prev)
            total_ground_truth.append(truth) # (B, S, S, C)
            preds.append(prev) # append the initial condition
            for t in range(T):
                x_in = torch.cat((prev, grid.unsqueeze(0).expand(prev.shape[0], -1, -1, -1)), dim=-1)
                print("x_in shape: ", x_in.shape)
                pred = model(x_in)
                print("pred shape: ", pred.shape)
                if pred.dim() == 5:
                    pred = pred.squeeze(-2)
                if pred.dim() == 4:
                    pred = pred.squeeze(-1)
                preds.append(pred)
                prev = pred
        
            pred_seq = torch.stack(preds, dim=-1)       # (B, S, S, C, T+1)
            total_pred.append(pred_seq)
        total_pred = torch.stack(total_pred, dim=0) # (N, S, S, C, T+1)
        initial_condition = torch.stack(initial_condition, dim=0) # (N, S, S, C)
        total_ground_truth = torch.stack(total_ground_truth, dim=0) # (N, S, S, C, T)
    print("total_pred shape: ", total_pred.shape, "total_ground_truth shape: ", total_ground_truth.shape, "initial_condition shape: ", initial_condition.shape)
    return initial_condition, total_pred, total_ground_truth


def evaluate_model(truth_seq, pred_seq, model_name, seed, save_dir, save_csv=False):
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

        k_bins, Ek_pred = compute_spectra_torch(ux_pred, uy_pred, 2 * math.pi, 2 * math.pi)
        _, Ek_true = compute_spectra_torch(ux_true, uy_true, 2 * math.pi, 2 * math.pi)
        
        _, Zk_true = compute_enstropy_torch(truth_seq_t.float(), 2 * math.pi, 2 * math.pi)
        _, Zk_pred = compute_enstropy_torch(pred_seq_t.float(), 2 * math.pi, 2 * math.pi)
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
    if save_csv:
        os.makedirs(save_folder, exist_ok=True)
        df_metric.to_csv(os.path.join(save_folder, f'{model_name}_seed{seed}_metrics.csv'), index=False)
    return metrics_dict


def compute_PDE_loss(initial_condition, truth_seq, pred_seq, forcing):
    """
    truth_seq:  (N, S, S, C, T)
    pred_seq:  (N, S, S, C, T+1)
    initial_condition: (N, S, S, C)
    """
    rel_l2 = LpLoss(size_average=True)
    test_rel_l2 = rel_l2(pred_seq[..., 1:], truth_seq)
    print(f"Test relative L2 loss: {test_rel_l2.item()}")

    u = pred_seq.permute(0, 3, 1, 2, 4) # (N, S, S, C, T+1) -> (N, C, S, S, T+1)
    u0 = initial_condition.permute(0, 3, 1, 2) # (N, S, S, C) -> (N, C, S, S)
    loss_ic, loss_cont, loss_momx, loss_momy = PINO_loss3d_vel(u, u0, forcing.clone())
    print(f"Predicted sequence PDE loss: {loss_cont.item()}, IC loss: {loss_ic.item()}, momx loss: {loss_momx.item()}, momy loss: {loss_momy.item()}")    


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
        truth_seq_t = truth_seq[0,..., t].detach().cpu()
        pred_seq_t = pred_seq[0,..., t].detach().cpu()
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
        pred_frame = pred_seq.detach().cpu()[0, ..., t_raw]
        truth_frame = truth_seq.detach().cpu()[0, ..., t_raw]
       
        # Spectral energy comparison 
        ux_pred, uy_pred = velocity_from_vorticity(pred_frame.float())
        ux_true, uy_true = velocity_from_vorticity(truth_frame.float())
        
        k_bins, Ek_pred = compute_spectra_torch(ux_pred, uy_pred, 2 * math.pi, 2 * math.pi)
        _, Ek_true = compute_spectra_torch(ux_true, uy_true, 2 * math.pi, 2 * math.pi)
        
        _, Zk_true = compute_enstropy_torch(truth_frame.float(), 2 * math.pi, 2 * math.pi)
        _, Zk_pred = compute_enstropy_torch(pred_frame.float(), 2 * math.pi, 2 * math.pi)
        
        k_np = k_bins.detach().cpu().numpy()
        valid_mask = range(1, min(len(k_np), truth_frame.shape[-1] // 2))
        k_np = k_np[valid_mask]
        Ek_true_np = Ek_true.detach().cpu().numpy()[valid_mask]
        Ek_pred_np = Ek_pred.detach().cpu().numpy()[valid_mask]
        Zk_true_np = Zk_true[valid_mask]
        Zk_pred_np = Zk_pred[valid_mask]
        save_path = os.path.join(save_dir, f'{model_name}_seed{seed}_energy_spectra_t{t_raw+1}.npz')
        np.savez(save_path, k_np=k_np, Ek_true_np=Ek_true_np, Ek_pred_np=Ek_pred_np, Zk_true_np=Zk_true_np, Zk_pred_np=Zk_pred_np)
        print(f"Saved energy spectra to {save_path}")
    return save_path


def main():
    parser = ArgumentParser(description='Evaluate 2D operator autoregressively')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--test_seed', type=int, default=42, help='Seed for the random test split')
    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    model_cfg = config['model']
    
    test_set = NSLoader2D(datapath=data_config['datapath'],
                                state='test',
                                train=False,
                                normalizer_path=data_config.get('normalizer_path', None))
                                
    test_set.transform_rollout(T=data_config['nt']) # convert  10 realizations of (N*T, H, W, C) to (N, T, H, W, C) for autoregressive rollout
    test_loader = DataLoader(test_set,
                                 batch_size=config['train']['batchsize'],
                                 shuffle=False,
                                 num_workers=config['train'].get('num_workers', 1))
    S_data = test_set.S # (H, W)
    T_data = test_set.T # T
    grid = torch2dgrid_2d(S_data[0], S_data[1], form=config['data']['grid_form'], device=device, dtype=torch.float32)
    forcing = get_forcing_vel(S_data[0]).to(device)
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
    else:
        raise ValueError(f'Model {model_name} not supported')
    # print('model structure: ', model)

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

    print(f'Evaluating on {len(test_set)} samples at resolution {S_data[0]}x{S_data[1]} for {T_data} steps.')
    # total_l2, step_l2, total_log_en_err, step_log_en_err, example = autoregressive_eval(model, sequences, device, grid)
    initial_condition, pred_seq, truth_seq = autoregressive_predict(model, test_loader, device, grid)
    # Function 1, evaluate the model and save the metrics
    # evaluate_model(truth_seq, pred_seq, model_name, seed=args.test_seed, save_dir=save_dir, save_csv=False)
    # exit(-1)
    
    compute_PDE_loss(initial_condition, truth_seq, pred_seq, forcing=forcing)
     
    # Function 2, for time_indecs = [0, 29, truth_seq.shape[-1] - 1], save the ground truth and predictions as npz file,
    # time_indices = [0, 29, truth_seq.shape[-1] - 1]
    # save_path = save_ground_truth_and_predictions(initial_condition, truth_seq, pred_seq, time_indices, save_dir, model_name, seed=args.test_seed)
    


if __name__ == '__main__':
    main()
