"""
Standalone script for analyzing neural operator predictions.

This script can:
1. Load .pth files (from test_AR_NO_Dedalus.py line 334) and print data shapes
   - Allows collaborator to compute their own metrics
   
2. Load .npz files (from test_AR_NO_Dedalus.py line 486) and:
   - Show data structure and variables
   - Explain how metrics were computed
   - Create animations (prediction comparison and spectral comparison)

Requirements:
- numpy
- torch
- matplotlib
- Files in data_generation/dedalus/ folder (for compute_spectra, streamfunction_to_velocity)

Usage:
    # Load .pth file and print shapes
    python analyze_predictions_collaborator.py --input_file path/to/test_data_prediction_long.pth --mode shapes
    
    # Load .npz file and create animations
    python analyze_predictions_collaborator.py --input_file path/to/test_data_prediction_long.npz --mode animate --num_animation_frames 250
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import time

# Add path to data_generation for spectral functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_generation', 'dedalus'))

try:
    from ns2d.spectral import compute_spectra
    from scripts.compute_diagnostics import streamfunction_to_velocity
except ImportError:
    print("Warning: Could not import from data_generation/dedalus. Some functions may not work.")
    print("Make sure data_generation/dedalus/ folder is accessible.")
    compute_spectra = None
    streamfunction_to_velocity = None



def load_npz_file(file_path):
    """
    Load .npz file and show data structure.
    
    Args:
        file_path: Path to .npz file
        
    Returns:
        Dictionary with all saved data
    """
    print(f"\n{'='*60}")
    print(f"Loading .npz file: {file_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    
    print("Data structure:")
    print(f"  Keys: {list(data.keys())}\n")
    
    # Show main arrays
    if 'pred' in data:
        pred = data['pred']
        print(f"  pred: {pred.shape} (N, H, W, T, C) - predictions")
    if 'output' in data:
        output = data['output']
        print(f"  output: {output.shape} (N, H, W, T, C) - ground truth")
    
    print("the channels are: pressure, velocity_x, velocity_y")
    
    # Convert to dictionary for easier access
    save_data = {}
    for key in data.keys():
        value = data[key]
        # Handle numpy array with object dtype (contains lists/dicts)
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                # Object array - extract the Python object
                save_data[key] = value.item()
            elif value.size == 1:
                # Scalar array - extract the scalar
                save_data[key] = value.item()
            else:
                # Regular array - keep as numpy array
                save_data[key] = value
        elif isinstance(value, (np.str_, np.unicode_)):
            # Convert numpy string to Python string
            save_data[key] = str(value)
        else:
            save_data[key] = value    
    return save_data


def explain_metrics(save_data):
    """
    Explain how the metrics in save_data were computed.
    """
    print(f"\n{'='*60}")
    print("How metrics were computed:")
    print(f"{'='*60}\n")
    
    if 'dataset_form' in save_data:
        form = save_data['dataset_form']
        print(f"Dataset form: {form}\n")
        
        if form == 'vorticity':
            print("For vorticity form:")
            print("  - Channels are [vorticity, streamfunction]")
            print("  - Velocity is computed from streamfunction:")
            print("    u = ∂ψ/∂y, v = -∂ψ/∂x")
            print("  - This uses streamfunction_to_velocity() function")
        elif form == 'velocity':
            print("For velocity form:")
            print("  - Channels are [pressure, velocity_x, velocity_y]")
            print("  - Velocity components are used directly")
    
    print("\n1. Relative L2 Loss:")
    print("   Computed per step using:")
    print("     rel_l2_error = ||pred - target||_2 / ||target||_2")
    print("   Stored in 'rel_l2_loss_by_step' as list of dicts")
    
    if 'spectral_data_by_step' in save_data and len(save_data['spectral_data_by_step']) > 0:
        print("\n2. Spectral Metrics:")
        print("   For each step:")
        print("   a) Extract velocity components (ux, uy)")
        print("   b) Compute energy spectrum E(k) using compute_spectra()")
        print("      - E(k) = 0.5 * <|û|²>_shell (shell-averaged in Fourier space)")
        print("   c) Compute enstrophy spectrum Z(k) using compute_spectra()")
        print("      - Z(k) = <|ω̂|²>_shell")
        print("   d) Compute total energy:")
        print("      - E = 0.5 * ∫ (ux² + uy²) dx / Area")
        print("   Stored in 'spectral_data_by_step' as list of dicts")
        print("   Each dict contains:")
        print("     - time_step: step index")
        print("     - k_bins: wavenumber bins")
        print("     - Ek_pred, Ek_target: energy spectra")
        print("     - Zk_pred, Zk_target: enstrophy spectra")
        print("     - energy_pred, energy_target: total energy")
        print("     - H, Lx: grid resolution and domain size")
    
    print()


def animate_predictions(save_data, output_dir=None, save_animation=True, fps=10, num_animation_frames=None):
    """
    Create animation showing target, prediction, and error for each channel.
    
    Args:
        save_data: Dictionary with 'pred' and 'output' arrays
        output_dir: Directory to save animation
        save_animation: Whether to save animation file
        fps: Frames per second for animation
        num_animation_frames: Desired number of frames (None = use all)
    """
    pred = save_data['pred']  # (N, H, W, T, C)
    target = save_data['output']  # (N, H, W, T, C)
    
    # Convert to numpy if torch tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    
    # Get total available steps
    total_steps = pred.shape[0]
    
    # Calculate step interval if num_animation_frames is specified
    step_interval = None
    if num_animation_frames is not None and num_animation_frames > 0:
        step_interval = max(1, total_steps // num_animation_frames)
        num_steps = min(num_animation_frames, total_steps // step_interval)
        step_indices = np.arange(0, total_steps, step_interval)[:num_steps]
        pred = pred[step_indices]
        target = target[step_indices]
        print(f"Using {num_steps} frames with step interval {step_interval} (from {total_steps} total steps)")
    else:
        num_steps = total_steps
        step_indices = np.arange(num_steps)
        print(f"Using all {num_steps} steps for animation")
    
    H, W = pred.shape[1], pred.shape[2]
    num_channels = pred.shape[-1]
    
    # Use first timestep in output (usually T=1)
    pred = pred[..., 0, :]  # (N, H, W, C)
    target = target[..., 0, :]  # (N, H, W, C)
    
    # Compute error
    error = pred - target  # (N, H, W, C)
    
    # Find min/max per channel
    vmin_common = []
    vmax_common = []
    vmin_error = []
    vmax_error = []
    
    for col in range(num_channels):
        vmin_ch = min(target[:, :, :, col].min(), pred[:, :, :, col].min())
        vmax_ch = max(target[:, :, :, col].max(), pred[:, :, :, col].max())
        vmin_common.append(vmin_ch)
        vmax_common.append(vmax_ch)
        
        vmin_err_ch = error[:, :, :, col].min()
        vmax_err_ch = error[:, :, :, col].max()
        vmax_error_abs_ch = max(abs(vmin_err_ch), abs(vmax_err_ch))
        vmin_error.append(-vmax_error_abs_ch)
        vmax_error.append(vmax_error_abs_ch)
    
    # Channel names
    dataset_form = save_data.get('dataset_form', 'vorticity')
    if dataset_form == 'vorticity':
        channel_names = ['Vorticity', 'Streamfunction']
    elif dataset_form == 'velocity':
        channel_names = ['Pressure', 'Velocity X', 'Velocity Y']
    else:
        channel_names = [f'Channel {i}' for i in range(num_channels)]
    
    # Create figure
    fig, axes = plt.subplots(3, num_channels, figsize=(5*num_channels, 12))
    if num_channels == 1:
        axes = axes.reshape(-1, 1)
    
    # Initialize images
    imgs = []
    for row in range(3):
        row_imgs = []
        for col in range(num_channels):
            ax = axes[row, col]
            if row == 0:  # Target
                im = ax.imshow(target[0, :, :, col], cmap='RdBu_r', 
                              vmin=vmin_common[col], vmax=vmax_common[col], origin='lower')
                ax.set_title(f'{channel_names[col] if col < len(channel_names) else f"Channel {col}"}\nTarget')
            elif row == 1:  # Prediction
                im = ax.imshow(pred[0, :, :, col], cmap='RdBu_r',
                              vmin=vmin_common[col], vmax=vmax_common[col], origin='lower')
                ax.set_title(f'Prediction')
            else:  # Error
                im = ax.imshow(error[0, :, :, col], cmap='RdBu_r',
                              vmin=vmin_error[col], vmax=vmax_error[col], origin='lower')
                ax.set_title(f'Error')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
            row_imgs.append(im)
        imgs.append(row_imgs)
    
    # Step counter
    step_text = fig.suptitle('Step: 0', fontsize=16, y=0.98)
    
    def animate(frame):
        actual_step = step_indices[frame] if frame < len(step_indices) else frame
        step_text.set_text(f'Step: {actual_step} (frame {frame})')
        for row in range(3):
            for col in range(num_channels):
                if row == 0:  # Target
                    imgs[row][col].set_data(target[frame, :, :, col])
                elif row == 1:  # Prediction
                    imgs[row][col].set_data(pred[frame, :, :, col])
                else:  # Error
                    imgs[row][col].set_data(error[frame, :, :, col])
        return [img for row in imgs for img in row] + [step_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=num_steps, 
                                  interval=1000/fps, blit=False, repeat=True)
    
    # Save animation
    if save_animation and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if step_interval is not None:
            step_range = f"steps{step_indices[0]}-{step_indices[-1]}_interval{step_interval}_frames{num_steps}"
        else:
            step_range = f"totalsteps{num_steps}"
        anim_path = f'{output_dir}/prediction_animation_{step_range}.mp4'
        print(f"Saving animation to {anim_path}...")
        print(f"  Animation details: {num_steps} frames, {fps} fps, bitrate=1800")
        start_time = time.time()
        anim.save(anim_path, writer='ffmpeg', fps=fps, bitrate=1800)
        elapsed_time = time.time() - start_time
        print(f"Animation saved (took {elapsed_time:.2f} seconds, ~{elapsed_time/60:.2f} minutes)")
    
    return anim, fig


def animate_spectral_comparison(save_data, output_dir=None, save_animation=True, 
                                fps=10, k_zoom_threshold=20, num_animation_frames=None):
    """
    Create animation comparing spectral energy and enstrophy.
    
    Args:
        save_data: Dictionary with spectral data
        output_dir: Directory to save animation
        save_animation: Whether to save animation file
        fps: Frames per second for animation
        k_zoom_threshold: Wavenumber threshold for zoomed view
        num_animation_frames: Desired number of frames (None = use all)
    """
    spectral_data_list = save_data['spectral_data_by_step']
    
    if not spectral_data_list:
        print("Warning: No spectral data found in save_data")
        return None, None
    
    # Get total available steps
    total_steps = len(spectral_data_list)
    
    # Calculate step interval if num_animation_frames is specified
    step_interval = None
    if num_animation_frames is not None and num_animation_frames > 0:
        step_interval = max(1, total_steps // num_animation_frames)
        num_steps = min(num_animation_frames, total_steps // step_interval)
        step_indices = np.arange(0, total_steps, step_interval)[:num_steps]
        spectral_data_list = [spectral_data_list[i] for i in step_indices]
        print(f"Using {num_steps} frames with step interval {step_interval} (from {total_steps} total steps)")
    else:
        num_steps = total_steps
        spectral_data_list = spectral_data_list[:num_steps]
        print(f"Using all {num_steps} steps for animation")
    
    # Extract k_bins
    k_bins = spectral_data_list[0]['k_bins']
    
    # Get H and Lx for Nyquist truncation
    H = spectral_data_list[0].get('H', k_bins.shape[0])
    Lx = spectral_data_list[0].get('Lx', 2 * np.pi)
    
    # Compute Nyquist truncation
    k_nyquist = int((np.pi * H) // Lx)
    start_truth = 1
    
    # Find global min/max for y-axis
    all_Ek_target = [data['Ek_target'][start_truth:k_nyquist] for data in spectral_data_list]
    all_Ek_pred = [data['Ek_pred'][start_truth:k_nyquist] for data in spectral_data_list]
    all_Zk_target = [data['Zk_target'][start_truth:k_nyquist] for data in spectral_data_list]
    all_Zk_pred = [data['Zk_pred'][start_truth:k_nyquist] for data in spectral_data_list]
    
    Ek_max = max([np.max(Ek) for Ek in all_Ek_target + all_Ek_pred if len(Ek) > 0])
    Ek_min = min([np.min(Ek[Ek > 0]) for Ek in all_Ek_target + all_Ek_pred if len(Ek) > 0 and np.any(Ek > 0)])
    
    Zk_max = max([np.max(Zk) for Zk in all_Zk_target + all_Zk_pred if len(Zk) > 0])
    Zk_min = min([np.min(Zk[Zk > 0]) for Zk in all_Zk_target + all_Zk_pred if len(Zk) > 0 and np.any(Zk > 0)])
    
    # Find zoom threshold index
    k_zoom_idx = np.where(k_bins[start_truth:k_nyquist] > k_zoom_threshold)[0]
    if len(k_zoom_idx) > 0:
        zoom_start_idx = k_zoom_idx[0] + start_truth
    else:
        zoom_start_idx = min(k_nyquist - 10, len(k_bins) - 10)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Full energy spectrum
    ax_energy_full = fig.add_subplot(gs[0, 0])
    ax_energy_full.set_xlabel('Wavenumber k')
    ax_energy_full.set_ylabel('Energy E(k)')
    ax_energy_full.set_title('Energy Spectrum (Full)')
    ax_energy_full.set_xscale('log')
    ax_energy_full.set_yscale('log')
    ax_energy_full.grid(True, alpha=0.3)
    line_Ek_target_full, = ax_energy_full.plot([], [], 'b-', label='Target', linewidth=2)
    line_Ek_pred_full, = ax_energy_full.plot([], [], 'r--', label='Prediction', linewidth=2)
    ax_energy_full.legend()
    
    # Zoomed energy spectrum
    ax_energy_zoom = fig.add_subplot(gs[1, 0])
    ax_energy_zoom.set_xlabel('Wavenumber k')
    ax_energy_zoom.set_ylabel('Energy E(k)')
    ax_energy_zoom.set_title(f'Energy Spectrum (Zoom: k > {k_zoom_threshold})')
    ax_energy_zoom.set_xscale('log')
    ax_energy_zoom.set_yscale('log')
    ax_energy_zoom.grid(True, alpha=0.3)
    line_Ek_target_zoom, = ax_energy_zoom.plot([], [], 'b-', label='Target', linewidth=2)
    line_Ek_pred_zoom, = ax_energy_zoom.plot([], [], 'r--', label='Prediction', linewidth=2)
    ax_energy_zoom.legend()
    
    # Full enstrophy spectrum
    ax_enstrophy_full = fig.add_subplot(gs[0, 1])
    ax_enstrophy_full.set_xlabel('Wavenumber k')
    ax_enstrophy_full.set_ylabel('Enstrophy Z(k)')
    ax_enstrophy_full.set_title('Enstrophy Spectrum (Full)')
    ax_enstrophy_full.set_xscale('log')
    ax_enstrophy_full.set_yscale('log')
    ax_enstrophy_full.grid(True, alpha=0.3)
    line_Zk_target_full, = ax_enstrophy_full.plot([], [], 'b-', label='Target', linewidth=2)
    line_Zk_pred_full, = ax_enstrophy_full.plot([], [], 'r--', label='Prediction', linewidth=2)
    ax_enstrophy_full.legend()
    
    # Zoomed enstrophy spectrum
    ax_enstrophy_zoom = fig.add_subplot(gs[1, 1])
    ax_enstrophy_zoom.set_xlabel('Wavenumber k')
    ax_enstrophy_zoom.set_ylabel('Enstrophy Z(k)')
    ax_enstrophy_zoom.set_title(f'Enstrophy Spectrum (Zoom: k > {k_zoom_threshold})')
    ax_enstrophy_zoom.set_xscale('log')
    ax_enstrophy_zoom.set_yscale('log')
    ax_enstrophy_zoom.grid(True, alpha=0.3)
    line_Zk_target_zoom, = ax_enstrophy_zoom.plot([], [], 'b-', label='Target', linewidth=2)
    line_Zk_pred_zoom, = ax_enstrophy_zoom.plot([], [], 'r--', label='Prediction', linewidth=2)
    ax_enstrophy_zoom.legend()
    
    # Step counter
    step_text = fig.suptitle('Step: 0', fontsize=16, y=0.98)
    
    # Set axis limits
    k_min_full = k_bins[start_truth]
    k_max_full = k_bins[k_nyquist - 1] if k_nyquist < len(k_bins) else k_bins[-1]
    k_min_zoom = k_bins[zoom_start_idx]
    k_max_zoom = k_bins[k_nyquist - 1] if k_nyquist < len(k_bins) else k_bins[-1]
    
    ax_energy_full.set_xlim(k_min_full, k_max_full)
    ax_energy_full.set_ylim(Ek_min, Ek_max)
    ax_energy_zoom.set_xlim(k_min_zoom, k_max_zoom)
    ax_energy_zoom.set_ylim(Ek_min, Ek_max)
    
    ax_enstrophy_full.set_xlim(k_min_full, k_max_full)
    ax_enstrophy_full.set_ylim(Zk_min, Zk_max)
    ax_enstrophy_zoom.set_xlim(k_min_zoom, k_max_zoom)
    ax_enstrophy_zoom.set_ylim(Zk_min, Zk_max)
    
    def animate(frame):
        if frame >= len(spectral_data_list):
            return []
        
        data = spectral_data_list[frame]
        k_bins = data['k_bins']
        Ek_target = data['Ek_target']
        Ek_pred = data['Ek_pred']
        Zk_target = data['Zk_target']
        Zk_pred = data['Zk_pred']
        
        # Apply truncation
        H_frame = data.get('H', H)
        Lx_frame = data.get('Lx', Lx)
        k_nyquist_frame = int((np.pi * H_frame) // Lx_frame)
        start_idx = start_truth
        end_idx = min(k_nyquist_frame, len(k_bins))
        
        k_plot = k_bins[start_idx:end_idx]
        Ek_target_plot = Ek_target[start_idx:end_idx]
        Ek_pred_plot = Ek_pred[start_idx:end_idx]
        Zk_target_plot = Zk_target[start_idx:end_idx]
        Zk_pred_plot = Zk_pred[start_idx:end_idx]
        
        # Skip zero values for log scale
        mask = k_plot > 0
        k_plot = k_plot[mask]
        Ek_target_plot = Ek_target_plot[mask]
        Ek_pred_plot = Ek_pred_plot[mask]
        Zk_target_plot = Zk_target_plot[mask]
        Zk_pred_plot = Zk_pred_plot[mask]
        
        # Remove zero energy/enstrophy values
        Ek_target_mask = Ek_target_plot > 0
        Ek_pred_mask = Ek_pred_plot > 0
        Zk_target_mask = Zk_target_plot > 0
        Zk_pred_mask = Zk_pred_plot > 0
        
        # Full spectrum
        line_Ek_target_full.set_data(k_plot[Ek_target_mask], Ek_target_plot[Ek_target_mask])
        line_Ek_pred_full.set_data(k_plot[Ek_pred_mask], Ek_pred_plot[Ek_pred_mask])
        line_Zk_target_full.set_data(k_plot[Zk_target_mask], Zk_target_plot[Zk_target_mask])
        line_Zk_pred_full.set_data(k_plot[Zk_pred_mask], Zk_pred_plot[Zk_pred_mask])
        
        # Zoomed spectrum
        zoom_mask = k_plot >= k_min_zoom
        Ek_target_zoom_mask = zoom_mask & Ek_target_mask
        Ek_pred_zoom_mask = zoom_mask & Ek_pred_mask
        Zk_target_zoom_mask = zoom_mask & Zk_target_mask
        Zk_pred_zoom_mask = zoom_mask & Zk_pred_mask
        
        line_Ek_target_zoom.set_data(k_plot[Ek_target_zoom_mask], Ek_target_plot[Ek_target_zoom_mask])
        line_Ek_pred_zoom.set_data(k_plot[Ek_pred_zoom_mask], Ek_pred_plot[Ek_pred_zoom_mask])
        line_Zk_target_zoom.set_data(k_plot[Zk_target_zoom_mask], Zk_target_plot[Zk_target_zoom_mask])
        line_Zk_pred_zoom.set_data(k_plot[Zk_pred_zoom_mask], Zk_pred_plot[Zk_pred_zoom_mask])
        
        step_text.set_text(f'Step: {data["time_step"]}')
        
        return [line_Ek_target_full, line_Ek_pred_full, line_Zk_target_full, line_Zk_pred_full,
                line_Ek_target_zoom, line_Ek_pred_zoom, line_Zk_target_zoom, line_Zk_pred_zoom,
                step_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=num_steps,
                                  interval=1000/fps, blit=False, repeat=True)
    
    # Save animation
    if save_animation and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if step_interval is not None and len(spectral_data_list) > 0:
            first_step = spectral_data_list[0].get('time_step', 0)
            last_step = spectral_data_list[-1].get('time_step', num_steps - 1)
            step_range = f"steps{first_step}-{last_step}_interval{step_interval}_frames{num_steps}"
        else:
            step_range = f"totalsteps{num_steps}"
        anim_path = f'{output_dir}/spectral_comparison_animation_{step_range}.mp4'
        print(f"Saving spectral animation to {anim_path}...")
        print(f"  Animation details: {num_steps} frames, {fps} fps, bitrate=1800")
        start_time = time.time()
        anim.save(anim_path, writer='ffmpeg', fps=fps, bitrate=1800)
        elapsed_time = time.time() - start_time
        print(f"Spectral animation saved (took {elapsed_time:.2f} seconds, ~{elapsed_time/60:.2f} minutes)")
    
    return anim, fig


def main():
    parser = argparse.ArgumentParser(description='Analyze neural operator predictions')
    parser.add_argument('--input_file', type=str, required=True, help='Path to .pth or .npz file')
    parser.add_argument('--mode', type=str, default='shapes', choices=['shapes', 'animate', 'explain'],
                        help='Mode: shapes (print shapes), animate (create animations), explain (show how metrics computed)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save animations (default: same as input file)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for animations')
    parser.add_argument('--num_animation_frames', type=int, default=None, help='Desired number of frames in animation (None = use all)')
    parser.add_argument('--k_zoom_threshold', type=float, default=20, help='Wavenumber threshold for zoomed spectral view')
    parser.add_argument('--save_animation', action='store_true', default=True, help='Save animation files')
    
    args = parser.parse_args()
    
    
    if args.input_file.endswith('.npz'):
        data = load_npz_file(args.input_file)
        
        if args.mode == 'explain':
            explain_metrics(data)
        
        elif args.mode == 'animate':
            output_dir = args.output_dir if args.output_dir else os.path.dirname(args.input_file)
            
            print("\nCreating prediction animation...")
            anim1, fig1 = animate_predictions(data, output_dir, args.save_animation, args.fps, args.num_animation_frames)
            
            if 'spectral_data_by_step' in data and len(data['spectral_data_by_step']) > 0:
                print("\nCreating spectral comparison animation...")
                anim2, fig2 = animate_spectral_comparison(data, output_dir, args.save_animation, 
                                                          args.fps, args.k_zoom_threshold, args.num_animation_frames)
            else:
                print("\nNo spectral data found. Skipping spectral animation.")
    
    else:
        raise ValueError(f"Unsupported file type. Expected .pth or .npz, got: {args.input_file}")


if __name__ == '__main__':
    """
    # 1. Load .npz file and see how metrics were computed
    python analyze_predictions_collaborator.py \
        --input_file logs/.../test_data_prediction_long.npz \
        --mode explain

    # 2. Create animations from .npz file
    python analyze_predictions_collaborator.py \
        --input_file logs/.../test_data_prediction_long.npz \
        --mode animate \
        --num_animation_frames 250 \
        --output_dir ./my_animations
    """
    main()

