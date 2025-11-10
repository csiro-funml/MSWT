#!/usr/bin/env python3
"""
Test script to verify that velocity fields computed from streamfunction 
match the ground truth velocity fields from the velocity form dataset.

This script:
1. Loads two datasets: one in vorticity form, one in velocity form
2. Gets the same sample from both datasets
3. Applies normalization/denormalization as in training
4. Computes velocity from streamfunction (vorticity form)
5. Extracts velocity directly (velocity form)
6. Compares the velocities and computes error metrics
7. Computes and compares energy spectra

Usage:
    python NSE/test_streamfunction_to_velocity.py --dataset ns2d_dedalus_big --sample_idx 0 --num_samples 10
    python NSE/test_streamfunction_to_velocity.py --dataset ns2d_dedalus_big --num_samples 5 --save_plots --normalize 1
    python NSE/test_streamfunction_to_velocity.py --dataset ns2d_dedalus_big --num_samples 5 --normalize 0 --Lx 6.283185307179586 --Ly 6.283185307179586
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.griddataset import MemmapDedalusBigDataset2D
from utils.compute_diagnostics import streamfunction_to_velocity, compute_spectra, compute_scalar_diagnostics


def compare_velocities(ux_from_psi, uy_from_psi, ux_gt, uy_gt, verbose=True):
    """
    Compare velocities computed from streamfunction with ground truth velocities.
    
    Args:
        ux_from_psi: x-velocity computed from streamfunction (H, W)
        uy_from_psi: y-velocity computed from streamfunction (H, W)
        ux_gt: ground truth x-velocity (H, W)
        uy_gt: ground truth y-velocity (H, W)
        verbose: whether to print detailed statistics
    
    Returns:
        dict: Dictionary containing error metrics
    """
    # Compute absolute errors
    err_ux = np.abs(ux_from_psi - ux_gt)
    err_uy = np.abs(uy_from_psi - uy_gt)
    
    # Compute relative errors (normalized by magnitude)
    ux_mag_gt = np.abs(ux_gt)
    uy_mag_gt = np.abs(uy_gt)
    rel_err_ux = err_ux / (ux_mag_gt + 1e-10)
    rel_err_uy = err_uy / (uy_mag_gt + 1e-10)
    
    # Compute L2 errors
    l2_err_ux = np.sqrt(np.mean(err_ux**2))
    l2_err_uy = np.sqrt(np.mean(err_uy**2))
    l2_err_total = np.sqrt(np.mean((ux_from_psi - ux_gt)**2 + (uy_from_psi - uy_gt)**2))
    
    # Compute relative L2 errors
    l2_norm_gt = np.sqrt(np.mean(ux_gt**2 + uy_gt**2))
    rel_l2_err = l2_err_total / (l2_norm_gt + 1e-10)
    
    # Compute max errors
    max_err_ux = np.max(err_ux)
    max_err_uy = np.max(err_uy)
    max_err_total = np.max(np.sqrt(err_ux**2 + err_uy**2))
    
    metrics = {
        'l2_err_ux': l2_err_ux,
        'l2_err_uy': l2_err_uy,
        'l2_err_total': l2_err_total,
        'rel_l2_err': rel_l2_err,
        'max_err_ux': max_err_ux,
        'max_err_uy': max_err_uy,
        'max_err_total': max_err_total,
        'mean_abs_err_ux': np.mean(err_ux),
        'mean_abs_err_uy': np.mean(err_uy),
        'mean_rel_err_ux': np.mean(rel_err_ux),
        'mean_rel_err_uy': np.mean(rel_err_uy),
    }
    
    if verbose:
        print("\n" + "="*70)
        print("Velocity Comparison Metrics")
        print("="*70)
        print(f"L2 Error (ux): {l2_err_ux:.6e}")
        print(f"L2 Error (uy): {l2_err_uy:.6e}")
        print(f"L2 Error (total): {l2_err_total:.6e}")
        print(f"Relative L2 Error: {rel_l2_err:.6e} ({rel_l2_err*100:.4f}%)")
        print(f"Max Error (ux): {max_err_ux:.6e}")
        print(f"Max Error (uy): {max_err_uy:.6e}")
        print(f"Max Error (total): {max_err_total:.6e}")
        print(f"Mean Absolute Error (ux): {np.mean(err_ux):.6e}")
        print(f"Mean Absolute Error (uy): {np.mean(err_uy):.6e}")
        print(f"Mean Relative Error (ux): {np.mean(rel_err_ux):.6e} ({np.mean(rel_err_ux)*100:.4f}%)")
        print(f"Mean Relative Error (uy): {np.mean(rel_err_uy):.6e} ({np.mean(rel_err_uy)*100:.4f}%)")
        print("="*70)
    
    return metrics


def compare_energies(ux_from_psi, uy_from_psi, ux_gt, uy_gt, Lx, Ly, verbose=True):
    """
    Compare energy computed from velocities in both forms.
    
    Args:
        ux_from_psi: x-velocity from streamfunction (H, W)
        uy_from_psi: y-velocity from streamfunction (H, W)
        ux_gt: ground truth x-velocity (H, W)
        uy_gt: ground truth y-velocity (H, W)
        Lx: Domain size in x
        Ly: Domain size in y
        verbose: whether to print detailed statistics
    
    Returns:
        dict: Dictionary containing energy comparison metrics
    """
    # Compute energy spectra
    k_bins_psi, Ek_psi, Zk_psi = compute_spectra(ux_from_psi, uy_from_psi, Lx, Ly)
    k_bins_gt, Ek_gt, Zk_gt = compute_spectra(ux_gt, uy_gt, Lx, Ly)
    
    # Compute scalar diagnostics (total energy)
    # For vorticity, we need to compute it from velocity
    # Create dummy vorticity (not used for energy calculation)
    dummy_vort = np.zeros_like(ux_from_psi)
    diag_psi = compute_scalar_diagnostics(ux_from_psi, uy_from_psi, dummy_vort, Lx, Ly)
    diag_gt = compute_scalar_diagnostics(ux_gt, uy_gt, dummy_vort, Lx, Ly)
    
    energy_psi = diag_psi['energy']
    energy_gt = diag_gt['energy']
    
    # Compute energy spectrum errors
    rel_err_spectrum = np.abs(Ek_psi - Ek_gt) / (Ek_gt + 1e-10)
    mean_rel_err_spectrum = np.mean(rel_err_spectrum[1:])  # Skip k=0
    
    # Energy difference
    energy_diff = np.abs(energy_psi - energy_gt)
    rel_energy_err = energy_diff / (energy_gt + 1e-10)
    
    metrics = {
        'energy_from_psi': energy_psi,
        'energy_gt': energy_gt,
        'energy_diff': energy_diff,
        'rel_energy_err': rel_energy_err,
        'mean_rel_spectrum_err': mean_rel_err_spectrum,
        'k_bins': k_bins_psi,
        'Ek_psi': Ek_psi,
        'Ek_gt': Ek_gt,
    }
    
    if verbose:
        print("\n" + "="*70)
        print("Energy Comparison Metrics")
        print("="*70)
        print(f"Energy (from streamfunction): {energy_psi:.6e}")
        print(f"Energy (ground truth): {energy_gt:.6e}")
        print(f"Energy Difference: {energy_diff:.6e}")
        print(f"Relative Energy Error: {rel_energy_err:.6e} ({rel_energy_err*100:.4f}%)")
        print(f"Mean Relative Spectrum Error: {mean_rel_err_spectrum:.6e} ({mean_rel_err_spectrum*100:.4f}%)")
        print("="*70)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Test streamfunction to velocity conversion')
    parser.add_argument('--dataset', type=str, default='ns2d_dedalus_big',
                       help='Dataset name')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to test')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to test')
    parser.add_argument('--Lx', type=float, default=2*np.pi,
                       help='Domain size in x direction')
    parser.add_argument('--Ly', type=float, default=2*np.pi,
                       help='Domain size in y direction')
    parser.add_argument('--normalize', type=int, default=1,
                       help='Whether to use normalization (1) or not (0)')
    parser.add_argument('--normalize_strategy', type=str, default='zscore',
                       choices=['zscore', 'minmax'],
                       help='Normalization strategy')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save comparison plots')
    parser.add_argument('--output_dir', type=str, default='logs/test_streamfunction_output',
                       help='Output directory for plots and results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Testing Streamfunction to Velocity Conversion")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Sample indices: {args.sample_idx} to {args.sample_idx + args.num_samples - 1}")
    print(f"Lx: {args.Lx}, Ly: {args.Ly}")
    print(f"Normalization: {args.normalize}, Strategy: {args.normalize_strategy}")
    print("="*70)
    
    # Load datasets
    print("\nLoading datasets...")
    dataset_vort = MemmapDedalusBigDataset2D(
        args.dataset,
        t_in=1,
        t_ar=1,
        form='vorticity',
        normalize=args.normalize,
        train='val',
        strategy=args.normalize_strategy
    )
    
    dataset_vel = MemmapDedalusBigDataset2D(
        args.dataset,
        t_in=1,
        t_ar=1,
        form='velocity',
        normalize=args.normalize,
        train='val',
        strategy=args.normalize_strategy
    )
    
    print(f"Vorticity dataset size: {len(dataset_vort)}")
    print(f"Velocity dataset size: {len(dataset_vel)}")
    
    # Verify datasets have compatible resolutions
    if dataset_vort.res != dataset_vel.res:
        raise ValueError(f"Dataset resolutions don't match: vorticity={dataset_vort.res}, velocity={dataset_vel.res}")
    
    print(f"Dataset resolution: {dataset_vort.res}")
    print(f"Vorticity form channels: {dataset_vort.n_channels_out[dataset_vort.form]}")
    print(f"Velocity form channels: {dataset_vel.n_channels_out[dataset_vel.form]}")
    
    # Store metrics for all samples
    all_velocity_metrics = []
    all_energy_metrics = []
    
    # Test multiple samples
    for i in range(args.num_samples):
        sample_idx = args.sample_idx + i
        if sample_idx >= len(dataset_vort) or sample_idx >= len(dataset_vel):
            print(f"Warning: Sample index {sample_idx} out of range. Skipping.")
            continue
        
        print(f"\n{'='*70}")
        print(f"Testing sample {sample_idx}")
        print(f"{'='*70}")
        
        # Get samples
        xx_vort, yy_vort = dataset_vort[sample_idx]  # (H, W, T_in, C), (H, W, T_ar, C)
        xx_vel, yy_vel = dataset_vel[sample_idx]     # (H, W, T_in, C), (H, W, T_ar, C)
        
        print(f"Vorticity form - Input shape: {xx_vort.shape}, Output shape: {yy_vort.shape}")
        print(f"Velocity form - Input shape: {xx_vel.shape}, Output shape: {yy_vel.shape}")
        
        # Convert to torch tensors and add batch dimension for normalization
        # Shape: (H, W, T, C) -> (1, H, W, T, C)
        xx_vort = xx_vort.unsqueeze(0)
        yy_vort = yy_vort.unsqueeze(0)
        xx_vel = xx_vel.unsqueeze(0)
        yy_vel = yy_vel.unsqueeze(0)
        
        # Apply normalization (as in training script)
        if args.normalize:
            print("\nApplying normalization...")
            xx_vort_norm = dataset_vort.normalize_x(xx_vort)
            yy_vort_norm = dataset_vort.normalize_x(yy_vort)
            xx_vel_norm = dataset_vel.normalize_x(xx_vel)
            yy_vel_norm = dataset_vel.normalize_x(yy_vel)
            
            # Denormalize (as in training script)
            print("Applying denormalization...")
            yy_vort_denorm = dataset_vort.denormalize_x(yy_vort_norm)
            yy_vel_denorm = dataset_vel.denormalize_x(yy_vel_norm)
        else:
            # No normalization: use data directly
            print("\nSkipping normalization (normalize=0)...")
            yy_vort_denorm = yy_vort
            yy_vel_denorm = yy_vel
        
        # Extract first time step and convert to numpy
        # Shape: (1, H, W, T, C) -> (H, W, C)
        target_vort = yy_vort_denorm[0, :, :, 0, :].detach().cpu().numpy()
        target_vel = yy_vel_denorm[0, :, :, 0, :].detach().cpu().numpy()
        
        print(f"\nTarget shapes after denormalization:")
        print(f"  Vorticity form: {target_vort.shape}")
        print(f"  Velocity form: {target_vel.shape}")
        
        # For vorticity form: extract streamfunction and compute velocity
        # Channel 0: vorticity, Channel 1: streamfunction
        psi_target = target_vort[:, :, 1]  # streamfunction
        print(f"\nStreamfunction stats: min={psi_target.min():.6f}, max={psi_target.max():.6f}, mean={psi_target.mean():.6f}")
        
        # Compute velocity from streamfunction
        print("Computing velocity from streamfunction...")
        ux_from_psi, uy_from_psi = streamfunction_to_velocity(psi_target, args.Lx, args.Ly)
        print(f"Velocity from streamfunction - ux: min={ux_from_psi.min():.6f}, max={ux_from_psi.max():.6f}, mean={ux_from_psi.mean():.6f}")
        print(f"Velocity from streamfunction - uy: min={uy_from_psi.min():.6f}, max={uy_from_psi.max():.6f}, mean={uy_from_psi.mean():.6f}")
        
        # For velocity form: extract velocity directly
        # Channel 0: pressure, Channel 1: velocity_x, Channel 2: velocity_y
        ux_gt = target_vel[:, :, 1]  # velocity_x
        uy_gt = target_vel[:, :, 2]  # velocity_y
        print(f"\nGround truth velocity - ux: min={ux_gt.min():.6f}, max={ux_gt.max():.6f}, mean={ux_gt.mean():.6f}")
        print(f"Ground truth velocity - uy: min={uy_gt.min():.6f}, max={uy_gt.max():.6f}, mean={uy_gt.mean():.6f}")
        
        # Compare velocities
        velocity_metrics = compare_velocities(ux_from_psi, uy_from_psi, ux_gt, uy_gt, verbose=True)
        all_velocity_metrics.append(velocity_metrics)
        
        # Compare energies
        energy_metrics = compare_energies(ux_from_psi, uy_from_psi, ux_gt, uy_gt, args.Lx, args.Ly, verbose=True)
        all_energy_metrics.append(energy_metrics)
        
        # Save plots for first sample
        if args.save_plots and i == 0:
            print("\nSaving comparison plots...")
            
            # Plot velocity fields
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # ux comparison
            im1 = axes[0, 0].imshow(ux_from_psi, cmap='RdBu', aspect='auto')
            axes[0, 0].set_title('ux from streamfunction')
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(ux_gt, cmap='RdBu', aspect='auto')
            axes[0, 1].set_title('ux ground truth')
            axes[0, 1].set_xlabel('x')
            axes[0, 1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[0, 1])
            
            im3 = axes[0, 2].imshow(np.abs(ux_from_psi - ux_gt), cmap='hot', aspect='auto')
            axes[0, 2].set_title('ux absolute error')
            axes[0, 2].set_xlabel('x')
            axes[0, 2].set_ylabel('y')
            plt.colorbar(im3, ax=axes[0, 2])
            
            # uy comparison
            im4 = axes[1, 0].imshow(uy_from_psi, cmap='RdBu', aspect='auto')
            axes[1, 0].set_title('uy from streamfunction')
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('y')
            plt.colorbar(im4, ax=axes[1, 0])
            
            im5 = axes[1, 1].imshow(uy_gt, cmap='RdBu', aspect='auto')
            axes[1, 1].set_title('uy ground truth')
            axes[1, 1].set_xlabel('x')
            axes[1, 1].set_ylabel('y')
            plt.colorbar(im5, ax=axes[1, 1])
            
            im6 = axes[1, 2].imshow(np.abs(uy_from_psi - uy_gt), cmap='hot', aspect='auto')
            axes[1, 2].set_title('uy absolute error')
            axes[1, 2].set_xlabel('x')
            axes[1, 2].set_ylabel('y')
            plt.colorbar(im6, ax=axes[1, 2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'velocity_comparison_sample_{sample_idx}.png'), dpi=150)
            plt.close()
            
            # Plot energy spectrum comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            k_bins = energy_metrics['k_bins']
            k_nyquist = len(k_bins) // 2
            start_idx = 1
            ax.loglog(k_bins[start_idx:k_nyquist], energy_metrics['Ek_psi'][start_idx:k_nyquist], 
                     'o-', markersize=3, label='From streamfunction', linewidth=1.5)
            ax.loglog(k_bins[start_idx:k_nyquist], energy_metrics['Ek_gt'][start_idx:k_nyquist], 
                     'X--', markersize=3, label='Ground truth', linewidth=1.5)
            ax.set_xlabel('Wavenumber k', fontsize=12)
            ax.set_ylabel('Energy E(k)', fontsize=12)
            ax.set_title('Energy Spectrum Comparison', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'energy_spectrum_comparison_sample_{sample_idx}.png'), dpi=150)
            plt.close()
            
            print(f"Plots saved to {args.output_dir}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("Summary Statistics Across All Samples")
    print("="*70)
    
    if len(all_velocity_metrics) > 0:
        print("\nVelocity Metrics:")
        print(f"  Mean L2 Error (total): {np.mean([m['l2_err_total'] for m in all_velocity_metrics]):.6e} ± {np.std([m['l2_err_total'] for m in all_velocity_metrics]):.6e}")
        print(f"  Mean Relative L2 Error: {np.mean([m['rel_l2_err'] for m in all_velocity_metrics]):.6e} ± {np.std([m['rel_l2_err'] for m in all_velocity_metrics]):.6e}")
        print(f"  Mean Max Error: {np.mean([m['max_err_total'] for m in all_velocity_metrics]):.6e} ± {np.std([m['max_err_total'] for m in all_velocity_metrics]):.6e}")
    
    if len(all_energy_metrics) > 0:
        print("\nEnergy Metrics:")
        print(f"  Mean Relative Energy Error: {np.mean([m['rel_energy_err'] for m in all_energy_metrics]):.6e} ± {np.std([m['rel_energy_err'] for m in all_energy_metrics]):.6e}")
        print(f"  Mean Relative Spectrum Error: {np.mean([m['mean_rel_spectrum_err'] for m in all_energy_metrics]):.6e} ± {np.std([m['mean_rel_spectrum_err'] for m in all_energy_metrics]):.6e}")
    
    print("="*70)
    print("Test completed!")


if __name__ == '__main__':
    main()

