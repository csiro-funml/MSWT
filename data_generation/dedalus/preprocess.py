#!/datasets/work/oa-tcch/work/mgroom/.python/dedalus/bin/python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for the 2-D forced Navier–Stokes dataset.

Produces:
  1) Scalar time series plots (energy, enstrophy, palinstrophy) from scalars/*.h5
  2) Spectra plots (E(k) and Z(k)) from spectra.h5
  3) Energy transfer T(k) and cumulative energy flux Π(k) from spectra.h5
  4) Snapshot frames (vorticity, pressure, streamfunction) from snapshots/*.h5
"""

import argparse
import pathlib
import re
import h5py
import numpy as np
import os
# import torch
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import logging



def print_data_structure(data_path):                                                                                                
                                                                                                                      
    file_path = os.path.join(data_path, 'snapshots_s1.h5')                                                              
                                                                                                                        
    with h5py.File(file_path, 'r') as f:                                                                                  
        print("HDF5 File Structure:")                                                                                     
        print("=" * 60)                                                                                                   
                                                                                                                        
        def print_structure(name, obj):                                                                                   
            if isinstance(obj, h5py.Dataset):                                                                             
                print(f"\nDataset: {name}")                                                                               
                print(f"  Shape: {obj.shape}")                                                                            
                print(f"  Dtype: {obj.dtype}")                                                                            
                print(f"  Size: {obj.size} elements")                                                                     
                # Calculate size in bytes                                                                                 
                size_bytes = obj.size * obj.dtype.itemsize                                                                
                if size_bytes < 1024:                                                                                     
                    print(f"  Memory: {size_bytes} bytes")                                                                
                elif size_bytes < 1024**2:                                                                                
                    print(f"  Memory: {size_bytes/1024:.2f} KB")                                                          
                else:                                                                                                     
                    print(f"  Memory: {size_bytes/1024**2:.2f} MB")                                                       
            elif isinstance(obj, h5py.Group):                                                                             
                print(f"\nGroup: {name}")                                                                                 
                                                                                                                        
        f.visititems(print_structure)                                                                                     
                                                                                                                        
        print("\n" + "=" * 60)                                                                                            
        print("\nTop-level keys:")                                                                                        
        print(list(f.keys()))   

def downsample_data(data_path):
    data = h5py.File(os.path.join(data_path, 'data_ns2d_T3990.h5' ), 'r')['data']

    # Get original size
    T, C, H, W = u.shape
    
    # Compute FFT
    u_hat = torch.fft.rfft2(u, norm='forward')
    
    # Create frequency selection mask
    freqs_h = torch.fft.fftfreq(H, d=1/H)
    freqs_w = torch.fft.rfftfreq(W, d=1/W)
    
    # Select frequencies within [-N/2, N/2-1] range
    sel_h = torch.logical_and(freqs_h >= -N/2, freqs_h <= N/2-1)
    sel_w = torch.logical_and(freqs_w >= -N/2, freqs_w <= N/2-1)
    
    # Apply frequency selection
    u_hat_down = u_hat[:, :, sel_h][:, :, :, sel_w]
    
    # Compute inverse FFT
    u_down = torch.fft.irfft2(u_hat_down, s=(N, N), norm='forward')
    torch.save(u_down, os.path.join(data_path, 'data_ns2d_T3990_downsampled.h5'))
    return u_down


def read_scalars_all(scalars_dir):
    """Concatenate scalar time series across all HDF5 files in `scalars_dir`.

    Required scalars: energy, enstrophy, palinstrophy
    Optional scalars (if present in all files): inj, drag_loss, visc_loss
    """
    files = sorted_h5_by_write_number(sorted(pathlib.Path(scalars_dir).glob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in scalars dir: {scalars_dir}")

    required_keys = ["energy", "enstrophy", "palinstrophy"]
    optional_keys = ["inj", "drag_loss", "visc_loss"]

    times_all = []
    series_req = {k: [] for k in required_keys}
    series_opt = {k: [] for k in optional_keys}
    opt_present_all = {k: True for k in optional_keys}

    for fp in files:
        print("reading file: ", fp)
        with h5py.File(fp, "r") as f:
            t = np.array(f["scales/sim_time"])
            times_all.append(t)

            # Required scalars must exist
            for key in required_keys:
                if key not in f["tasks"]:
                    raise KeyError(f"Task '{key}' not found in {fp}")
                arr = np.array(f[f"tasks/{key}"]).squeeze()
                series_req[key].append(arr)

            # Optional scalars: include only if present in ALL files
            for key in optional_keys:
                if key in f["tasks"]:
                    arr = np.array(f[f"tasks/{key}"]).squeeze()
                    series_opt[key].append(arr)
                else:
                    opt_present_all[key] = False

    # Concatenate time and series
    times = np.concatenate(times_all)

    out_series = {}
    for k in required_keys:
        out_series[k] = np.concatenate(series_req[k])
    for k in optional_keys:
        if opt_present_all[k]:
            out_series[k] = np.concatenate(series_opt[k])

    # Sort by time consistently for all included keys
    order = np.argsort(times)
    times = times[order]
    # print("time shape", times.shape, "range", times[0], times[-1], "delta t1", times[1] - times[0], "delta tN", times[-1] - times[-2])
    for k in out_series:
        out_series[k] = out_series[k][order]
        print("out_series %s shape" % k, out_series[k].shape)
    
     # I will save the time difference to plot cuz delta t is not uniform
    delta_t = times[1:] - times[:-1]
    out_series['delta_t'] = delta_t
    return times, out_series


def sorted_h5_by_write_number(h5_paths):
    """Sort Dedalus analysis files by their 'scales/write_number'[0]."""
    def first_write_number(p):
        try:
            with h5py.File(p, "r") as f:
                wn = f["scales/write_number"][0]
            return int(wn)
        except Exception:
            m = re.search(r"(\d+)(?=\.h5$)", p.name)
            return int(m.group(1)) if m else 0
    return sorted(h5_paths, key=first_write_number)


def plot_time_series(times, series, outdir, dpi=300):
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # Time difference
    plt.figure(figsize=(7, 4))
    plt.plot(times[:-1], series["delta_t"])
    plt.xlabel("t"); plt.ylabel("delta_t"); plt.title("Time difference vs time")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(outdir / "delta_t.png", dpi=dpi); plt.close()

    # Energy
    plt.figure(figsize=(7, 4))
    plt.plot(times, series["energy"])
    plt.xlabel("t"); plt.ylabel("Energy"); plt.title("Kinetic energy vs time")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(outdir / "energy.png", dpi=dpi); plt.close()

    # Enstrophy
    plt.figure(figsize=(7, 4))
    plt.plot(times, series["enstrophy"])
    plt.xlabel("t"); plt.ylabel("Enstrophy"); plt.title("Enstrophy vs time")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(outdir / "enstrophy.png", dpi=dpi); plt.close()

    # Palinstrophy
    plt.figure(figsize=(7, 4))
    plt.plot(times, series["palinstrophy"])
    plt.xlabel("t"); plt.ylabel("Palinstrophy"); plt.title("Palinstrophy vs time")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(outdir / "palinstrophy.png", dpi=dpi); plt.close()

    # Optional new scalars, plot only if present
    if "inj" in series:
        plt.figure(figsize=(7, 4))
        plt.plot(times, series["inj"])
        plt.xlabel("t"); plt.ylabel("inj"); plt.title("Energy injection rate inj vs time")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(outdir / "inj.png", dpi=dpi); plt.close()

    if "drag_loss" in series:
        plt.figure(figsize=(7, 4))
        plt.plot(times, series["drag_loss"])
        plt.xlabel("t"); plt.ylabel("drag_loss"); plt.title("Linear drag dissipation drag_loss vs time")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(outdir / "drag_loss.png", dpi=dpi); plt.close()

    if "visc_loss" in series:
        plt.figure(figsize=(7, 4))
        plt.plot(times, series["visc_loss"])
        plt.xlabel("t"); plt.ylabel("visc_loss"); plt.title("Viscous dissipation visc_loss vs time")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(outdir / "visc_loss.png", dpi=dpi); plt.close()

    # Plot sum of inj, drag_loss, and visc_loss if all are present
    if all(k in series for k in ["inj", "drag_loss", "visc_loss"]):
        total = series["inj"] - series["drag_loss"] - series["visc_loss"]
        plt.figure(figsize=(7, 4))
        plt.plot(times, total)
        plt.xlabel("t"); plt.ylabel("inj - (drag_loss + visc_loss)")
        plt.title("Total energy balance (inj - (drag_loss + visc_loss)) vs time")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(outdir / "energy_balance_sum.png", dpi=dpi); plt.close()

def load_real_data(data_path, truncate_time=100, max_time=200, save_path=None):
    file_path = os.path.join(data_path, 'snapshots_s1.h5')                                                              
    var_names = ["scales/sim_time", "scales/timestep", "tasks/vorticity", "tasks/streamfunction", "tasks/pressure", "tasks/velocity_x", "tasks/velocity_y"] 
    var_data = {}
    truncate_idx = None
    with h5py.File(file_path, 'r') as f:                                                                                  
        for var_name in var_names:
            if var_name not in f and 'velocity' not in var_name:
                raise KeyError(f"Variable '{var_name}' not found in {file_path}")
            elif 'velocity' not in var_name:
                var_data_full = np.array(f[var_name]) # load data as numpy array (T, (C), H, W)
                if var_name == "sim_time":
                    truncate_idx = np.where(var_data[var_name] >= truncate_time and var_data[var_name] <= truncate_time + max_time)
                    print("total time steps after truncation", truncate_idx[0])
                else:
                    var_data[var_name] = var_data_full[truncate_idx]
            else:
                if var_name == 'velocity_y':
                    var_data[var_name] = np.array(f['velocity'])[:, 1, ...][truncate_idx]
                else:
                    var_data[var_name] = np.array(f['velocity'])[:, 0, ...][truncate_idx]   
            var_data[var_name] = var_data_full[truncate_idx]
    # concanate all variables as numpy array and save it to npz file
    data_np = [x for x in var_data.values()]
    data_np = np.concatenate(data_np, axis=0)  # (C, T, H, W)
    data_np = data_np.transpose(2, 3, 1, 0) # (H, W, T, C)
    print("data_np shape", data_np.shape)
    np.savez(os.path.join(save_path, 'snapshots_s1_truncated.npz'), data_np)
    
       


if __name__ == '__main__':
    dirc_path = '/datasets/work/oa-tcch/work/forXuesong/data/realisation_0000/'
    out_root = "/datastore/wan410/ns2d_dedalus/data/realisation_0000/" # I do not have access to write dirc_path
    
    ## Read the scalars (energy, enstrophy, palinstrophy)
    # scalars_dir = dirc_path + 'scalars'

    # times, series = read_scalars_all(scalars_dir)
    # plot_time_series(times, series, out_root+"plots", dpi=300)

    # Print data structure 
    print_data_structure(dirc_path + 'snapshots')
    # load 1000 steps of variables (vorticity, streamfunction, pressure, velocity, timestep) and save to h5 file
    load_real_data(dirc_path + 'snapshots', truncate_time=100, max_time=200, save_path=out_root)
                                                                                         

    # Preprocess the data
    # load data from /datasets/work/oa-tcch/work/forXuesong/data/realisation_0000/snapshots
    

    # Option 1: Use compressed NPZ cache (recommended)
    # create_animation(data_path, cache_dir='/datastore/wan410/ns2d_dedalus')
    # 

    # Option 2: Create individual H5 files per timestep (for very large datasets)
    # create_individual_h5_cache(data_path)
    
    # Option 3: Use custom cache directory
    # create_animation(data_path, cache_dir='/path/to/custom/cache')


# merged_snap = merge_snapshot_files(snaps_dir)

# if merged_snap is None:
#     print(f"[warn] No snapshot files found or merge failed in {snaps_dir}")
# else:
#     plot_snapshot_frames(
#         merged_snap,
#         outdir=out_snap,
#         start=args.snap_start,
#         count=args.snap_count,
#         stride=max(1, args.snap_stride),
#         dpi=args.dpi,
#     )