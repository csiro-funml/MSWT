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
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import logging


def setup_logger(log_dir):
    """Set up logging to both file and console for Slurm jobs."""
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "preprocess.log"
    
    # Create logger
    logger = logging.getLogger('preprocess')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def merge_snapshot_files(snaps_dir):
    """
    Merge distributed snapshot files from MPI runs into a single file.

    Returns the path to the merged file, or None if merging fails.
    If only one file exists or a merged file already exists, returns that path.

    Handles two Dedalus file organization patterns:
    1. snapshots_s1.h5, snapshots_s2.h5, ... in main directory
    2. snapshots_s1/snapshots_s1_p0.h5, snapshots_s1_p1.h5, ... in subdirectories
    """
    snaps_dir = pathlib.Path(snaps_dir)

    # Check if already merged (standard name)
    merged_file = snaps_dir / "snapshots.h5"
    if merged_file.exists():
        print(f"[info] Using existing merged file: {merged_file}")
        return merged_file

    # Look for snapshot set files (snapshots_s1.h5, snapshots_s2.h5, etc.)
    snap_set_files = sorted(snaps_dir.glob("snapshots_s*.h5"))

    # If we have set files in main directory, use the first one or merge if multiple
    if len(snap_set_files) == 1:
        print(f"[info] Using existing snapshot set file: {snap_set_files[0]}")
        return snap_set_files[0]
    elif len(snap_set_files) > 1:
        print(f"[info] Found {len(snap_set_files)} snapshot set files. Using first: {snap_set_files[0]}")
        return snap_set_files[0]

    # Look for distributed process files in subdirectories
    # Pattern: snapshots_s1/snapshots_s1_p*.h5
    snap_subdirs = sorted(snaps_dir.glob("snapshots_s*"))
    snap_subdirs = [d for d in snap_subdirs if d.is_dir()]

    if snap_subdirs:
        # Use the first snapshot set subdirectory
        subdir = snap_subdirs[0]
        process_files = sorted(subdir.glob("snapshots_s*_p*.h5"))

        if len(process_files) > 1:
            print(f"[info] Found {len(process_files)} distributed process files in {subdir.name}. Merging...")
            # Extract base name (e.g., "snapshots_s1" from "snapshots_s1_p0.h5")
            base_name = process_files[0].stem.rsplit('_p', 1)[0]
            try:
                post.merge_process_files(str(subdir / base_name), cleanup=False)
                # Merged file should be created in parent directory
                merged_output = snaps_dir / f"{base_name}.h5"
                if merged_output.exists():
                    print(f"[info] Successfully merged into {merged_output}")
                    return merged_output
                else:
                    print("[warn] Merge completed but merged file not found")
                    return None
            except Exception as e:
                print(f"[error] Failed to merge snapshot files: {e}")
                return None

    print("[warn] No snapshot files found")
    return None
 
 
 # Merge distributed snapshot files if necessary


def plot_snapshot_frames(snap_h5, outdir, start, count, stride=1, dpi=300):
    """Render 3-panel frames (vorticity, pressure, streamfunction) from one snapshots HDF5 file."""
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    tasks = ["vorticity", "pressure", "streamfunction"]
    nrows, ncols = 1, 3

    scale = 2
    image = plot_tools.Box(1, 2)
    pad = plot_tools.Frame(0.20, 0.00, 0.00, 0.00)
    margin = plot_tools.Frame(0.20, 0.10, 0.00, 0.00)

    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    with h5py.File(snap_h5, "r") as f:
        times = np.array(f["scales/sim_time"])
        writes = np.array(f["scales/write_number"])

        # First pass: compute global symmetrical colorbar limits for each task
        clims = {}
        for task in tasks:
            if f"tasks/{task}" not in f:
                raise KeyError(f"Task '{task}' not found in {snap_h5}")
            dset = f[f"tasks/{task}"]
            # Find min/max over the requested range
            data_min = np.inf
            data_max = -np.inf
            for local_index in range(start, min(start + count, times.shape[0]), stride):
                data = np.array(dset[local_index, :, :])
                data_min = min(data_min, data.min())
                data_max = max(data_max, data.max())
            # Make limits symmetrical
            abs_max = max(abs(data_min), abs(data_max))
            clims[task] = (-abs_max, abs_max)

        # Second pass: plot with fixed colorbar limits
        for local_index in range(start, start + count, stride):
            if local_index >= times.shape[0]:
                break
            for n, task in enumerate(tasks):
                i, j = divmod(n, ncols)
                ax = mfig.add_axes(i, j, [0, 0, 1, 1])
                dset = f[f"tasks/{task}"]
                plot_tools.plot_bot_3d(dset, 0, local_index, axes=ax,
                                       title=task, clim=clims[task], visible_axes=False)
            tstr = f"t = {times[local_index]:.3f}"
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(tstr, x=0.45, y=title_height, ha="left")
            savename = f"write_{int(writes[local_index]):06d}.png"
            fig.savefig(str(outdir / savename), dpi=dpi)
            fig.clear()
    plt.close(fig)



def load_dedalus_data(data_path):
    filelist = [os.path.join(data_path,'snapshots_s1_p%d.h5' % i) for i in range(16)] # 16 files
    total_data = []
    var_list = ['pressure', 'velocity', 'vorticity', 'streamfunction']
    logger = logging.getLogger()
    for file in filelist:
        # use logger in case my print message is omitted in slurm
        logger.info("Loading file: %s" % file)
        with h5py.File(file, "r") as f:
            data_var = []
            for var in var_list:
                data = np.array(f['tasks'][var])
                if var == 'velocity': # velocity is a 2D field
                    data_var.append(data[:,0,:])
                    data_var.append(data[:,1,:])
                else:
                    data_var.append(data)
                logger.info(f"{var}: {data.shape}")
            data_var = np.stack(data_var, axis=0) # (5, N_t, 256, 16) # (pressure, vx, vy, vorticity, streamfunction)
            data_var = data_var.transpose(1,2,3,0) # (3990, 256, 16, 5)
            total_data.append(data_var)
    total_data = np.concatenate(total_data, axis=-2) # (N_t, 256, 256, 5)
    logger.info(total_data.shape) # (3990, 256, 256)
    # save the data to an h5 file
    with h5py.File(os.path.join('/datastore/wan410/ns2d_dedalus', 'data_ns2d_T%d.h5' % total_data.shape[0]), 'w') as f:
        f.create_dataset('data', data=total_data)
    logger.info("Data saved to: %s" % os.path.join(data_path, 'data_ns2d_T%d.pt' % total_data.shape[0]))
    return total_data


def load_or_cache_data(data_path, cache_dir=None):
    """Load data from HDF5 or use cached version if available."""
    if cache_dir is None:
        cache_dir = os.path.join(data_path, 'cached_data')
    
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if cached data exists
    cache_file = cache_dir / 'animation_data.npz'
    vmin_vmax_file = cache_dir / 'vmin_vmax.npz'
    
    logger = setup_logger(cache_dir)
    
    if cache_file.exists() and vmin_vmax_file.exists():
        logger.info("Loading cached data from disk...")
        cached_data = np.load(cache_file)
        vmin_vmax = np.load(vmin_vmax_file)
        
        # Convert back to dict format
        cached_dict = {key: cached_data[key] for key in cached_data.keys()}
        vmin_vmax_dict = {key: (vmin_vmax[key][0], vmin_vmax[key][1]) for key in vmin_vmax.keys()}
        
        logger.info(f"Loaded cached data with keys: {list(cached_dict.keys())}")
        return cached_dict, vmin_vmax_dict, logger
    
    # Load from original HDF5 and cache
    logger.info("Loading data from HDF5 and creating cache...")
    file_path = os.path.join(data_path, 'snapshots_s1.h5')            
    logger.info(f"Loading data from: {file_path}")
    
    cached_data = {}
    vmin_vmax = {}
    
    with h5py.File(file_path, 'r') as f:
        data_tasks = f['tasks']
        varlist = ['vorticity', 'streamfunction', 'pressure','velocity']
        split_var = ['velocity_x', 'velocity_y']
        
        for key in varlist:
            if key != 'velocity':
                # Force immediate loading by copying to new array
                data_c = np.array(data_tasks[key]).copy()  # .copy() ensures it's fully loaded
                cached_data[key] = data_c
                logger.info(f"Loaded {key} with shape: {data_c.shape}")
                
                # Pre-compute vmin/vmax once
                vmax = np.max(np.abs(data_c))
                vmin = -vmax if np.min(data_c) < 0 else np.min(data_c)
                vmin_vmax[key] = (vmin, vmax)
                logger.info(f"Pre-computed {key} range: [{vmin:.3f}, {vmax:.3f}]")
            else:
                # Handle velocity components
                velocity_data = np.array(data_tasks[key]).copy()  # Force immediate loading
                logger.info(f"Loaded {key} with shape: {velocity_data.shape}")
                
                for j, data_c in enumerate([velocity_data[:,0], velocity_data[:,1]]):
                    var_name = split_var[j]
                    cached_data[var_name] = data_c
                    logger.info(f"Loaded {var_name} with shape: {data_c.shape}")
                    
                    # Pre-compute vmin/vmax for each velocity component
                    vmax = np.max(np.abs(data_c))
                    vmin = -vmax if np.min(data_c) < 0 else np.min(data_c)
                    vmin_vmax[var_name] = (vmin, vmax)
                    logger.info(f"Pre-computed {var_name} range: [{vmin:.3f}, {vmax:.3f}]")
    
    # Save to cache
    logger.info("Saving data to cache...")
    np.savez_compressed(cache_file, **cached_data)
    
    # Save vmin/vmax separately for easier loading
    vmin_vmax_arrays = {key: np.array([vmin, vmax]) for key, (vmin, vmax) in vmin_vmax.items()}
    np.savez(vmin_vmax_file, **vmin_vmax_arrays)
    
    logger.info(f"Cache saved to: {cache_file}")
    logger.info(f"Vmin/Vmax saved to: {vmin_vmax_file}")
    
    return cached_data, vmin_vmax, logger


def create_individual_h5_cache(data_path, cache_dir=None):
    """Alternative: Create individual H5 files per timestep for very large datasets."""
    if cache_dir is None:
        cache_dir = os.path.join(data_path, 'timestep_cache')
    
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(cache_dir)
    
    file_path = os.path.join(data_path, 'snapshots_s1.h5')
    logger.info(f"Creating individual H5 files from: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        data_tasks = f['tasks']
        varlist = ['vorticity', 'streamfunction', 'pressure','velocity']
        
        # Get total number of timesteps
        sample_var = varlist[0] if varlist[0] != 'velocity' else varlist[1]
        total_timesteps = data_tasks[sample_var].shape[0]
        
        logger.info(f"Processing {total_timesteps} timesteps...")
        
        for t in range(total_timesteps):
            if t % 100 == 0:  # Log progress every 100 timesteps
                logger.info(f"Processing timestep {t}/{total_timesteps}")
            
            timestep_file = cache_dir / f'timestep_{t:06d}.h5'
            
            with h5py.File(timestep_file, 'w') as cache_f:
                for key in varlist:
                    if key != 'velocity':
                        data_slice = data_tasks[key][t, :, :]
                        cache_f.create_dataset(key, data=data_slice)
                    else:
                        # Store velocity components separately
                        vx = data_tasks[key][t, 0, :, :]
                        vy = data_tasks[key][t, 1, :, :]
                        cache_f.create_dataset('velocity_x', data=vx)
                        cache_f.create_dataset('velocity_y', data=vy)
    
    logger.info(f"Created {total_timesteps} individual H5 files in {cache_dir}")
    return cache_dir


def create_animation(data_path, cache_dir=None):
    # Load or create cached data
    cached_data, vmin_vmax, logger = load_or_cache_data(data_path, cache_dir)
    logger.info("Starting animation creation...")
    
    varlist = ['vorticity', 'streamfunction', 'pressure','velocity']
    split_var = ['velocity_x', 'velocity_y']
    cmap = 'RdBu_r'
    fig, ax = plt.subplots(1, 5, figsize=(50, 10))
    titles = {}
    imgs = {}
    
    # Initialize plots with cached data and pre-computed ranges
    for i, key in enumerate(varlist):
        if key != 'velocity':
            data_c = cached_data[key]
            vmin, vmax = vmin_vmax[key]
            imgs[key] = ax[i].imshow(data_c[0], vmin=vmin, vmax=vmax, cmap=cmap)
            ax[i].axis('off')
            titles[key] = ax[i].set_title(key + ' T=0')
        else:
            for j, data_c in enumerate([cached_data['velocity_x'], cached_data['velocity_y']]):
                var_name = split_var[j]
                vmin, vmax = vmin_vmax[var_name]
                imgs[var_name] = ax[i+j].imshow(data_c[0], vmin=vmin, vmax=vmax, cmap=cmap)
                ax[i+j].axis('off')
                titles[var_name] = ax[i+j].set_title(var_name + ' T=0')

    def update(frame_idx):
        logger.info(f"Updating frame {frame_idx}")
        for i, key in enumerate(varlist):
            if key != 'velocity':
                data_c = cached_data[key]  # Use cached data
                vmin, vmax = vmin_vmax[key]  # Use pre-computed range
                imgs[key].set_array(data_c[frame_idx])  # Update existing image data
                titles[key].set_text(key + ' T=' + str(frame_idx))
            else:
                for j, data_c in enumerate([cached_data['velocity_x'], cached_data['velocity_y']]):
                    var_name = split_var[j]
                    vmin, vmax = vmin_vmax[var_name]  # Use pre-computed range
                    imgs[var_name].set_array(data_c[frame_idx])  # Update existing image data
                    titles[var_name].set_text(var_name + ' T=' + str(frame_idx))
        # Don't return anything when blit=False
        return []

    logger.info("Creating animation with 200 frames...")
    anim = FuncAnimation(fig, update, frames=np.arange(0,1000,5), interval=200, blit=False)

    nt=3990
    sample_id = 0
    gif_path = f'/datastore/wan410/ns2d_dedalus/sample_{sample_id}_nt{nt}.gif'
    logger.info(f"Saving animation to: {gif_path}")
    
    try:
        anim.save(gif_path, writer=PillowWriter(fps=2))
        logger.info("Animation saved successfully!")
    except Exception as e:
        logger.error(f'Failed to save GIF due to: {e}')

    plt.close(fig)
    logger.info("Animation creation completed.")
    return



def load_real_data(data_path):                                                                                                
                                                                                                                      
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
    print("time shape", times.shape, "range", times[0], times[-1], "delta t1", times[1] - times[0], "delta tN", times[-1] - times[-2])
    for k in out_series:
        out_series[k] = out_series[k][order]
        print("out_series %s shape" % k, out_series[k].shape)
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



if __name__ == '__main__':
    data_path = '/datasets/work/oa-tcch/work/forXuesong/snapshots/'
    dirc_path = '/datasets/work/oa-tcch/work/forXuesong/data/realisation_0000/'
    ## Read the scalars (energy, enstrophy, palinstrophy)
    scalars_dir = dirc_path + 'scalars'
    out_root = "/datastore/wan410/ns2d_dedalus/data/realisation_0000/" + 'plots'
    times, series = read_scalars_all(scalars_dir)
    plot_time_series(times, series, out_root, dpi=300)

    
    # Option 1: Use compressed NPZ cache (recommended)
    # create_animation(data_path, cache_dir='/datastore/wan410/ns2d_dedalus')
    


    # plot the trend of the data

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