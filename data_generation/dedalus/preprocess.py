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
import torch
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import logging


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


def create_animation(data_path):
    data = h5py.File(os.path.join(data_path, 'data_ns2d_T3990.h5' ), 'r')['data']
    print(data.shape)
    varlist = ['pressure', 'vx', 'vy', 'vorticity', 'streamfunction']

    # keys are (u, vx, vy), shape (T, H, W)
    cmap = 'RdBu_r'
    
    fig, ax = plt.subplots(2,3, figsize=(20, 10))
    titles = {}
    imgs = {}
    for i, key in enumerate(varlist):
        data_c = data[...,i]
        print("data channel %s shape: " % key, data_c.shape)
        vmax = np.max(np.abs(data_c))
        vmin = -vmax if np.min(data_c) <0 else np.min(data_c)
        row, col = divmod(i, 3)
        imgs[key] = ax[row, col].imshow(data_c[0,...,i], vmin=vmin, vmax=vmax, cmap=cmap) # the first time step
        ax[row, col].axis('off')
        titles[key] =ax[row, col].set_title(key + ' T=0')
    ax[1, 2].axis('off') # turn off the last axis

    def update(frame_idx):
        print("frame_idx", frame_idx)
        for i, key in enumerate(varlist):
            data_c = data[...,i]
            vmax = np.max(np.abs(data_c))
            vmin = -vmax if np.min(data_c) <0 else np.min(data_c)
            imgs[key].set_data(data_c[frame_idx,...,i])
            imgs[key].set_clim(vmin=vmin, vmax=vmax)
            titles[key].set_text(key + ' T=' + str(frame_idx))
        # Don't return anything when blit=False
        return []

    anim = FuncAnimation(fig, update, frames=data.shape[0], interval=200, blit=False)

    nt=4990
    sample_id = 0
    gif_path = f'{data_path}/sample_{sample_id}_nt{nt}.gif'
    try:
        anim.save(gif_path, writer=PillowWriter(fps=2))
    except Exception as e:
        print(f'Failed to save GIF due to: {e}')

    plt.close(fig)
    return


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

data_path = '/datasets/work/oa-tcch/work/forXuesong/snapshots/snapshots_s1'
data = load_dedalus_data(data_path)
create_animation(data_path)


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