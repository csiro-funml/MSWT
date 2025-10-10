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

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


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
    file = os.path.join(data_path, 'snapshots_s1_p0.h5')
    with h5py.File(file, "r") as f:
        vars = f.keys()
        for var in vars:
            data = np.array(f[var])
            print(f"{var}: {data.shape}")
    return data



data_path = '/datasets/work/oa-tcch/work/forXuesong/snapshots/snapshots_s1'
data = load_dedalus_data(data_path)


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