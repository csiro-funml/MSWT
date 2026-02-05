"""
Convert large NPZ files to HDF5 format for efficient loading.
This handles the high-resolution ERA5 data that's too large to load entirely into memory.
"""
import numpy as np
import h5py
from tqdm import tqdm
import os


def normalize(data, diff=False):
    data_mean = np.mean(data)
    data_std = np.std(data)
    if diff:
        data_norm = data / data_std
    else:
        data_norm = (data - data_mean) / data_std
    return data_norm, data_mean, data_std


def convert_regridded_npz_to_h5(npz_path, output_h5_path, high_res=True):
    """
    Convert regridded NPZ file to HDF5 format.
    
    Args:
        npz_path: Path to the regridded .npz file
        output_h5_path: Output path for the HDF5 file
        high_res: Whether this is high resolution data (affects variable selection)
    """
    print(f"Loading data from {npz_path}")
    data = np.load(npz_path)
    
    vars = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'precipitation', 'tisr', 'orography']
    raw_vars = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'tisr', 'orography']
    diff_vars = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure']
    diag_vars = ['precipitation']
    
    # Get data dimensions
    n_samples = len(data['temperature']) - 1
    n_vars_inp = len(vars) - 1  # 7
    n_vars_tar = len(vars) - 2  # 6
    
    # Get spatial dimensions
    if high_res:
        h, w = 512, 1024
    else:
        h, w = 48, 96
    
    print(f"Data shape: {n_samples} samples, input channels: {n_vars_inp}, target channels: {n_vars_tar}, spatial: {h}x{w}")
    
    # Create HDF5 file with compression
    with h5py.File(output_h5_path, 'w') as f:
        # Create datasets with chunking for efficient I/O
        # Chunk size: 1 sample to allow random access
        data_inp = f.create_dataset(
            'data_inp',
            shape=(n_samples, n_vars_inp, h, w),
            dtype=np.float32,
            chunks=(1, n_vars_inp, h, w),  # Chunk per sample
            compression='gzip',
            compression_opts=4
        )
        
        data_tar = f.create_dataset(
            'data_tar',
            shape=(n_samples, n_vars_tar, h, w),
            dtype=np.float32,
            chunks=(1, n_vars_tar, h, w),
            compression='gzip',
            compression_opts=4
        )
        
        # Storage for statistics
        raw_means = f.create_dataset('raw_means', shape=(len(raw_vars),), dtype=np.float32)
        raw_stds = f.create_dataset('raw_stds', shape=(len(raw_vars),), dtype=np.float32)
        diag_means = f.create_dataset('diag_means', shape=(len(diag_vars),), dtype=np.float32)
        diag_stds = f.create_dataset('diag_stds', shape=(len(diag_vars),), dtype=np.float32)
        diff_means = f.create_dataset('diff_means', shape=(len(diff_vars),), dtype=np.float32)
        diff_stds = f.create_dataset('diff_stds', shape=(len(diff_vars),), dtype=np.float32)
        
        # Compute normalization statistics first
        print("Computing normalization statistics...")
        for idx, var in tqdm(enumerate(raw_vars), total=len(raw_vars), desc="Computing stats"):
            var_data = data[var]
            raw_means[idx], raw_stds[idx] = normalize(var_data[:-1], diff=False)[1:]
            if var in diff_vars:
                diff_means[idx], diff_stds[idx] = normalize(var_data[1:] - var_data[:-1], diff=True)[1:]
        
        for idx, var in enumerate(diag_vars):
            var_data = np.log(data[var][1:] / 1e-2 + 1)
            diag_means[idx], diag_stds[idx] = normalize(var_data, diff=False)[1:]
        
        # Process and store data sample by sample to avoid memory issues
        print("Processing and storing data...")
        for idx, var in tqdm(enumerate(raw_vars), total=len(raw_vars), desc="Processing vars"):
            var_data = data[var]
            
            # Normalize input
            data_norm, _, _ = normalize(var_data[:-1], diff=False)
            data_inp[:, idx, :, :] = data_norm
            
            # Normalize target (time derivative for diff_vars)
            if var in diff_vars:
                diff_idx = diff_vars.index(var)
                diff_data = var_data[1:] - var_data[:-1]
                diff_norm, _, _ = normalize(diff_data, diff=True)
                data_tar[:, diff_idx, :, :] = diff_norm
        
        # Process diagnostic variables
        for idx, var in enumerate(diag_vars):
            var_data = np.log(data[var][1:] / 1e-2 + 1)
            diag_norm, _, _ = normalize(var_data, diff=False)
            data_tar[:, len(diff_vars), :, :] = diag_norm
    
    print(f"Conversion complete! Output saved to {output_h5_path}")
    print(f"File size: {os.path.getsize(output_h5_path) / (1024**3):.2f} GB")


def convert_preprocessed_npz_to_h5(npz_path, output_h5_path):
    """
    Convert already-preprocessed NPZ file to HDF5 format (simpler, just change format).
    Useful when preprocessed.npz already exists.
    
    Args:
        npz_path: Path to the preprocessed .npz file
        output_h5_path: Output path for the HDF5 file
    """
    print(f"Loading preprocessed data from {npz_path}")
    data = np.load(npz_path)
    
    print(f"Converting to HDF5...")
    with h5py.File(output_h5_path, 'w') as f:
        # Copy each dataset
        for key in data.keys():
            arr = data[key]
            f.create_dataset(
                key,
                data=arr,
                dtype=arr.dtype,
                chunks=(1,) + arr.shape[1:] if len(arr.shape) > 1 else None,  # Chunk first dimension
                compression='gzip',
                compression_opts=4
            )
    
    print(f"Conversion complete! Output saved to {output_h5_path}")
    print(f"File size: {os.path.getsize(output_h5_path) / (1024**3):.2f} GB")


if __name__ == '__main__':
    import sys
    
    # Determine paths
    data_folder = '/scratch3/wan410/operator_learning_data/LUCIE' if os.path.exists('/scratch3/wan410/operator_learning_data/LUCIE') else '.'
    
    # Check if we're converting high-res or low-res
    if len(sys.argv) > 1 and sys.argv[1] == 'high_res':
        high_res = True
        npz_file = os.path.join(data_folder, 'era5_512gg_1985-2004_regridded.npz')
        output_file = os.path.join(data_folder, 'era5_512gg_1985-2004_preprocessed.h5')
        preprocessed_npz = os.path.join(data_folder, 'era5_512gg_1985-2004_preprocessed.npz')
        
        # Try to use preprocessed NPZ if it exists, otherwise convert from regridded
        if os.path.exists(preprocessed_npz):
            print("Preprocessed NPZ exists, converting to HDF5...")
            convert_preprocessed_npz_to_h5(preprocessed_npz, output_file)
        else:
            print("Converting from regridded NPZ...")
            convert_regridded_npz_to_h5(npz_file, output_file, high_res=True)
    else:
        high_res = False
        npz_file = os.path.join(data_folder, 'era5_T30_regridded.npz')
        output_file = os.path.join(data_folder, 'era5_T30_preprocessed.h5')
        preprocessed_npz = os.path.join(data_folder, 'era5_T30_preprocessed.npz')
        
        if os.path.exists(preprocessed_npz):
            print("Preprocessed NPZ exists, converting to HDF5...")
            convert_preprocessed_npz_to_h5(preprocessed_npz, output_file)
        else:
            print("Converting from regridded NPZ...")
            convert_regridded_npz_to_h5(npz_file, output_file, high_res=False)
    
    print("Done!")

