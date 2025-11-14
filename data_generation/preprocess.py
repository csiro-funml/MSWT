#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import pickle
import numpy as np
import os
import random
import scipy
import scipy.io
import h5py
from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.make_master_file import DATASET_DICT
# from data_generation.cfdbench import get_auto_dataset



def preprocess_mat():
    data = h5py.File('/home/haozhongkai/files/ml4phys/mgn/pdessl/data/ns2d/ns_V1e-3_N5000_T50.mat')
    data = np.array(data['u'])
    data = np.transpose(data, (3,1,2,0))
    train_u = data[:4800]
    test_u = data[4800:]
    print(train_u.shape, test_u.shape)
    pickle.dump(train_u, open('/home/haozhongkai/files/ml4phys/mgn/pdessl/data/ns2d/ns2d_1e-3_train.pkl','wb'))
    pickle.dump(test_u, open('/home/haozhongkai/files/ml4phys/mgn/pdessl/data/ns2d/ns2d_1e-3_test.pkl','wb'))




def save_hdf5():
    import pickle
    import h5py
    import os

    # 文件名列表
    file_names = [
        "ns2d_1e-3_test.pkl", "ns2d_1e-3_train.pkl",
        "ns2d_1e-4_test.pkl", "ns2d_1e-4_train.pkl",
        "ns2d_1e-5_test.pkl", "ns2d_1e-5_train.pkl"
    ]

    for fname in file_names:
        with open(os.path.join('/datasets/opb/pretrain',fname), 'rb') as f:
            data = pickle.load(f)

        hdf5_name = os.path.splitext(fname)[0] + '.hdf5'

        with h5py.File(os.path.join('/datasets/opb/pretrain',hdf5_name), 'w') as hf:
            hf.create_dataset('data', data=data)

    print("Conversion completed!")


### run with root
def process_pdebench_data(path='/data/pdebench/164690',save_name='/data/pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2',n_train=9000, n_test=1000):
    ### link: https://darus.uni-stuttgart.de/file.xhtml?fileId=164690&version=3.0
    ### keys: Vx, Vy, Vz, density, pressure, t-coordinate, x-coordinate, y-coordinate, z-coordinate
    if not os.path.exists(save_name):
        os.mkdir(save_name)
        os.mkdir(save_name + '/train')
        os.mkdir(save_name + '/test' )
        print('path created')
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        keys.sort()
        print(keys)
        vx = f['Vx']
        vy = f['Vy']
        # vz = f['Vz']
        density = f['density']
        pressure = f['pressure']
        t = f['t-coordinate']
        x = f['x-coordinate']
        y = f['y-coordinate']
        # z = f['z-coordinate']

        vx = np.array(vx, dtype=np.float32)
        vy = np.array(vy, dtype=np.float32)
        # vz = np.array(vz, dtype=np.float32)
        density = np.array(density, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)

        t = np.array(t, dtype=np.float32)    ###, t, x are equispaced
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        # z = np.array(z, dtype=np.float32)
        print('Content loaded:', vx.shape, vy.shape, density.shape, pressure.shape, t.shape, x.shape, y.shape)

        ## storage: x: u(t0), y: u(t1~t20), order: [B, T, X, Y ,C]
        data = np.stack([vx, vy, density, pressure],axis=-1).transpose(0,2,3,1,4)
        # X = data[:,0:1]
        # Y = data[:,1:]
        print(data.shape)   # B, X, Y, T, C
    del vx, vy,  density, pressure


    def split_data(N):

        all_ids = list(range(N))
        test_size = N // 10
        test_ids = random.sample(all_ids, test_size)
        train_ids = [id_ for id_ in all_ids if id_ not in test_ids]

        return train_ids, test_ids

    # train_ids, test_ids = split_data(10000)
    train_ids, test_ids = np.arange(int(9/10 * data.shape[0])), np.arange(int(9/10 * data.shape[0]),data.shape[0])
    print('train ids',train_ids)
    print('test ids',test_ids)

    for i in range(n_train):
        with h5py.File(save_name + '/train/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))
    for i in range(n_test):
        start = data.shape[0] - n_test
        with h5py.File(save_name + '/test/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)
            # f.create_dataset('data', data=data[start + i], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))


    print('file saved')

### Shallow water PDE
def process_swe_pdebench(path, save_name, n_train=900, n_test=100):
    ## t: 0~ 5, [101], x, y: -1~1. [128]
    os.mkdir(save_name)
    os.mkdir(save_name + '/train')
    os.mkdir(save_name + '/test')
    print('path created')
    data = []
    with h5py.File(path, 'r') as fp:
        for i in range(len(fp.keys())):
            data.append(fp["{0:0=4d}/data".format(i)])


        data = np.stack(data, axis=0).transpose(0,2,3,1,4)  # 1000,128,128,101,2
        print(data.shape)

    train_ids, test_ids = np.arange(int(n_train)), np.arange(n_train, n_train + n_test)
    print('train ids', train_ids)
    print('test ids', test_ids)

    for i in range(n_train):
        with h5py.File(save_name + '/train/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))
    for i in range(n_test):
        start = data.shape[0] - n_test
        with h5py.File(save_name + '/test/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)
            # f.create_dataset('data', data=data[start + i], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))

    print('file saved')
    return


### Diffusion Reaction PDE
def process_dr_pdebench(path, save_name, n_train=900, n_test=100):
    ## t: 0~1, [101], x, y: -2.5~2.5 [128]
    os.mkdir(save_name)
    os.mkdir(save_name + '/train')
    os.mkdir(save_name + '/test')
    print('path created')
    data = []
    with h5py.File(path, 'r') as fp:
        for i in range(len(fp.keys())):
            data.append(fp["{0:0=4d}/data".format(i)])


        data = np.stack(data, axis=0).transpose(0,2,3,1,4)  # 1000,128,128,101,2
        print(data.shape)

    train_ids, test_ids = np.arange(int(n_train)), np.arange(n_train, n_train + n_test)
    print('train ids', train_ids)
    print('test ids', test_ids)

    for i in range(n_train):
        with h5py.File(save_name + '/train/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))
    for i in range(n_test):
        start = data.shape[0] - n_test
        with h5py.File(save_name + '/test/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)
            # f.create_dataset('data', data=data[start + i], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))

    print('file saved')
    return


### run with root
def process_pdebench3d_data(path,save_name,n_train=90, n_test=10):
    ### link: https://darus.uni-stuttgart.de/file.xhtml?fileId=164693&version=3.0
    ### keys: Vx, Vy, Vz, density, pressure, t-coordinate, x-coordinate, y-coordinate, z-coordinate
    if not os.path.exists(save_name):
        os.mkdir(save_name)
    os.mkdir(save_name + '/train')
    os.mkdir(save_name + '/test' )
    print('path created')
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        keys.sort()
        print(keys)
        vx = f['Vx']
        vy = f['Vy']
        vz = f['Vz']
        density = f['density']
        pressure = f['pressure']
        t = f['t-coordinate']
        x = f['x-coordinate']
        y = f['y-coordinate']
        z = f['z-coordinate']

        vx = np.array(vx, dtype=np.float32)
        vy = np.array(vy, dtype=np.float32)
        vz = np.array(vz, dtype=np.float32)
        density = np.array(density, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)

        t = np.array(t, dtype=np.float32)    ###, t, x are equispaced
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        z = np.array(z, dtype=np.float32)
        print('Content loaded:', vx.shape, density.shape, pressure.shape, t.shape, x.shape, y.shape)

        ## storage: x: u(t0), y: u(t1~t20), order: [B, T, X, Y, Z ,C]
        data = np.stack([vx, vy, vz, pressure, density],axis=-1).transpose(0,2,3,4,1,5)
        # X = data[:,0:1]
        # Y = data[:,1:]
        print(data.shape)   # B, X, Y, T, C
    del vx, vy,  density, pressure

    def split_data(N):

        all_ids = list(range(N))
        test_size = N // 10
        test_ids = random.sample(all_ids, test_size)
        train_ids = [id_ for id_ in all_ids if id_ not in test_ids]

        return train_ids, test_ids

    # train_ids, test_ids = split_data(10000)
    train_ids, test_ids = np.arange(int(9/10 * data.shape[0])), np.arange(int(9/10 * data.shape[0]),data.shape[0])
    print('train ids',train_ids)
    print('test ids',test_ids)

    for i in range(n_train):
        with h5py.File(save_name + '/train/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))
    for i in range(n_test):
        start = data.shape[0] - n_test
        with h5py.File(save_name + '/test/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)
            # f.create_dataset('data', data=data[start + i], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))


    print('file saved')




def preprocess_ns2d(load_path='data/large/pdearena/NavierStokes-2D',
                    save_path='data/large/pdearena/ns2d_pda'):
    """
    Preprocess the Navier-Stokes 2D dataset from PDEArena

    there are 3 channels in the dataset:
        u, vx, vy
    data shape: (N, 128, 128, 14, 3)
    """
    LOAD_PATH = load_path
    SAVE_PATH_TEST = save_path + '/test'
    SAVE_PATH_TRAIN = save_path + '/train'
    SAVE_PATH_VAL = save_path + '/val'

    # Create new folders if SAVE_PATH does not exist
    os.makedirs(SAVE_PATH_TEST, exist_ok=True)
    os.makedirs(SAVE_PATH_TRAIN, exist_ok=True)
    os.makedirs(SAVE_PATH_VAL, exist_ok=True)
    test_tot = 0
    train_tot = 0
    val_tot = 0
    # Traverse the file in LOAD_PATH
    for root, dirs, files in os.walk(LOAD_PATH):
        for file in tqdm(files):
            # Skip the file if it is not a HDF5 file
            if not file.endswith('.h5'):
                continue
            # Open the file
            try:
                with h5py.File(os.path.join(root, file), 'r') as f:
                    if 'test' in file:
                        key = 'test'
                        path = SAVE_PATH_TEST
                    elif 'train' in file:
                        key = 'train'
                        path = SAVE_PATH_TRAIN
                    elif 'valid' in file:
                        key = 'valid'
                        path = SAVE_PATH_VAL
                    else:
                        raise ValueError('Unknown file type {}!'.format(file))

                    u = f[key]['u'][:]
                    vx = f[key]['vx'][:]
                    vy = f[key]['vy'][:]

                    out = np.stack([u, vx, vy], axis=-1)
                    out = np.transpose(out, (0, 2, 3, 1, 4))

                    # Create the destination file
                    for data in out:
                        if key == 'test':
                            idx = test_tot
                            test_tot += 1
                        elif key == 'valid':
                            idx = val_tot
                            val_tot += 1
                        else:
                            idx = train_tot
                            train_tot += 1
                        dst_file = 'data_{}.hdf5'.format(idx)
                        save_path = os.path.join(path, dst_file)
                        with h5py.File(save_path, 'w') as g:
                            # Write data as a hdf5 dataset
                            # with key 'data'
                            g.create_dataset('data', data=data)
            except Exception as e:
                print('Error in file {}: {}'.format(file, e))
                continue



def preprocess_ns2d_longrollout(load_path='data/large/pdearena/NavierStokes-2D',
                    save_path='data/large/pdearena/ns2d_pda'):
    """
    Preprocess the Navier-Stokes 2D dataset from PDEArena

    there are 3 channels in the dataset:
        u, vx, vy
    data shape: (N, 128, 128, 14, 3)
    """
    if not torch.cuda.is_available():
        load_path = '/Users/wan410/Documents/VSCode/pdearena/pdearena_data/navierstokes/'
        save_data_path = 'pdearena/ns2d_pda/test_long/'
    else:
        load_path = '/home/wan410/pdearena/pdearena_data/navierstokes'
        save_data_path = '/scratch3/wan410/operator_learning_data/pdearena/ns2d_pda/test_long/'
 
    # file = 'NavierStokes2D_test_300_0.50000.h5'
    file = 'NavierStokes2D_test_198210_0.50000.h5'
    # SAVE_PATH_VAL = save_path + 'test_long/'

    # Create new folders if SAVE_PATH does not exist
    # os.makedirs(SAVE_PATH_VAL, exist_ok=True)
    test_tot = 0
    train_tot = 0
    val_tot = 0
    # Traverse the file in LOAD_PATH

    # for file in tqdm([load_path]):
        # Skip the file if it is not a HDF5 file
        # if not file.endswith('.h5'):
            # continue
    # Open the file
    try:
        with h5py.File(os.path.join(load_path, file), 'r') as f:
            key = 'test' 

            u = f[key]['u'][:]
            vx = f[key]['vx'][:]
            vy = f[key]['vy'][:]

            out = np.stack([u, vx, vy], axis=-1)
            out = np.transpose(out, (0, 2, 3, 1, 4))

            # Create the destination file
            for data in out:
                if key == 'test':
                    idx = test_tot
                    test_tot += 1
                elif key == 'valid':
                    idx = val_tot
                    val_tot += 1
                else:
                    idx = train_tot
                    train_tot += 1
                dst_file = 'data_{}.hdf5'.format(idx)
                save_path = os.path.join(save_data_path, dst_file)
                with h5py.File(save_path, 'w') as g:
                    # Write data as a hdf5 dataset
                    # with key 'data'
                    g.create_dataset('data', data=data)
    except Exception as e:
        print('Error in file {}: {}'.format(file, e))
        # continue


def _read_slice_from_file(args):
    """Helper function to read a single slice from a file (for multiprocessing)."""
    file_path, t = args
    with h5py.File(file_path, 'r') as f:
        vorticity_slice = np.array(f['tasks/vorticity'][t], dtype=np.float32)
        stream_slice = np.array(f['tasks/streamfunction'][t], dtype=np.float32)
        velocity_x_slice = np.array(f['tasks/velocity'][t, 0], dtype=np.float32)
        velocity_y_slice = np.array(f['tasks/velocity'][t, 1], dtype=np.float32)
        pressure_slice = np.array(f['tasks/pressure'][t], dtype=np.float32)
        forcing_x_slice = np.array(f['tasks/forcing'][t, 0], dtype=np.float32)
        forcing_y_slice = np.array(f['tasks/forcing'][t, 1], dtype=np.float32)
    return (vorticity_slice, stream_slice, velocity_x_slice, velocity_y_slice, pressure_slice, forcing_x_slice, forcing_y_slice)


def preprocess_dedalus_to_shards(dataset_name='ns2d_dedalus', save_dir='./data/large/dedalus_memmap', 
                        start_time_id=100, end_time_id=500, shard_size=2048, dtype='float16', state='train', num_workers=None):
    """
    Preprocess Dedalus virtual dataset (p0–p15 spatial slices) into memmap shards.

    Output layout:
        save_dir/
          - shards/shard_00000.dat, shard_00001.dat, ... (memmap files, shape (T_shard, C, H, W))
          - meta.json (schema, shapes, dtype, splits, normalizer)

    Assumptions:
      - Source at DATASET_DICT[dataset_name]['data_path'] points to snapshots_s1.h5 (virtual dataset root)
      - 16 slice files exist alongside it: snapshots_s1_p00.h5 ... snapshots_s1_p15.h5
      - Channels: vorticity, streamfunction → C=2
    
    Args:
        num_workers: Number of parallel workers for reading files. If None, uses cpu_count().
    """
    
    if num_workers is None:
        # Check for SLURM CPU allocation first, then NUM_WORKERS env var, then use all CPUs
        num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 
                                         os.environ.get('NUM_WORKERS', 
                                                       cpu_count())))
    print(f"Using {num_workers} workers for parallel file reading")

    os.makedirs(save_dir, exist_ok=True)
    shards_dir = os.path.join(save_dir, 'shards')
    os.makedirs(shards_dir, exist_ok=True)
    print("keys", DATASET_DICT[dataset_name].keys())
    data_path = DATASET_DICT[dataset_name][state + '_raw_path'] # it is a list of folders with the data
    
    # Handle both list and single path for backward compatibility
    if isinstance(data_path, str):
        realization_paths = [data_path]
    else:
        realization_paths = data_path
    
    # Calculate total timesteps for global progress bar
    total_timesteps = 0
    for realization_path in realization_paths:
        base_path = realization_path.replace('.h5', '')
        first_file = f"{base_path}/snapshots_s1_p0.h5"
        if os.path.exists(first_file):
            with h5py.File(first_file, 'r') as f:
                T_realization, _, _ = f['tasks/vorticity'].shape
                total_timesteps += T_realization
    
    # Configure dtype
    np_dtype = np.float16 if str(dtype) == 'float16' else np.float32

    # Streaming mean/std computation using Welford's online algorithm
    # Store running statistics: {channel: {'mean': float, 'M2': float, 'count': int}}
    ch_stats = {
        'vorticity': {'mean': 0.0, 'M2': 0.0, 'count': 0},
        'streamfunction': {'mean': 0.0, 'M2': 0.0, 'count': 0},
        'velocity_x': {'mean': 0.0, 'M2': 0.0, 'count': 0},
        'velocity_y': {'mean': 0.0, 'M2': 0.0, 'count': 0},
        'pressure': {'mean': 0.0, 'M2': 0.0, 'count': 0},
        'forcing_x': {'mean': 0.0, 'M2': 0.0, 'count': 0},
        'forcing_y': {'mean': 0.0, 'M2': 0.0, 'count': 0}
    }
    
    def update_stats(channel_name, data_flat):
        """Update statistics using Welford's online algorithm (vectorized batch update)."""
        stats = ch_stats[channel_name]
        n_new = len(data_flat)
        
        if stats['count'] == 0:
            # First batch
            stats['mean'] = float(np.mean(data_flat))
            stats['M2'] = float(np.sum((data_flat - stats['mean']) ** 2))
            stats['count'] = n_new
        else:
            # Update with new batch
            old_mean = stats['mean']
            new_mean = float(np.mean(data_flat))
            total_count = stats['count'] + n_new
            
            # Update mean
            stats['mean'] = (stats['count'] * old_mean + n_new * new_mean) / total_count
            
            # Update M2 (sum of squared differences)
            M2_new = float(np.sum((data_flat - new_mean) ** 2))
            correction = stats['count'] * n_new * ((old_mean - new_mean) ** 2) / total_count
            stats['M2'] = stats['M2'] + M2_new + correction
            
            stats['count'] = total_count

    # Initialize shard variables (shared across all realizations)
    shard_index = 0
    t_written_in_shard = 0
    shard_files = []
    H = None
    W_full = None
    T_total_all = 0  # Total timesteps across all realizations
    shard_path = None
    shard_mm = None
    curr_shard_len = None

    # Allocate shards on demand
    def open_shard(shard_index, shard_len):
        fname = os.path.join(shards_dir, f'shard_{shard_index:05d}.dat')
        mm = np.memmap(fname, dtype=np_dtype, mode='w+', shape=(shard_len, 7, H, W_full))
        return fname, mm

    # Create global progress bar (shared across all realizations)
    print(f"Total timesteps to process: {total_timesteps}")
    global_pbar = tqdm(total=total_timesteps, desc='Overall progress', unit='timestep', position=0, leave=True)
    start_time = time.time()

    # Loop over all realizations
    for realization_idx, realization_path in enumerate(realization_paths):
        base_path = realization_path.replace('.h5', '')
        file_paths = [f"{base_path}/snapshots_s1_p{i:d}.h5" for i in range(16)]

        # Probe metadata (first time only, or verify consistency)
        with h5py.File(file_paths[0], 'r') as f0:
            T_realization, H_realization, W_partial = f0['tasks/vorticity'].shape
            W_full_realization = W_partial * 16
            
            if H is None:
                # First realization: initialize dimensions
                H = H_realization
                W_full = W_full_realization
                # Initialize first shard
                curr_shard_len = min(shard_size, T_realization)
                shard_path, shard_mm = open_shard(shard_index, curr_shard_len)
            else:
                # Verify dimensions are consistent
                assert H == H_realization, f"Height mismatch: {H} vs {H_realization}"
                assert W_full == W_full_realization, f"Width mismatch: {W_full} vs {W_full_realization}"

        print(f"Processing realization {realization_idx + 1}/{len(realization_paths)}: {realization_path} ({T_realization} timesteps)")
        
        # Iterate over all timesteps in this realization and write contiguously
        if num_workers > 1:
            # Use multiprocessing to read files in parallel
            with Pool(processes=num_workers) as pool:
                for t in range(start_time_id,T_realization):
                    # Prepare arguments for parallel reading: (file_path, timestep) for each of 16 files
                    read_args = [(fp, t) for fp in file_paths]
                    
                    # Read all 16 slices in parallel
                    results = pool.map(_read_slice_from_file, read_args)
                    
                    # Unpack results and organize by channel
                    vorticity_slices = [r[0] for r in results]
                    stream_slices = [r[1] for r in results]
                    velocity_x_slices = [r[2] for r in results]
                    velocity_y_slices = [r[3] for r in results]
                    pressure_slices = [r[4] for r in results]
                    forcing_x_slices = [r[5] for r in results]
                    forcing_y_slices = [r[6] for r in results]
                    
                    # Process this timestep (concatenate, compute stats, write to shard)
                    vorticity_full = np.concatenate(vorticity_slices, axis=1)  # (H, W)
                    stream_full = np.concatenate(stream_slices, axis=1)        # (H, W)
                    velocity_x_full = np.concatenate(velocity_x_slices, axis=1)  # (H, W)
                    velocity_y_full = np.concatenate(velocity_y_slices, axis=1)  # (H, W)
                    pressure_full = np.concatenate(pressure_slices, axis=1)      # (H, W)
                    forcing_x_full = np.concatenate(forcing_x_slices, axis=1)   # (H, W)
                    forcing_y_full = np.concatenate(forcing_y_slices, axis=1)    # (H, W)
                    
                    # Update stats across all timesteps and spatial points using online algorithm
                    update_stats('vorticity', vorticity_full.flatten())
                    update_stats('streamfunction', stream_full.flatten())
                    update_stats('velocity_x', velocity_x_full.flatten())
                    update_stats('velocity_y', velocity_y_full.flatten())
                    update_stats('pressure', pressure_full.flatten())
                    update_stats('forcing_x', forcing_x_full.flatten())
                    update_stats('forcing_y', forcing_y_full.flatten())

                    # Write to shard (T, C, H, W)
                    shard_mm[t_written_in_shard, 0] = vorticity_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 1] = stream_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 2] = velocity_x_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 3] = velocity_y_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 4] = pressure_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 5] = forcing_x_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 6] = forcing_y_full.astype(np_dtype, copy=False)
                    t_written_in_shard += 1
                    T_total_all += 1
                    
                    # Update global progress bar
                    global_pbar.update(1)

                    # Rotate shard if full
                    if t_written_in_shard == curr_shard_len:
                        # finalize current shard
                        shard_mm.flush()
                        shard_files.append({'file': os.path.relpath(shard_path, start=save_dir), 'length': int(curr_shard_len)})
                        # Check if there are more timesteps to process
                        is_last_timestep = (realization_idx == len(realization_paths) - 1) and (t == T_realization - 1)
                        if not is_last_timestep:
                            # open next shard
                            shard_index += 1
                            # Calculate remaining timesteps if we're in the last realization
                            if realization_idx == len(realization_paths) - 1:
                                # Last realization: calculate remaining timesteps
                                remaining_in_realization = T_realization - (t + 1)
                                curr_shard_len = min(shard_size, remaining_in_realization)
                            else:
                                # Not last realization: use full shard_size
                                curr_shard_len = shard_size
                            shard_path, shard_mm = open_shard(shard_index, curr_shard_len)
                            t_written_in_shard = 0
        else:
            # Sequential reading (no multiprocessing overhead)
            h5_files = [h5py.File(fp, 'r') for fp in file_paths]
            try:
                for t in range(T_realization):
                    vorticity_slices = []
                    stream_slices = []
                    velocity_x_slices = []
                    velocity_y_slices = []
                    pressure_slices = []
                    forcing_x_slices = []
                    forcing_y_slices = []
                    for f in h5_files:
                        vorticity_slice = np.array(f['tasks/vorticity'][t], dtype=np.float32)
                        stream_slice = np.array(f['tasks/streamfunction'][t], dtype=np.float32)
                        velocity_x_slice = np.array(f['tasks/velocity'][t, 0], dtype=np.float32)
                        velocity_y_slice = np.array(f['tasks/velocity'][t, 1], dtype=np.float32)
                        pressure_slice = np.array(f['tasks/pressure'][t], dtype=np.float32)
                        forcing_x_slice = np.array(f['tasks/forcing'][t, 0], dtype=np.float32)
                        forcing_y_slice = np.array(f['tasks/forcing'][t, 1], dtype=np.float32)
                        vorticity_slices.append(vorticity_slice)
                        stream_slices.append(stream_slice)
                        velocity_x_slices.append(velocity_x_slice)
                        velocity_y_slices.append(velocity_y_slice)
                        pressure_slices.append(pressure_slice)
                        forcing_x_slices.append(forcing_x_slice)
                        forcing_y_slices.append(forcing_y_slice)
                    
                    # Process this timestep (concatenate, compute stats, write to shard)
                    vorticity_full = np.concatenate(vorticity_slices, axis=1)  # (H, W)
                    stream_full = np.concatenate(stream_slices, axis=1)        # (H, W)
                    velocity_x_full = np.concatenate(velocity_x_slices, axis=1)  # (H, W)
                    velocity_y_full = np.concatenate(velocity_y_slices, axis=1)  # (H, W)
                    pressure_full = np.concatenate(pressure_slices, axis=1)      # (H, W)
                    forcing_x_full = np.concatenate(forcing_x_slices, axis=1)   # (H, W)
                    forcing_y_full = np.concatenate(forcing_y_slices, axis=1)    # (H, W)
                    
                    # Update stats across all timesteps and spatial points using online algorithm
                    update_stats('vorticity', vorticity_full.flatten())
                    update_stats('streamfunction', stream_full.flatten())
                    update_stats('velocity_x', velocity_x_full.flatten())
                    update_stats('velocity_y', velocity_y_full.flatten())
                    update_stats('pressure', pressure_full.flatten())
                    update_stats('forcing_x', forcing_x_full.flatten())
                    update_stats('forcing_y', forcing_y_full.flatten())

                    # Write to shard (T, C, H, W)
                    shard_mm[t_written_in_shard, 0] = vorticity_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 1] = stream_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 2] = velocity_x_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 3] = velocity_y_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 4] = pressure_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 5] = forcing_x_full.astype(np_dtype, copy=False)
                    shard_mm[t_written_in_shard, 6] = forcing_y_full.astype(np_dtype, copy=False)
                    t_written_in_shard += 1
                    T_total_all += 1
                    
                    # Update global progress bar
                    global_pbar.update(1)

                    # Rotate shard if full
                    if t_written_in_shard == curr_shard_len:
                        # finalize current shard
                        shard_mm.flush()
                        shard_files.append({'file': os.path.relpath(shard_path, start=save_dir), 'length': int(curr_shard_len)})
                        # Check if there are more timesteps to process
                        is_last_timestep = (realization_idx == len(realization_paths) - 1) and (t == T_realization - 1)
                        if not is_last_timestep:
                            # open next shard
                            shard_index += 1
                            # Calculate remaining timesteps if we're in the last realization
                            if realization_idx == len(realization_paths) - 1:
                                # Last realization: calculate remaining timesteps
                                remaining_in_realization = T_realization - (t + 1)
                                curr_shard_len = min(shard_size, remaining_in_realization)
                            else:
                                # Not last realization: use full shard_size
                                curr_shard_len = shard_size
                            shard_path, shard_mm = open_shard(shard_index, curr_shard_len)
                            t_written_in_shard = 0
            finally:
                for f in h5_files:
                    f.close()

    # Close global progress bar
    global_pbar.close()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted processing {T_total_all} timesteps in {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
    print(f"Average speed: {T_total_all/elapsed_time:.2f} timesteps/second")

    # finalize last shard
    if shard_mm is not None and t_written_in_shard > 0:
        shard_mm.flush()
        shard_files.append({'file': os.path.relpath(shard_path, start=save_dir), 'length': int(t_written_in_shard)})

    # Compute final statistics from accumulated stats
    final_stats = {}
    for channel_name in ch_stats.keys():
        stats = ch_stats[channel_name]
        mean = stats['mean']
        if stats['count'] > 1:
            std = np.sqrt(stats['M2'] / stats['count'])
        else:
            std = 0.0
        final_stats[channel_name] = {'mean': float(mean), 'std': float(std)}
        print(f"Final stats for {channel_name}: mean={mean:.6e}, std={std:.6e}, count={stats['count']}")
    
    # Write meta.json
    meta = {
        'source': data_path,
        'dtype': str(dtype),
        'axis_order': 'TCHW',
        'shape': {'T': int(T_total_all), 'C': len(ch_stats.keys()), 'H': int(H), 'W': int(W_full)},
        'shard_size': int(shard_size),
        'num_shards': int(len(shard_files)),
        'shards': shard_files,
        'normalizer': {
            'type': 'zscore',
            'vorticity': {
                'mean': final_stats['vorticity']['mean'],
                'std': final_stats['vorticity']['std']
            },
            'streamfunction': {
                'mean': final_stats['streamfunction']['mean'],
                'std': final_stats['streamfunction']['std']
            },
            'velocity_x': {
                'mean': final_stats['velocity_x']['mean'],
                'std': final_stats['velocity_x']['std']
            },
            'velocity_y': {
                'mean': final_stats['velocity_y']['mean'],
                'std': final_stats['velocity_y']['std']
            },
            'pressure': {
                'mean': final_stats['pressure']['mean'],
                'std': final_stats['pressure']['std']
            },
            'forcing_x': {
                'mean': final_stats['forcing_x']['mean'],
                'std': final_stats['forcing_x']['std']
            },
            'forcing_y': {
                'mean': final_stats['forcing_y']['mean'],
                'std': final_stats['forcing_y']['std']
            }
            }
    }

    # attach splits if present
    for key in ['train_range', 'val_range', 'test_range', 'temporal_downsample', 'downsample', 't_in']:
        if key in DATASET_DICT[dataset_name]:
            meta[key] = DATASET_DICT[dataset_name][key]

    with open(os.path.join(save_dir, 'meta.json'), 'w') as fp:
        json.dump(meta, fp, indent=2)

    print('Memmap shards written to', save_dir)

def preprocess_torchcfd_ns2d(load_path, save_path, total_time=100):
    """
    Preprocess the Navier-Stokes 2D dataset from torch-cfd
    there are 2 channels in the dataset:
        vorticity, stream
    data shape: (N, 128, 128, 100, 2)
    """
    LOAD_PATH = load_path
    SAVE_PATH_TEST = save_path + '/test'
    SAVE_PATH_TRAIN = save_path + '/train'
    SAVE_PATH_VAL = save_path + '/val'

    # Create new folders if SAVE_PATH does not exist
    os.makedirs(SAVE_PATH_TEST, exist_ok=True)
    os.makedirs(SAVE_PATH_TRAIN, exist_ok=True)
    os.makedirs(SAVE_PATH_VAL, exist_ok=True)
    test_tot = 0
    train_tot = 0
    val_tot = 0
    # load path for train, test, val
    Re = 'Re1000' if 'Re1000' in save_path else 'Re5000'
    if Re == 'Re1000':
        train_path  = os.path.join(LOAD_PATH, 'McWilliams2d_128x128_N5000_{}_T100.pt'.format(Re))
        test_path = os.path.join(LOAD_PATH, 'McWilliams2d_128x128_N500_{}_T100.pt'.format(Re))
        val_path = os.path.join(LOAD_PATH, 'McWilliams2d_128x128_N256_{}_T100.pt'.format(Re))
    else:
        train_path  = os.path.join(LOAD_PATH, 'McWilliams2d_128x128_N1000_{}_T30.pt'.format(Re))
        test_path = os.path.join(LOAD_PATH, 'McWilliams2d_128x128_N512_{}_T30.pt'.format(Re))
        val_path = os.path.join(LOAD_PATH, 'McWilliams2d_128x128_N256_{}_T30.pt'.format(Re))
    # train_path = os.path.join(LOAD_PATH, 'McWilliams2d_128x128_N5000_{}_T30.pt'.format(Re))
    # test_path = os.path.join(LOAD_PATH, 'McWilliams2d_128x128_N500_{}_T30.pt'.format(Re))
    # val_path = os.path.join(LOAD_PATH, 'McWilliams2d_128x128_N256_{}_T30.pt'.format(Re))
    try:
        # for path in [train_path, test_path, val_path]:
        for path in [test_path]:
            data = torch.load(path)
        
            vorticity = data['vorticity']
            stream = data['stream']

            out = np.stack([vorticity, stream], axis=-1).squeeze(1) # (N, T, H, W, 2)
            out = np.transpose(out, (0, 2, 3, 1, 4)) # (N, H, W, T, 2)
            if total_time < out.shape[-2]: # subsampling the time
                step = out.shape[-2] // total_time
                print("subsampling step", step)
                out = out[:,:,:,::step,:]
                print("subsampled out.shape", out.shape)
            # Create the destination file
            key = 'train'
            if path == train_path:
                key = 'train'
            elif path == test_path:
                key = 'test'
            elif path == val_path:
                key = 'valid'

            for idx, data in tqdm(enumerate(out)):
                if key == 'test':
                    test_tot += 1
                    path = SAVE_PATH_TEST
                elif key == 'valid':
                    val_tot += 1
                    path = SAVE_PATH_VAL
                else:
                    path = SAVE_PATH_TRAIN
                    train_tot += 1
                dst_file = 'data_{}.hdf5'.format(idx)
                save_path = os.path.join(path, dst_file)
                with h5py.File(save_path, 'w') as g:
                    # Write data as a hdf5 dataset
                    # with key 'data'
                    g.create_dataset('data', data=data)
    except Exception as e:
        print('Error in file {}: {}'.format(path, e))
    print('file saved')



def preprocess_ns2d_cond():
    """
    Preprocess the Navier-Stokes 2D conditioned
        dataset from PDEArena

    there are 3 channels in the dataset:
        u, vx, vy
    data shape: (N, 128, 128, 56, 3)
    """
    preprocess_ns2d(
        load_path='data/large/pdearena/NavierStokes-2D-conditoned',
        save_path='data/large/pdearena/ns2d_cond_pda'
    )


def preprocess_shallow_water(load_path, save_path):
    """
    Preprocess the Shallow Water dataset from PDEArena

    there are 5 channels in the dataset:
        u, v, div, vor, pres
    data shape: (N, 96, 192, 88, 5)
    """
    LOAD_PATH = load_path
    SAVE_PATH_TEST = save_path + '/test'
    SAVE_PATH_TRAIN = save_path + '/train'

    # Create new folders if SAVE_PATH does not exist
    os.makedirs(SAVE_PATH_TEST, exist_ok=True)
    os.makedirs(SAVE_PATH_TRAIN, exist_ok=True)

    test_tot = 0
    train_tot = 0

    # Traverse the file in LOAD_PATH
    for root, dirs, files in tqdm(os.walk(LOAD_PATH)):
        print("dir", dirs)
        for file in files:
            # Skip the file if it is not a HDF5 file
            if not file.endswith('.nc'):
                continue
            # Open the file
            try:
                with h5py.File(os.path.join(root, file), 'r') as f:
                    if 'test' in root:
                        key = 'test'
                        path = SAVE_PATH_TEST
                    elif 'train' in root:
                        key = 'train'
                        path = SAVE_PATH_TRAIN
                    elif 'valid' in root:
                        key = 'valid'
                        path = SAVE_PATH_TRAIN
                    else:
                        raise ValueError('Unknown file type {}!'.format(file))

                    u = f['u'][:]
                    u = u[:, 0, ...]
                    v = f['v'][:]
                    v = v[:, 0, ...]
                    div = f['div'][:]
                    div = div[:, 0, ...]
                    vor = f['vor'][:]
                    vor = vor[:, 0, ...]
                    pres = f['pres'][:]

                    data = np.stack([u, v, div, vor, pres], axis=-1)
                    data = np.transpose(data, (1, 2, 0, 3))

                    # Create the destination file
                    if key == 'test':
                        idx = test_tot
                        test_tot += 1
                    else:
                        idx = train_tot
                        train_tot += 1
                    dst_file = 'data_{}.hdf5'.format(idx)
                    save_path = os.path.join(path, dst_file)
                    with h5py.File(save_path, 'w') as g:
                        # Write data as a hdf5 dataset
                        # with key 'data'
                        g.create_dataset('data', data=data)
            except Exception as e:
                print('Error in file {}: {}'.format(file, e))
                continue




def preprocess_cfdbench_data():
    delta_cavity = 0.1
    delta_cylinder = 0.1
    delta_tube = 0.1
    delta_dam = 0.1
    train_data_cavity, dev_data_cavity, test_data_cavity = get_auto_dataset(
        data_dir=Path('../../data/large/cfdbench'),
        data_name='cavity_prop_bc_geo',
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
    )
    train_data_cylinder, dev_data_cylinder, test_data_cylinder = get_auto_dataset(
        data_dir=Path('../../data/large/cfdbench'),
        data_name='cylinder_prop_bc_geo',
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
    )
    # train_data_dam, dev_data_dam, test_data_dam = get_auto_dataset(
    #     data_dir=Path('../../data/large/cfdbench'),
    #     data_name='dam_prop',
    #     delta_time=0.1,
    #     norm_props=True,
    #     norm_bc=True,
    # )
    train_data_tube, dev_data_tube, test_data_tube = get_auto_dataset(
        data_dir=Path('../../data/large/cfdbench'),
        data_name='tube_prop_bc_geo',
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
    )

    cavity_lens = [data.shape[0] for data in train_data_cavity.all_features]
    cylinder_lens = [data.shape[0] for data in train_data_cylinder.all_features]
    tube_lens = [data.shape[0] for data in train_data_tube.all_features]
    # dam_lens = [data.shape[0] for data in train_data_dam.all_features]

    train_cavity_feats, train_cylinder_feats, train_tube_feats = train_data_cavity.all_features, train_data_cylinder.all_features, train_data_tube.all_features
    test_cavity_feats, test_cylinder_feats, test_tube_feats = test_data_cavity.all_features, test_data_cylinder.all_features, test_data_tube.all_features

    train_feats = train_cavity_feats + train_cylinder_feats + train_tube_feats
    test_feats = test_cavity_feats + test_cylinder_feats + test_tube_feats


    print(cavity_lens)
    print(cylinder_lens)
    print(tube_lens)
    # print(dam_lens)

    infer_steps = 20

    def split_trajectory(data_list, time_step, grid_size=64):
        traj_split = []
        for i, x in enumerate(data_list):
            T = x.shape[0]
            num_segments = int(np.ceil(T / time_step))
            padded_length = num_segments * time_step
            padded_array = np.zeros((padded_length, *x.shape[1:]))

            # Copy the original data into the padded array
            padded_array[:T, ...] = x

            # If needed, pad the last segment with the last frame of the original array
            if T % time_step != 0:
                last_frame = x[-1, ...]
                padded_array[T:, ...] = last_frame

            # Reshape the array into segments
            padded_array = F.interpolate(torch.from_numpy(padded_array),size=(grid_size,grid_size),mode='bilinear',align_corners=True).numpy()
            padded_array = padded_array.reshape((num_segments, time_step, *padded_array.shape[1:]))

            traj_split.append(padded_array)

        traj_split = np.concatenate(traj_split, axis=0)
        return traj_split


    train_data = split_trajectory(train_feats, infer_steps,grid_size=64)
    test_data = split_trajectory(test_feats, infer_steps,grid_size=64)
    train_data, test_data = train_data.transpose(0,3,4,1,2), test_data.transpose(0, 3, 4, 1, 2) # B, X, Y, T, C
    print(train_data.shape, test_data.shape)

    with h5py.File('./../data/cfdbench/ns2d_cdb_train.hdf5','w') as fp:
        fp.create_dataset('data',data=train_data,compression=None)



    with h5py.File('./../data/cfdbench/ns2d_cdb_test.hdf5','w') as fp:
        fp.create_dataset('data',data=test_data,compression=None)


if __name__ == '__main__':


    #### FNO datasets
    # preprocess_mat()

    #### PDEBench datasets
    # process_pdebench_data(path='./../data/164687',save_name='./../data/pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',n_train=9000, n_test=1000)
    # process_pdebench_data(path='./../data/164688',save_name='./../data/pdebench/ns2d_pdb_M1e-1_eta1e-1_zeta1e-1',n_train=9000, n_test=1000)
    
    # path = '/scratch3/wan410/operator_learning_data/pdebench/data/3D/Train/164690'
    # save_name = '/scratch3/wan410/operator_learning_data/pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2'
    # process_pdebench_data(path=path, save_name=save_name,n_train=9000, n_test=1000)
    # process_pdebench_data(path='./../data/164691',save_name='./../data//pdebench/ns2d_pdb_M1_eta1e-1_zeta1e-1',n_train=9000, n_test=1000)
    # process_pdebench_data(path='./../data/164685',save_name='./../data/pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512',n_train=900, n_test=100)
    # process_pdebench_data(path='./../data/164686',save_name='./../data/pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512',n_train=900, n_test=100)
    # process_pdebench_data(path='./../data/164689',save_name='./../data/pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_rand_512',n_train=900, n_test=100)
    # process_pdebench_data(path='./../data/164692',save_name='./../data/pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_rand_512',n_train=900, n_test=100)
    # process_swe_pdebench(path='./../data/133021',save_name='./../data/pdebench/swe_pdb',n_train=900, n_test=100)
    # process_dr_pdebench(path='./../data/133017',save_name='./../data/pdebench/dr_pdb',n_train=900, n_test=100)
    # process_pdebench3d_data(path='./../data/164693',save_name='./../data/pdebench/ns3d_pdb_M1_rand',n_train=90, n_test=10)
    # process_pdebench3d_data(path='./../data/173286',save_name='./../data/pdebench/ns3d_pdb_M1e-1_rand',n_train=90, n_test=10)
    # process_pdebench3d_data(path='./../data/164694',save_name='./../data/pdebench/ns3d_pdb_M1_turb',n_train=540, n_test=60)

    #### PDEArena datasets
    # load_path = '/scratch3/wan410/operator_learning_data/pdearena/NavierStokes-2D'
    # save_path = '/scratch3/wan410/operator_learning_data/pdearena/ns2d_pda'
    # preprocess_ns2d(load_path=load_path,
    #                 save_path=save_path)
    # preprocess_ns2d_longrollout()
    
    # preprocess_ns2d()
    # preprocess_ns2d_cond()
    # load_path = '/scratch3/wan410/operator_learning_data/pdearena/ShallowWater-2D'
    # save_path = '/scratch3/wan410/operator_learning_data/pdearena/sw2d_pda'
    # preprocess_shallow_water(load_path=load_path, save_path=save_path)


    #### CFDBench datasets
    # preprocess_cfdbench_data()


    #### torch-cfd datasets
    # load_path = '/scratch3/wan410/operator_learning_data/NS_torchcfd/data'
    # save_path = '/scratch3/wan410/operator_learning_data/NS_torchcfd/data/Re1000'
    # save_path = '/scratch3/wan410/operator_learning_data/NS_torchcfd/data/Re5000'
    # preprocess_torchcfd_ns2d(load_path=load_path, save_path=save_path, total_time=30)



    #### Dedalus datasets
    # states = ['val', 'test','train']
    states = ['test_long']
    for state in states:
        preprocess_dedalus_to_shards(dataset_name='ns2d_dedalus_big', save_dir='/scratch3/wan410/operator_learning_data/Dedalus/Forcing/'+state, 
        start_time_id=1000, # skip 1000 steps to keep stable initial condition
        shard_size=2048, dtype='float16', state=state)