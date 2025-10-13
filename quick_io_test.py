#!/usr/bin/env python3
"""
Quick I/O test to measure HDF5 access performance
"""
import time
import numpy as np
import h5py
import os

def test_hdf5_performance(data_path):
    """Quick test of HDF5 performance"""
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
    
    print(f"Testing HDF5 performance on: {data_path}")
    
    # Test 1: File open/close overhead
    print("\n1. File open/close overhead:")
    times = []
    for i in range(5):
        start = time.time()
        with h5py.File(data_path, 'r') as f:
            _ = f['tasks/vorticity'].shape
        times.append(time.time() - start)
    print(f"   Average file open/close: {np.mean(times):.3f}s")
    
    # Test 2: Single sample access
    print("\n2. Single sample access:")
    with h5py.File(data_path, 'r') as f:
        times = []
        for i in range(10):
            start = time.time()
            vorticity = np.array(f['tasks/vorticity'][i])
            streamfunction = np.array(f['tasks/streamfunction'][i])
            timestep = np.array(f['scales/timestep'][i])
            times.append(time.time() - start)
        print(f"   Average single sample access: {np.mean(times):.3f}s")
        print(f"   Data shape: {vorticity.shape}")
    
    # Test 3: Batch access
    print("\n3. Batch access:")
    with h5py.File(data_path, 'r') as f:
        start = time.time()
        vorticity_batch = np.array(f['tasks/vorticity'][:10])
        streamfunction_batch = np.array(f['tasks/streamfunction'][:10])
        timestep_batch = np.array(f['scales/timestep'][:10])
        batch_time = time.time() - start
        print(f"   Batch access (10 samples): {batch_time:.3f}s")
        print(f"   Speedup vs single: {np.mean(times) / (batch_time/10):.1f}x")
    
    # Test 4: Memory usage estimation
    print("\n4. Memory usage estimation:")
    with h5py.File(data_path, 'r') as f:
        total_samples = f['tasks/vorticity'].shape[0]
        sample_size = f['tasks/vorticity'][0].nbytes
        total_size_mb = (total_samples * sample_size * 3) / (1024 * 1024)  # 3 variables
        print(f"   Total samples: {total_samples}")
        print(f"   Sample size: {sample_size / 1024:.1f} KB")
        print(f"   Total dataset size: {total_size_mb:.1f} MB")
        print(f"   Memory for 100 samples: {(100 * sample_size * 3) / (1024 * 1024):.1f} MB")

if __name__ == "__main__":
    # Update this path to your actual data file
    data_path = "/datasets/work/oa-tcch/work/forXuesong/data/realisation_0000/snapshots/snapshots_s1.h5"
    test_hdf5_performance(data_path)
