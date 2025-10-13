#!/usr/bin/env python3
"""
Benchmark script to test dataloader performance
"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py
import psutil
import os
from utils.griddataset import DedalusDataset2D

def benchmark_current_implementation():
    """Benchmark the current inefficient implementation"""
    print("=== Benchmarking Current Implementation ===")
    
    # Create dataset
    dataset = DedalusDataset2D(
        data_name='ns2d_dedalus',  # Assuming this exists in your DATASET_DICT
        t_in=7, 
        t_ar=1, 
        form='vorticity',
        normalize=True,
        train='train'
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Benchmark
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    times = []
    for i, (x, y) in enumerate(dataloader):
        if i >= 10:  # Test first 10 samples
            break
        
        sample_start = time.time()
        print(f"Sample {i}: x.shape={x.shape}, y.shape={y.shape}")
        sample_time = time.time() - sample_start
        times.append(sample_time)
        
        # Print memory usage
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"  Sample time: {sample_time:.3f}s, Memory: {current_memory:.1f}MB")
    
    total_time = time.time() - start_time
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    print(f"\nResults:")
    print(f"  Total time for 10 samples: {total_time:.3f}s")
    print(f"  Average time per sample: {np.mean(times):.3f}s")
    print(f"  Memory usage: {start_memory:.1f}MB -> {end_memory:.1f}MB")
    
    return times

def benchmark_hdf5_raw_access():
    """Benchmark raw HDF5 access to see baseline performance"""
    print("\n=== Benchmarking Raw HDF5 Access ===")
    
    # You'll need to update this path
    data_path = "/datasets/work/oa-tcch/work/forXuesong/data/realisation_0000/snapshots/snapshots_s1.h5"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    times = []
    
    # Test different access patterns
    with h5py.File(data_path, 'r') as f:
        print(f"HDF5 file structure: {list(f.keys())}")
        
        # Test 1: Single timestep access
        start_time = time.time()
        for i in range(10):
            vorticity = np.array(f['tasks/vorticity'][i])
            streamfunction = np.array(f['tasks/streamfunction'][i])
        single_access_time = time.time() - start_time
        times.append(single_access_time)
        print(f"  Single timestep access (10 samples): {single_access_time:.3f}s")
        
        # Test 2: Batch access
        start_time = time.time()
        vorticity_batch = np.array(f['tasks/vorticity'][:10])
        streamfunction_batch = np.array(f['tasks/streamfunction'][:10])
        batch_access_time = time.time() - start_time
        times.append(batch_access_time)
        print(f"  Batch access (10 samples): {batch_access_time:.3f}s")
        
        # Test 3: File open/close overhead
        file_times = []
        for i in range(10):
            start_time = time.time()
            with h5py.File(data_path, 'r') as f2:
                _ = np.array(f2['tasks/vorticity'][0])
            file_times.append(time.time() - start_time)
        file_overhead = np.mean(file_times)
        times.append(file_overhead)
        print(f"  File open/close overhead (per sample): {file_overhead:.3f}s")
    
    return times

def benchmark_optimized_version():
    """Benchmark the optimized version with caching"""
    print("\n=== Benchmarking Optimized Version ===")
    
    try:
        from utils.optimized_griddataset import OptimizedDedalusDataset2D
        
        # Create optimized dataset
        dataset = OptimizedDedalusDataset2D(
            data_name='ns2d_dedalus',
            t_in=7, 
            t_ar=1, 
            form='vorticity',
            normalize=False,
            train='train',
            cache_size_mb=500  # 500MB cache
        )
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Benchmark
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        times = []
        for i, (x, y) in enumerate(dataloader):
            if i >= 10:
                break
            
            sample_start = time.time()
            print(f"Optimized Sample {i}: x.shape={x.shape}, y.shape={y.shape}")
            sample_time = time.time() - sample_start
            times.append(sample_time)
            
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"  Sample time: {sample_time:.3f}s, Memory: {current_memory:.1f}MB")
        
        total_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"\nOptimized Results:")
        print(f"  Total time for 10 samples: {total_time:.3f}s")
        print(f"  Average time per sample: {total_time/len(times):.3f}s")
        print(f"  Memory usage: {start_memory:.1f}MB -> {end_memory:.1f}MB")
        
        return times
        
    except ImportError as e:
        print(f"Could not import optimized version: {e}")
        return None

def benchmark_fully_cached_version():
    """Benchmark the fully cached version"""
    print("\n=== Benchmarking Fully Cached Version ===")
    
    try:
        from utils.optimized_griddataset import FullyCachedDedalusDataset2D
        
        # Create fully cached dataset
        dataset = FullyCachedDedalusDataset2D(
            data_name='ns2d_dedalus',
            t_in=7, 
            t_ar=1, 
            form='vorticity',
            normalize=False,
            train='train'
        )
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Benchmark
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        times = []
        for i, (x, y) in enumerate(dataloader):
            if i >= 10:
                break
            
            sample_start = time.time()
            print(f"Fully Cached Sample {i}: x.shape={x.shape}, y.shape={y.shape}")
            sample_time = time.time() - sample_start
            times.append(sample_time)
            
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"  Sample time: {sample_time:.3f}s, Memory: {current_memory:.1f}MB")
        
        total_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"\nFully Cached Results:")
        print(f"  Total time for 10 samples: {total_time:.3f}s")
        print(f"  Average time per sample: {np.mean(times):.3f}s")
        print(f"  Memory usage: {start_memory:.1f}MB -> {end_memory:.1f}MB")
        
        return times
        
    except ImportError as e:
        print(f"Could not import fully cached version: {e}")
        return None
    
if __name__ == "__main__":
    print("Starting DataLoader Performance Benchmark...")
    
    # Run benchmarks
    current_times = benchmark_current_implementation()
    hdf5_times = benchmark_hdf5_raw_access()
    optimized_times = benchmark_optimized_version()
    cached_times = benchmark_fully_cached_version()
    
    # Analysis
    print(f"\n=== Performance Analysis ===")
    if current_times and hdf5_times:
        print(f"Current dataloader: {np.mean(current_times):.3f}s per sample")
        print(f"Raw HDF5 batch: {hdf5_times[1]:.3f}s for 10 samples")
        print(f"File overhead: {hdf5_times[2]:.3f}s per sample")
        
        if optimized_times:
            print(f"Optimized dataloader: {np.mean(optimized_times):.3f}s per sample")
            print(f"Speedup vs current: {np.mean(current_times) / np.mean(optimized_times):.1f}x")
        
        if cached_times:
            print(f"Fully cached dataloader: {np.mean(cached_times):.3f}s per sample")
            print(f"Speedup vs current: {np.mean(current_times) / np.mean(cached_times):.1f}x")
        
        print(f"Potential theoretical speedup: {np.mean(current_times) / hdf5_times[2]:.1f}x")
