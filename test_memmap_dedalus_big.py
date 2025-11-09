#!/usr/bin/env python3
"""
Test script for MemmapDedalusBigDataset2D dataset.
Tests dataset loading, shapes, normalization, and basic functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch
from utils.griddataset import MemmapDedalusBigDataset2D
from utils.make_master_file import DATASET_DICT

def test_dataset_basic():
    """Test basic dataset functionality."""
    print("=" * 70)
    print("Testing MemmapDedalusBigDataset2D - Basic Functionality")
    print("=" * 70)
    
    dataset_name = 'ns2d_dedalus_big'
    
    # Test parameters
    t_in = 1
    t_ar = 1
    form = 'vorticity'
    normalize = True
    
    print(f"\nDataset: {dataset_name}")
    print(f"T_in: {t_in}, T_ar: {t_ar}, form: {form}, normalize: {normalize}")
    print(f"Data path: {DATASET_DICT[dataset_name]['data_path']}")
    
    # Test train dataset
    print("\n" + "-" * 70)
    print("Testing TRAIN dataset")
    print("-" * 70)
    try:
        train_dataset = MemmapDedalusBigDataset2D(
            dataset_name, 
            t_in=t_in, 
            t_ar=t_ar, 
            form=form, 
            normalize=normalize, 
            train='train'
        )
        
        print(f"Dataset length: {len(train_dataset)}")
        print(f"Total timesteps (T_total): {train_dataset.T_total}")
        print(f"Number of channels (C_all): {train_dataset.C_all}")
        print(f"Selected channels: {train_dataset.channel_indices}")
        print(f"Number of selected channels: {train_dataset.n_channels}")
        print(f"Spatial resolution (H, W): ({train_dataset.H}, {train_dataset.W})")
        print(f"Number of shards: {len(train_dataset.shards)}")
        
        if normalize:
            print(f"\nNormalization stats:")
            print(f"  Mean shape: {train_dataset.norm_mean.shape}")
            print(f"  Std shape: {train_dataset.norm_std.shape}")
            print(f"  Mean: {train_dataset.norm_mean}")
            print(f"  Std: {train_dataset.norm_std}")
        
        # Test getting a sample
        print("\n" + "-" * 70)
        print("Testing __getitem__")
        print("-" * 70)
        if len(train_dataset) > 0:
            x, y = train_dataset[0]
            print(f"Sample 0:")
            print(f"  x shape: {x.shape} (should be (H, W, t_in, C))")
            print(f"  y shape: {y.shape} (should be (H, W, t_out, C))")
            print(f"  x dtype: {x.dtype}")
            print(f"  y dtype: {y.dtype}")
            print(f"  x min/max: {x.min().item():.6f} / {x.max().item():.6f}")
            print(f"  y min/max: {y.min().item():.6f} / {y.max().item():.6f}")
            
            # Test a few more samples
            if len(train_dataset) > 1:
                x2, y2 = train_dataset[1]
                print(f"\nSample 1:")
                print(f"  x shape: {x2.shape}")
                print(f"  y shape: {y2.shape}")
        else:
            print("Dataset is empty!")
            
    except Exception as e:
        print(f"ERROR loading train dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test val dataset
    print("\n" + "-" * 70)
    print("Testing VAL dataset")
    print("-" * 70)
    try:
        val_dataset = MemmapDedalusBigDataset2D(
            dataset_name, 
            t_in=t_in, 
            t_ar=t_ar, 
            form=form, 
            normalize=normalize, 
            train='val'
        )
        print(f"Dataset length: {len(val_dataset)}")
        print(f"Total timesteps (T_total): {val_dataset.T_total}")
        
        if len(val_dataset) > 0:
            x, y = val_dataset[0]
            print(f"Sample 0:")
            print(f"  x shape: {x.shape}")
            print(f"  y shape: {y.shape}")
            
    except Exception as e:
        print(f"ERROR loading val dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Test test dataset
    print("\n" + "-" * 70)
    print("Testing TEST dataset")
    print("-" * 70)
    try:
        test_dataset = MemmapDedalusBigDataset2D(
            dataset_name, 
            t_in=t_in, 
            t_ar=t_ar, 
            form=form, 
            normalize=normalize, 
            train='test'
        )
        print(f"Dataset length: {len(test_dataset)}")
        print(f"Total timesteps (T_total): {test_dataset.T_total}")
        
        if len(test_dataset) > 0:
            x, y = test_dataset[0]
            print(f"Sample 0:")
            print(f"  x shape: {x.shape}")
            print(f"  y shape: {y.shape}")
        
        # Test get_all_sequence for test dataset
        print("\n" + "-" * 70)
        print("Testing get_all_sequence()")
        print("-" * 70)
        x_all, y_all = test_dataset.get_all_sequence()
        print(f"x_all shape: {x_all.shape}")
        print(f"y_all shape: {y_all.shape}")
        print(f"x_all dtype: {x_all.dtype}")
        print(f"y_all dtype: {y_all.dtype}")
        print(f"x_all min/max: {x_all.min().item():.6f} / {x_all.max().item():.6f}")
        print(f"y_all min/max: {y_all.min().item():.6f} / {y_all.max().item():.6f}")
        
    except Exception as e:
        print(f"ERROR loading test dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Test DataLoader
    print("\n" + "-" * 70)
    print("Testing DataLoader")
    print("-" * 70)
    try:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=4, 
            shuffle=True, 
            num_workers=0
        )
        
        batch_x, batch_y = next(iter(train_loader))
        print(f"Batch from DataLoader:")
        print(f"  batch_x shape: {batch_x.shape} (should be (batch_size, H, W, t_in, C))")
        print(f"  batch_y shape: {batch_y.shape} (should be (batch_size, H, W, t_out, C))")
        print(f"  batch_x dtype: {batch_x.dtype}")
        print(f"  batch_y dtype: {batch_y.dtype}")
        
    except Exception as e:
        print(f"ERROR with DataLoader: {e}")
        import traceback
        traceback.print_exc()
    
    # Test normalization
    print("\n" + "-" * 70)
    print("Testing Normalization")
    print("-" * 70)
    try:
        if normalize and train_dataset.norm_mean is not None:
            x, y = train_dataset[0]
            x_norm = train_dataset.normalize_x(x)
            x_denorm = train_dataset.denormalize_x(x_norm)
            
            print(f"Original x shape: {x.shape}")
            print(f"Normalized x shape: {x_norm.shape}")
            print(f"Denormalized x shape: {x_denorm.shape}")
            print(f"Normalized x mean: {x_norm.mean().item():.6f} (should be ~0)")
            print(f"Normalized x std: {x_norm.std().item():.6f} (should be ~1)")
            print(f"Reconstruction error (max abs diff): {(x - x_denorm).abs().max().item():.6e}")
            
    except Exception as e:
        print(f"ERROR testing normalization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


def test_different_forms():
    """Test different form options."""
    print("\n" + "=" * 70)
    print("Testing Different Forms")
    print("=" * 70)
    
    dataset_name = 'ns2d_dedalus_big'
    t_in = 1
    t_ar = 1
    
    forms = ['vorticity']  # Add 'velocity' if needed
    
    for form in forms:
        print(f"\nForm: {form}")
        try:
            dataset = MemmapDedalusBigDataset2D(
                dataset_name, 
                t_in=t_in, 
                t_ar=t_ar, 
                form=form, 
                normalize=False, 
                train='train'
            )
            print(f"  Channel indices: {dataset.channel_indices}")
            print(f"  Number of channels: {dataset.n_channels}")
            
            if len(dataset) > 0:
                x, y = dataset[0]
                print(f"  x shape: {x.shape}")
                print(f"  y shape: {y.shape}")
                
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    test_dataset_basic()
    test_different_forms()

