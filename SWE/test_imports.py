#!/usr/bin/env python3
"""
Test script to verify that all imports work correctly in the NSE module.
"""

import sys
import os

# Add parent directory to Python path to access utils and models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test all critical imports."""
    print("Testing imports for NSE module...")
    
    try:
        # Test utils imports
        from utils.optimizer import Adam, Lamb
        from utils.utilities import count_parameters
        from utils.criterion import RelL2Norm
        from utils.griddataset import TemporalDataset2D
        from utils.make_master_file import DATASET_DICT
        print("✓ Utils imports successful")
        
        # Test model imports
        from models.fno import FNO2d
        from models.uno import UNO
        from models.wavelet_transform import CrossWaveletTransformer
        from models.high_frequency_scaling import ResUNet
        from models.unet import UNet_with_BottleneckHFS, UNet_withoutHFS
        from models.hano import HANO2d
        from models.pderefiner import PDERefiner
        print("✓ Model imports successful")
        
        # Test external dependencies
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        print("✓ External dependencies available")
        
        # Test dataset availability
        print(f"✓ Available datasets: {list(DATASET_DICT.keys())}")
        
        print("\n🎉 All imports successful! NSE module is ready to use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_device_availability():
    """Test device availability."""
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available, will use CPU")
    
    print(f"✓ PyTorch version: {torch.__version__}")

if __name__ == "__main__":
    print("=" * 60)
    print("NSE Module Import Test")
    print("=" * 60)
    
    success = test_imports()
    test_device_availability()
    
    if success:
        print("\n✅ NSE module is properly configured and ready for use!")
        sys.exit(0)
    else:
        print("\n❌ NSE module has configuration issues.")
        sys.exit(1)
