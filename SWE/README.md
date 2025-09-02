# NSE (Navier-Stokes Equations) Module

This folder contains a copy of the main training and testing scripts specifically organized for NSE (Navier-Stokes Equations) experiments. This separation allows for easier experimentation with different datasets like SWE (Shallow Water Equations) without conflicts.

## Files Included

- `train_AR_NO.py` - Training script for autoregressive neural operators
- `test_AR_NO.py` - Testing script for autoregressive neural operators  
- `train_diffusion_NO.py` - Training script for diffusion-based neural operators
- `test_diffusion_NO.py` - Testing script for diffusion-based neural operators
- `train_pderefiner.py` - Training script for PDERefiner model
- `test_pderefiner.py` - Testing script for PDERefiner model
- `utils_plot.py` - Utilities for plotting and visualization

## Key Changes Made

### Import Path Fixes
All scripts have been updated to properly import dependencies from the parent directory:

```python
import sys
import os
# Add parent directory to Python path to access utils and models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```

This allows the scripts to access:
- `utils/` directory (datasets, optimizers, criteria, etc.)
- `models/` directory (neural network architectures)
- Other dependencies from the main project

### Dependencies
The scripts depend on the following directories from the parent folder:
- `../utils/` - Utility functions, dataset loaders, optimizers
- `../models/` - Neural network model implementations
- `../pdearena/` - Data files (if using local data)

## Usage

### Running from NSE directory
```bash
cd NSE
python3 train_AR_NO.py --model FNO --dataset ns2d_pda --epochs 100
```

### Running from parent directory
```bash
python3 NSE/train_AR_NO.py --model FNO --dataset ns2d_pda --epochs 100
```

## Available Models
- `FNO` - Fourier Neural Operator
- `UNO` - U-shaped Neural Operator
- `wavelet_transformer` - Wavelet Transformer
- `HFS` - High Frequency Scaling
- `UNet_withoutHFS` - U-Net without High Frequency Scaling
- `UNet` - U-Net with Bottleneck HFS
- `HANO` - Hierarchical Attention Neural Operator
- `PDERefiner` - PDE Refiner model

## Available Datasets
- `ns2d_pda` - 2D Navier-Stokes equations (PDE Arena format)
- `ns2d_fno_1e-3` - 2D Navier-Stokes equations (FNO format)
- `sw2d_pda` - 2D Shallow Water equations
- And others as defined in `../utils/make_master_file.py`

## Example Commands

### Train FNO on NSE data
```bash
python3 train_AR_NO.py --model FNO --dataset ns2d_pda --epochs 2000 --batch_size 64 --lr 0.001
```

### Test trained model
```bash
python3 test_AR_NO.py --model FNO --dataset ns2d_pda --epochs 2000
```

### Train with diffusion model
```bash
python3 train_diffusion_NO.py --model FNO --dataset ns2d_pda --epochs 3000
```

## Notes

1. **Data Location**: Make sure the data files are in the correct location (`../pdearena/` for local data)
2. **GPU Usage**: Scripts will automatically use GPU if available, CPU otherwise
3. **Logging**: Results are saved to `../logs/` directory by default
4. **Dependencies**: Ensure all required Python packages are installed (torch, numpy, matplotlib, etc.)

## Troubleshooting

If you encounter import errors:
1. Make sure you're running the scripts with `python3`
2. Verify that the parent directory structure is intact
3. Check that `../utils/` and `../models/` directories exist

## Future Extensions

This modular structure makes it easy to:
- Add new datasets (SWE, etc.) without affecting NSE experiments
- Modify NSE-specific configurations
- Run comparative experiments between different equation types
- Maintain separate logging and results for different physics domains
