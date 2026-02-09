# SW2D PDA (Shallow Water 2D — PDE Arena) Benchmark

This folder contains the **2D Shallow Water (PDE Arena)** benchmark for neural operator learning. Models are trained to predict **vorticity and pressure** one step ahead and evaluated via **autoregressive rollout** on a held-out validation set, with metrics on relative L² error, spectral energy, and enstrophy (computed from the vorticity channel).

Data and setup follow the **PDE Arena** Shallow Water 2D dataset: [pdearena/ShallowWater-2D](https://huggingface.co/datasets/pdearena/ShallowWater-2D).

---

## Benchmark Setup

### PDE and Data

- **Equation:** 2D Shallow Water (vorticity and pressure).
- **State variables used:** **vorticity (`vor`)** and **pressure (`pres`)** only (2 channels). Raw PDE Arena data also has `u`, `v`, `div`; this benchmark uses only the last two channels.
- **Spatial resolution:** 96 × 192 (rectangular grid).
- **Temporal resolution:** 88 time steps per trajectory.
- **Input/output:** Each time step has shape `(H, W, 2)` — vorticity and pressure. Models take the current state (and optionally a spatial grid) and predict the next state.

### Train and Validation Data

- **Format:** Preprocessed `.npy` arrays of shape `(N, 96, 192, 88, 2)` (N trajectories, 96×192 space, 88 time steps, 2 channels: vor, pres).
- **Normalization:** A normalizer is saved separately (`normstats.pt`) with per-channel mean and std for `vor` and `pres`. Training and test loaders use it so that data are normalized consistently.
- **Typical setup:**
  - **Train:** `sw2d_pda_data_train.npy` (e.g. 4000 samples), built from PDE Arena train (and optionally valid) HDF5 files.
  - **Validation:** `sw2d_pda_data_val.npy` (e.g. 100 samples).
- **Paths:** Set `data.datapath`, `data.normalizer_path`, and `test_data.datapath` in your config to match your environment.

### Data Preparation (from raw PDE Arena)

Raw PDE Arena data are in NetCDF/HDF5 with 5 channels (`u`, `v`, `div`, `vor`, `pres`). This benchmark keeps only `vor` and `pres`. The `data_utils/datasets.py` module provides:

1. **`preprocess_shallow_water(load_path, save_path)`** — Convert raw `.nc` files to HDF5 with a `data` key.
2. **`load_save_sw_data(folder_path, max_files=4000, state='train')`** — Load HDF5 files from a folder, take the last two channels (vor, pres), stack into one array, and save as e.g. `sw2d_pda_data_train.npy` or `sw2d_pda_data_val.npy` in the parent directory. Compute and save `normstats.pt` separately if you need a fixed normalizer (see `SWLoader2D.normalize()`).

---

## Project Layout

```
SW2D_PDA/
├── configs/
│   ├── linear/                    # Linear grid configs
│   │   ├── FNO.yaml
│   │   ├── HFS.yaml
│   │   ├── WNO.yaml
│   │   ├── SAOT.yaml
│   │   ├── PDERefinerUNet.yaml
│   │   └── MSWT_patching.yaml
│   └── periodic_used_in_paper/    # Periodic grid configs (paper)
│       ├── FNO_periodic.yaml
│       ├── HFS_periodic.yaml
│       ├── WNO_periodic.yaml
│       ├── SAOT_periodic.yaml
│       ├── PDERefinerUNet_periodic.yaml
│       └── MSWT_periodic.yaml
├── data_utils/
│   └── datasets.py                # SWLoader2D, preprocessing, normalizer
├── train_operator_AR_rell2_2d.py # Training (one-step L2)
├── test_operator_AR_rell2_2d.py   # Evaluation (AR rollout + metrics)
├── post_processing_metric_table.py # Aggregate metrics, LaTeX tables, figures
├── submit_bash.sh                # Single SLURM job
└── submit_multiseed.sh           # One job per seed
```

Run training and testing from the **repository root** (parent of `SW2D_PDA/`) so that `models/` and `utils/` are on the path.

---

## Configuration (YAML)

Each config has four main sections:

| Section      | Role |
|-------------|------|
| `data`      | Train data path, `normalizer_path`, resolution (`nx`, `ny`, `nt`), `sub`/`sub_t`, `grid_form`, etc. |
| `model`     | Architecture and hyperparameters. For SW2D, **input/output channels** are 2 (vor, pres); with grid concatenated, `in_dim` is typically 6 (e.g. 2 + 4 for 2D grid), `out_dim` 2. |
| `train`     | `batchsize`, `epochs`, `milestones`, `base_lr`, `scheduler_gamma`, `save_dir`, `save_name`, optional `num_workers`. |
| `test_data` | Validation data path, resolution, `nx`, `ny`, `nt`. |

Important:

- **Normalizer:** `data.normalizer_path` must point to a `.pt` file with keys `vor` and `pres`, each with `mean` and `std` (e.g. `(1, H, W)`). Used by `SWLoader2D` and by `load_sw_sequences()` in the test script.
- **Checkpoints:** `train.save_name` is suffixed with `_seed{test_seed}.pt` so each seed has its own checkpoint under `train.save_dir`.
- **Grid:** `data.grid_form` can be `linear` or `periodic`; configs under `periodic_used_in_paper/` use `periodic`.

---

## Training

One-step-ahead training: the model sees state at time \(t\) (and optionally grid) and is trained to predict state at \(t+1\) (both vor and pres) with relative L² loss.

**Command:**

```bash
cd /path/to/multiscale_neural_operator
python SW2D_PDA/train_operator_AR_rell2_2d.py \
  --config_path SW2D_PDA/configs/periodic_used_in_paper/FNO_periodic.yaml \
  --test_ratio 0.25 \
  --test_seed 42
```

**Options:**

| Argument              | Default | Description |
|-----------------------|--------|-------------|
| `--config_path`       | —      | Path to YAML config (required). |
| `--test_ratio`        | 0.0    | Fraction of data for validation (e.g. 0.25). |
| `--test_seed`         | 42     | Random seed for train/val split. |
| `--synthetic_samples` | 0      | If &gt;0, use synthetic random data. |
| `--max_epochs`        | None   | Override `train.epochs` for shorter runs. |
| `--resume_training`   | false  | Resume from last checkpoint in `save_dir`. |
| `--resume_ckpt`       | None   | Resume from a specific checkpoint in `save_dir`. |

Checkpoints are saved under `train.save_dir` as `{save_name}` with `_seed{test_seed}.pt` appended.

---

## Evaluation (Autoregressive Rollout)

Evaluation loads a checkpoint and runs **autoregressive rollout** on the validation set: start from the first time step, then repeatedly feed the model’s prediction as the next input.

**Command:**

```bash
python SW2D_PDA/test_operator_AR_rell2_2d.py \
  --config_path SW2D_PDA/configs/periodic_used_in_paper/FNO_periodic.yaml \
  --test_seed 42
```

- **Note:** `--test_seed` is required (no default in the script). It must match the seed used when training the checkpoint.
- The script loads test data from `test_data.datapath`, normalizes with `data.normalizer_path`, and expects the checkpoint at `{train.save_dir}/{train.save_name}` (with `.pt` already replaced by `_seed{test_seed}.pt` in the loaded config).
- Outputs:
  - **Metrics:** `{save_dir}/evaluation_metrics/` (e.g. CSV per model/seed).
  - **Fields and spectra:** `{save_dir}/saved_plots/` (ground truth, prediction, error, energy/enstrophy spectra; vorticity channel used for spectra).

Metrics are computed at time indices defined in the script (e.g. 0, 40, 80 corresponding to steps 1, 41, 81 in the LaTeX table).

### Metrics

| Metric | Description |
|--------|-------------|
| **Rel L²** | Relative L² error on the full state (vor + pres). |
| **SMLR**   | Spectral Mean Log-Ratio (energy spectrum from vorticity). |
| **EMLR**   | Enstrophy Mean Log-Ratio. |
| **SMAE**   | Spectral Mean Absolute Error. |
| **EMAE**   | Enstrophy Mean Absolute Error. |

---

## Post-Processing (Tables and Figures)

`post_processing_metric_table.py` provides:

1. **`aggregate_metric_table(grid_form='linear')`** — Reads per-seed CSVs from each model’s `evaluation_metrics/` folder under `save_folder`, groups by model, computes **mean ± std** across seeds, and writes `total_evaluation_metrics_{grid_form}.csv` and `avg_evaluation_metrics_{grid_form}.csv`. Uses a fixed list of model folder names (FNO, PDERefinerUNet, WNO, SAOT, HFS, MSWT_patching).
2. **`process_metric_table_to_latex()`** — Reads the aggregated CSV (e.g. `logs/SW2D_PDA/avg_evaluation_metrics_periodic.csv`), formats two LaTeX tables (Rel L²/SMLR/EMLR and SMAE/EMAE) for steps 1, 41, 81, and writes `.tex` files into `logs/SW2D_PDA/`.
3. **Plotting functions** — Load saved predictions and spectra from `saved_plots/`, plot vorticity prediction/error and energy/enstrophy spectra at selected steps.

Set `CSV_PATH`, `save_folder`, and (if needed) scratch paths inside the script to match your setup. Run the desired functions (e.g. from `if __name__ == '__main__'`).

---

## Slurm Scripts

- **`submit_bash.sh`**  
  Single job: edit the script to choose the config and whether to run training and/or testing. Uncomment the appropriate `python3 train_...` or `python3 test_...` line. **Note:** The comments in the script use `configs/periodict_used_in_paper/` (typo); the actual directory is `configs/periodic_used_in_paper/`. Adjust `#SBATCH` (time, account, GPU, etc.) as needed.

- **`submit_multiseed.sh`**  
  Submits one job per seed (default: 42, 43, 44, 45, 46) for a given config.  
  **Usage:**  
  `./submit_multiseed.sh [config_path] [test_ratio]`  
  Example:  
  `./submit_multiseed.sh configs/periodic_used_in_paper/FNO_periodic.yaml 0.25`  
  Each job runs the **test** script with the corresponding seed. Change the script if you want to run training per seed.

Run from the `SW2D_PDA` directory so that `configs/` is found. Adapt module and venv lines to your cluster.

---

## Baselines

| Model | Config name | Description |
|-------|-------------|-------------|
| **FNO** | `fno2d` (default if no `name`) | Fourier Neural Operator 2D; `in_dim`/`out_dim` set for 2 channels + grid. |
| **HFS** | `hfs` | High-Frequency Scaling (ResUNet). |
| **WNO** | `wno` / `wno2d` | Wavelet Neural Operator. |
| **SAOT** | `saot` / `saot2d` | Space-time attention operator. |
| **PDERefiner / Unet** | `pderefiner` / `refiner_unet` | PDE refinement with UNet. |
| **MSWT** | `multiscale_wavelet2d_periodic_patching` (or `mswt_periodic_patching`) | Multiscale wavelet with patching. |

Configs in `configs/linear/` and `configs/periodic_used_in_paper/` are the main references.

---

## Quick Start Summary

1. **Data:** Download or generate PDE Arena Shallow Water 2D data; run preprocessing to get `sw2d_pda_data_train.npy`, `sw2d_pda_data_val.npy`, and `normstats.pt`. Set paths in a config.
2. **Train:**  
   `python SW2D_PDA/train_operator_AR_rell2_2d.py --config_path SW2D_PDA/configs/periodic_used_in_paper/FNO_periodic.yaml --test_ratio 0.25 --test_seed 42`
3. **Test:**  
   `python SW2D_PDA/test_operator_AR_rell2_2d.py --config_path SW2D_PDA/configs/periodic_used_in_paper/FNO_periodic.yaml --test_seed 42`
4. **Post-process:** Run `aggregate_metric_table(grid_form='periodic')` (or `'linear'`), then `process_metric_table_to_latex()` and any plotting in `post_processing_metric_table.py` after evaluating all models/seeds.

For multi-seed evaluation via SLURM, use `submit_multiseed.sh` with the desired config and (optionally) test ratio.
