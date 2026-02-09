# ERA5 Climate Prediction Benchmark

This folder contains the **ERA5-based global atmospheric prediction** benchmark for neural operators on the sphere. Models are trained to predict **one-step tendencies** (and precipitation) from current atmospheric state and forcing, then evaluated via **long autoregressive rollout**. The main evaluation metric is **climatology bias**: the mean absolute difference between the rollout temporal mean and the true climatology over a long window (e.g. 10 years).

The setup follows **LUCIE**-style global weather / climate modeling: spherical grid (latitude × longitude), Legendre–Gauss quadrature for loss, and prescribed forcing (e.g. TISR, orography) that cycles annually.

---

## Benchmark Setup

### Data and Grid

- **Source:** ERA5 reanalysis (e.g. regridded to a coarse resolution for training).
- **Grid:** **48 × 96** (latitude × longitude), typically on a **Legendre–Gauss** latitude grid. Longitude is equidistant.
- **Variables (6 prognostic + 2 forcing):**
  - **Input (7 channels):** temperature, humidity, u_wind, v_wind, surface_pressure (normalized); **tisr** (top-of-atmosphere solar radiation), **orography** (repeating / fixed). The first 5 are the “prognostic” state; the last 2 are forcing that cycles (e.g. 1460 steps per year).
  - **Output (6 channels):** For the first 5 channels the model predicts **normalized tendencies** (difference from previous step); for the 6th (precipitation) it predicts a **diagnostic** in log space. Autoregressive inference then denormalizes and combines tendency + previous state to get the next state.

### Train / Val Split (in code)

- **Training:** first 16,000 time steps from the preprocessed dataset.
- **Validation:** next 100 time steps.
- **Out-of-sample evaluation:** initial state from index `16000+100`, forcing from the corresponding position in the 1460-step forcing cycle; rollout length (e.g. 14,600 steps) is set in the test script.

### Preprocessed Data Files

Training and testing expect preprocessed `.npz` files in a data folder (path set in `data_utils/data_utils.py` and `data_utils/data_preprocessing.py`):

| File | Role |
|------|------|
| `era5_T30_preprocessed.npz` | Standardized inputs/targets and normalizer stats: `data_inp`, `data_tar`, `raw_means`, `raw_stds`, `diag_means`, `diag_stds`, `diff_means`, `diff_stds`. |
| `era5_T30_regridded.npz` | Raw regridded ERA5 (used to compute climatology if needed). |
| `era5_T30_clim.npz` | Saved climatology (`true_clim`) for evaluation. |
| (optional) `era5_512gg_*` | High-resolution variant when `load_high_res=True`. |

**Data folder:** Default in `data_utils/data_utils.py` is `/scratch3/wan410/operator_learning_data/LUCIE` when CUDA is available, else `saved_data`. Update for your environment.

**Preprocessing:** `data_utils/data_preprocessing.py` builds `era5_T30_preprocessed.npz` from `era5_T30_regridded.npz`: it normalizes inputs, computes tendencies and their stats for the first 5 channels, and log-normalizes precipitation. Run it once to generate the preprocessed file (and ensure `data_folder` and input file paths are set correctly).

---

## Project Layout

```
ERA5/
├── config/
│   ├── LUCIE.yaml          # Spherical FNO (LUCIE)
│   ├── HFS_sphere.yaml     # High-Frequency Scaling on sphere
│   └── MSWT_sphere.yaml    # Multiscale wavelet on sphere
├── data_utils/
│   ├── data_utils.py       # load_data_era5, paths, normalizer arrays
│   ├── data_preprocessing.py # Build preprocessed .npz from regridded data
│   ├── convert_npz_to_h5.py
│   ├── load_era5_for_gadi.py
│   └── ...
├── models/                 # Local sphere-specific models (FNO, HFS, MSWT, torch_harmonics)
│   ├── periodic_mswt.py
│   ├── high_frequency_scaling.py
│   ├── torch_harmonics_local.py  # LUCIE / spherical FNO
│   └── ...
├── utils/
│   ├── compute_diagnostics.py
│   └── utils.py
├── 2d_train_rel_l2.py      # Training (one-step relative L2 + optional spectral reg)
├── 2d_test_rel_l2.py       # Evaluation (rollout, climatology bias, metrics)
├── lucie_inference.py      # Autoregressive inference with forcing cycle
├── post_processing_metric_table.py # Aggregate metrics, tables, bias plots
├── submit_bash.sh
└── submit_multiseed.sh
```

Run training and testing from the **ERA5** directory (or set `PYTHONPATH` so that `models` and `utils` resolve to `ERA5/models` and `ERA5/utils`).

---

## Configuration (YAML)

Configs under `config/` have two main sections:

| Section | Role |
|--------|------|
| `model` | Architecture name and hyperparameters. `name` can be `lucie`, `hfs`, `mswt_sphere`, `mswt_patch_sphere`, etc. Channel counts: 7 in, 6 out. |
| `train` | `save_dir`, `save_name`, `base_lr`, `scheduler_gamma`, `milestones`, `epochs`. |

Checkpoints are saved under `train.save_dir`. The training script appends `_seed{seed}.pt` to `save_name`, and also saves a **best** model as `{save_name}` with `.pt` replaced by `_best.pt` (lowest rollout climatology bias after epoch 60). The test script loads the **best** checkpoint: it replaces `.pt` with `_seed{seed}_best.pt` (see below).

---

## Training

One-step training: the model sees the current state (7 channels) and predicts the 6-channel target (tendencies for the first 5, diagnostic for precipitation). Loss is **relative L² on the sphere** (Legendre–Gauss quadrature) for the first 5 channels plus MSE for precipitation; an optional spectral regularizer is applied after a set number of epochs.

**Command:**

```bash
cd /path/to/multiscale_neural_operator/ERA5
python 2d_train_rel_l2.py --config_path config/MSWT_sphere.yaml --seed 42
```

**Options:**

| Argument | Default | Description |
|----------|--------|-------------|
| `--config_path` | `config.yaml` | Path to YAML config. |
| `--seed` | 42 | Random seed; also used for checkpoint suffix `_seed{seed}.pt`. |
| `--resume_training` | False | Resume from last checkpoint in `save_dir`. |
| `--load_high_res` | False | Load high-resolution ERA5 data if available. |
| `--use_writer` | False | Use TensorBoard writer. |

Training periodically runs a **2-year rollout** (2920 steps), computes the rollout climatology and its bias vs. true climatology, and after epoch 60 saves the current model every 10 epochs and a **best** model whenever this bias improves. The best checkpoint is saved with the same `save_name` but with `.pt` replaced by `_best.pt` (training script); the test script then looks for `_seed{seed}_best.pt` (see Evaluation).

---

## Evaluation (Long Rollout and Climatology)

Evaluation loads a checkpoint and runs **out-of-sample autoregressive rollout** (e.g. 14,600 steps) from a fixed initial state and forcing index. The temporal mean of the rollout is compared to the true climatology.

**Command:**

```bash
python 2d_test_rel_l2.py --config_path config/MSWT_sphere.yaml --seed 45
```

- **Checkpoint:** The script builds the path from `config['train']['save_dir']` and `config['train']['save_name']`, then **replaces `.pt` with `_seed{args.seed}_best.pt`**. So it expects the **best** checkpoint file named like `MSWT_earth_padding-dim512-layer3_seed45_best.pt` in the config’s `save_dir`. Ensure the training script has saved the best model with the same naming convention (e.g. `_best.pt` in the same directory; the test script applies the seed in the filename).
- **Rollout:** Uses `lucie_inference.inference()`: at each step the model gets the current normalized state (7 ch) and predicts 6 ch; tendencies are denormalized and combined with the previous state; forcing (TISR, orography) is updated from the 1460-step cycle.
- **Outputs:**
  - Rollout and true climatology can be saved (e.g. `rollout_model_{model_name}_seed{seed}.pt`, `true_clim.pt`) for reuse.
  - `evaluate_rollout()` computes per-channel **Min bias**, **Max bias**, **Mean bias**, **RMSE** between rollout temporal mean and true climatology. The script currently **does not** write these to CSV; to use `post_processing_metric_table.table_metric()` you can uncomment or add a line like `save_df.to_csv(os.path.join(save_dir, f'evaluation_metrics_seed{seed}.csv'))` in `evaluate_rollout()`.

### Metrics

- **Climatology bias:** Mean absolute difference between rollout temporal mean and true climatology (over the evaluation window).
- **Per-channel (Min/Max/Mean bias, RMSE):** Between the rollout temporal mean and true climatology for temperature, humidity, u_wind, v_wind, surface_pressure, precipitation.

---

## Post-Processing (Tables and Figures)

`post_processing_metric_table.py` provides:

1. **`table_metric()`** — Reads `evaluation_metrics_seed{seed}.csv` from each model folder under a central `save_dir`, aggregates over seeds (mean ± std), applies unit scaling (e.g. humidity ×1000, surface pressure /100, precipitation ×4×1000 for 6-hourly to daily), and writes `total_evaluation_metrics_raw.csv` and `total_evaluation_metrics.csv`.
2. **`plot_bias()`** — Loads rollout and true climatology, converts units, and plots bias (and optionally prediction vs truth) per channel; can use Cartopy for map projection.
3. Other plotting/analysis helpers as in the file.

Set `save_dir` (and any paths) inside the script to your model output root (e.g. `/scratch3/wan410/operator_learning_model/ERA5/`). Ensure the test script writes `evaluation_metrics_seed{seed}.csv` per model/seed if you want `table_metric()` to run without manual CSV creation.

---

## Slurm Scripts

- **`submit_bash.sh`**  
  Single job: uncomment the line for the desired command (train or test) and config. Examples:
  - `python 2d_train_rel_l2.py --config_path config/LUCIE.yaml`
  - `python 2d_train_rel_l2.py --config_path config/HFS_sphere.yaml`
  - `python 2d_train_rel_l2.py --config_path config/MSWT_sphere.yaml --seed 42`
  - `python 2d_test_rel_l2.py --config_path config/MSWT_sphere.yaml --seed 45`
  Edit `#SBATCH` directives as needed for your cluster.

- **`submit_multiseed.sh`**  
  Submits one job per seed (default seeds: 42, 43, 44, 45, 46) for a given config.  
  **Usage:**  
  `./submit_multiseed.sh [config_path]`  
  Example:  
  `./submit_multiseed.sh config/MSWT_sphere.yaml`  
  Each job runs the **test** script with the corresponding seed. Change the script to run training per seed if desired.

Run from the **ERA5** directory so that `config/` and scripts are found. The default config in the script is `configs/MSWT_sphere.yaml`; the actual directory is **`config/`** (no “s”), so use `config/MSWT_sphere.yaml` when calling the script.

---

## Baselines

| Model | Config name | Description |
|-------|-------------|-------------|
| **LUCIE** | `lucie` | Spherical Fourier Neural Operator (spherical harmonics, SHT). |
| **HFS** | `hfs` | High-Frequency Scaling (ResUNet) with spherical grid. |
| **MSWT (sphere)** | `mswt_sphere`, `mswt_patch_sphere`, `mswt_residual_sphere_efficient` | Multiscale wavelet with spherical grid and optional patching/residual. |

Configs: `config/LUCIE.yaml`, `config/HFS_sphere.yaml`, `config/MSWT_sphere.yaml`.

---

## Quick Start Summary

1. **Data:** Obtain ERA5 regridded to the chosen resolution (e.g. 48×96). Run `data_utils/data_preprocessing.py` to generate `era5_T30_preprocessed.npz` (and climatology if needed). Set the data folder in `data_utils/data_utils.py`.
2. **Train:**  
   `python 2d_train_rel_l2.py --config_path config/MSWT_sphere.yaml --seed 42`  
   Training will save checkpoints and the best model (by rollout climatology bias) in `save_dir`.
3. **Test:**  
   `python 2d_test_rel_l2.py --config_path config/MSWT_sphere.yaml --seed 42`  
   Loads the best checkpoint for that seed, runs out-of-sample rollout, and evaluates climatology metrics. Optionally save per-seed CSV for post-processing.
4. **Post-process:** Run `table_metric()` and `plot_bias()` (and any other functions) in `post_processing_metric_table.py` after evaluating all models/seeds; set `save_dir` and ensure evaluation CSVs exist.

For multi-seed evaluation via SLURM: `./submit_multiseed.sh config/MSWT_sphere.yaml`.
