# NS2D Chaotic Kolmogorov Flow Benchmark

This folder contains the **2D incompressible Navier–Stokes (Chaotic Kolmogorov flow)** benchmark for neural operator learning. Models are trained to predict vorticity **one step ahead** and evaluated via **autoregressive rollout** on a held-out test set, with metrics on relative L² error, spectral energy, and enstrophy.

For full benchmark details, see **“Physics-Informed Neural Operator for Learning Partial Differential Equations”** by Zongyi Li et al. (Section 4: *Experiments — Navier–Stokes Equation. Chaotic Kolmogorov flow*).

---

## Benchmark Setup

### PDE and Data

- **Equation:** 2D incompressible Navier–Stokes in vorticity form  
- **Spatial domain:** \( x \in (0, 2\pi)^2 \) (periodic)  
- **Temporal domain:** \( t \in [0, 0.5] \) (train) / test rollout as in config  
- **Forcing:** \( -4\cos(4x_2) \)  
- **Reynolds number:** 500  
- **Input/output:** vorticity only (no velocity channels)

### Train Set

- **Shape:** `(N, T, X, Y)` — N trajectories, T time steps, X×Y spatial resolution  
- **Source:** [NS_fft_Re500_T4000.npy](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_fft_Re500_T4000.npy)  
- **Size:** 4000 × 64 × 64 × 65 (4k trajectories, 64×64 space, 65 time steps)

### Test Set

- **Shape:** `(N, T, X, Y)`  
- **Source:** [NS_Re500_s256_T100_test.npy](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_Re500_s256_T100_test.npy)  
- **Size:** 100 × 129 × 256 × 256  
- **Usage:** Subsampled in config (e.g. `sub: 4` → 64×64) for autoregressive evaluation over time.

Download the `.npy` files and set the paths in your config(s) under `data.datapath` and `test_data.datapath`.

---

## Project Layout

```
NS2D_ChaoticKolmogorovFlow/
├── configs/
│   ├── linear_used_in_paper/   # Configs used in the paper (linear grid)
│   │   ├── FNO.yaml
│   │   ├── HFS.yaml
│   │   ├── WNO.yaml
│   │   ├── SAOT.yaml
│   │   ├── PDERefinerUNet.yaml
│   │   └── MSWT_patching.yaml
│   ├── periodic/               # Periodic grid variants
│   └── ablations/              # MSWT patching ablations (patch size 1,2,4,8)
├── data_utils/
│   └── datasets.py             # NSLoader2D, train/val split, step-ahead sampling
├── train_operator_AR_rell2_2d.py   # Training script (one-step L2)
├── test_operator_AR_rell2_2d.py    # Evaluation script (AR rollout + metrics)
├── post_processing_metric_table.py # Aggregate metrics, LaTeX tables, figures
├── submit_bash.sh              # Single SLURM job (edit to pick config/seed)
└── submit_multiseed.sh         # Submit one job per seed for a given config
```

Training and testing expect to be run from the **repository root** (parent of `NS2D_ChaoticKolmogorovFlow/`) so that `models/` and `utils/` are on `sys.path`, or with the project root in `PYTHONPATH`.

---

## Configuration (YAML)

Each config has four main sections:

| Section      | Role |
|-------------|------|
| `data`      | Train data path, resolution (`nx`, `nt`), subsampling (`sub`, `sub_t`), `time_interval`, `grid_form`, etc. |
| `model`     | Architecture (FNO, HFS, WNO, SAOT, PDERefiner/UNet, MSWT patching) and its hyperparameters. |
| `train`     | `batchsize`, `epochs`, `milestones`, `base_lr`, `scheduler_gamma`, `save_dir`, `save_name`. |
| `test_data` | Test `.npy` path, resolution, subsampling, `time_interval`, `n_sample`. |

Important details:

- **Grid:** `data.grid_form` is typically `linear` (or `periodic` in `configs/periodic/`). It controls how the 2D spatial grid is built for the model input.
- **Checkpoints:** `train.save_name` is automatically suffixed with `_seed{test_seed}.pt` so each train/val split (seed) gets its own file under `train.save_dir`.
- **Paths:** Update `data.datapath`, `test_data.datapath`, and `train.save_dir` for your environment (e.g. `/scratch3/...` or local paths).

---

## Training

One-step-ahead training: at each step the model sees vorticity at time \(t\) (and grid) and is trained to predict vorticity at \(t+1\) with relative L² loss.

**Command:**

```bash
cd /path/to/multiscale_neural_operator   # repo root
python NS2D_ChaoticKolmogorovFlow/train_operator_AR_rell2_2d.py \
  --config_path NS2D_ChaoticKolmogorovFlow/configs/linear_used_in_paper/FNO.yaml \
  --test_ratio 0.25 \
  --test_seed 42
```

**Options:**

| Argument              | Default | Description |
|-----------------------|--------|-------------|
| `--config_path`       | —      | Path to YAML config (required). |
| `--test_ratio`        | 0.0    | Fraction of data used as validation (e.g. 0.25). |
| `--test_seed`         | 42     | Random seed for train/val split. |
| `--synthetic_samples` | 0      | If &gt;0, use synthetic random data (sanity check). |
| `--resume_training`   | false  | Resume from last checkpoint in `save_dir`. |
| `--resume_ckpt`       | None   | Resume from a specific checkpoint file in `save_dir`. |

Checkpoints are saved under `train.save_dir` as `{save_name}` (with `_seed{test_seed}.pt` appended). Validation L2 is logged and can be written to TensorBoard via the script’s writer.

---

## Evaluation (Test / Autoregressive Rollout)

Evaluation loads a checkpoint and runs **autoregressive rollout** on the test set: start from the first time step, then repeatedly feed the model’s prediction as the next input.

**Command:**

```bash
python NS2D_ChaoticKolmogorovFlow/test_operator_AR_rell2_2d.py \
  --config_path NS2D_ChaoticKolmogorovFlow/configs/linear_used_in_paper/FNO.yaml \
  --test_seed 42
```

- The script uses `test_data` from the config and expects the checkpoint at  
  `{train.save_dir}/{train.save_name}` with `.pt` replaced by `_seed{test_seed}.pt`.
- It computes metrics at fixed time indices (e.g. steps 1, 30, 64) and writes:
  - **Metrics:** under `{save_dir}/evaluation_metrics/` (e.g. CSV per model/seed).
  - **Fields:** under `{save_dir}/saved_plots/` (e.g. ground truth, prediction, error, spectra as `.npz`).

### Metrics

| Metric | Description |
|--------|-------------|
| **Rel L²** | Relative L² error on vorticity. |
| **SMLR**   | Spectral Mean Log-Ratio (energy spectrum). |
| **EMLR**   | Enstrophy Mean Log-Ratio. |
| **SMAE**   | Spectral Mean Absolute Error (energy). |
| **EMAE**   | Enstrophy Mean Absolute Error. |

These are computed at rollout steps 1, 30, and T (e.g. 64) and stored in the CSV.

---

## Post-Processing (Tables and Figures)

`post_processing_metric_table.py` provides:

1. **Aggregate metrics over seeds:** reads per-seed CSVs from each model’s `evaluation_metrics/` folder, groups by model, and computes **mean ± std** across seeds.
2. **LaTeX tables:** formats the aggregated metrics into two tables (e.g. Rel L² / SMLR / EMLR and SMAE / EMAE) and writes `.tex` files.
3. **Figures:** prediction vs ground truth, error fields, and energy/enstrophy spectra at selected steps (using the saved `.npz` in `saved_plots/`).

**Usage:**

- Set `CSV_PATH` (and optionally `save_folder`) inside the script to point to your aggregated CSV and output directory (e.g. `logs/NS2D_ChaoticKolmogorovFlow/` or a scratch path).
- Run the relevant functions (e.g. `aggregate_metric_table()`, `process_metric_table_to_latex()`, `plot_error_energy_supp_fig()`), typically by uncommenting or calling them in `if __name__ == '__main__'` or from a small runner script.

The aggregation step expects a fixed list of model folder names and seeds (e.g. FNO, HFS, WNO, SAOT, PDERefinerUNet, MSWT_patching and seeds 42–46) and writes something like `avg_evaluation_metrics_linear.csv`, which the LaTeX and plotting functions then use.

---

## Slurm Scripts

- **`submit_bash.sh`**  
  Single job: edit the script to choose the config and whether to run training and/or testing. Uncomment the desired `python3 train_...` or `python3 test_...` line(s). Set `#SBATCH` (time, account, GPU, etc.) as needed.

- **`submit_multiseed.sh`**  
  Submits one job per seed (default seeds: 42, 43, 44, 45, 46) for a given config.  
  **Usage:**  
  `./submit_multiseed.sh [config_path] [test_ratio]`  
  Example:  
  `./submit_multiseed.sh configs/linear_used_in_paper/MSWT_patching.yaml 0.25`  
  Each job runs the **test** script (not training) for one seed. Adjust the script if you want to run training per seed instead.

Both scripts assume a SLURM environment (e.g. `module load pytorch/...`, `source $HOME/.venvs/pytorch/bin/activate`). Adapt module and venv lines to your cluster.

---

## Baselines

| Model | Config key / name | Description |
|-------|--------------------|-------------|
| **FNO** | `fno2d` (default if no `name`) | Fourier Neural Operator (2D). |
| **HFS** | `hfs` | High-Frequency Scaling (ResUNet-style). |
| **WNO** | `wno` / `wno2d` | Wavelet Neural Operator. |
| **SAOT** | `saot` / `saot2d` | Space-time attention operator. |
| **PDERefiner / Unet** | `pderefiner` / `refiner_unet` | PDE Refiner. |
| **MSWT** | `multiscale_wavelet2d_periodic_patching` (or `mswt_periodic_patching`) | Multi-Scale Wavelet Transformers |

Configs in `configs/linear_used_in_paper/` and `configs/periodic/` are the main references; `configs/ablations/` varies MSWT patch size (1, 2, 4, 8).

---

## Quick Start Summary

1. **Data:** Download `NS_fft_Re500_T4000.npy` and `NS_Re500_s256_T100_test.npy`; set paths in a config.
2. **Train:**  
   `python NS2D_ChaoticKolmogorovFlow/train_operator_AR_rell2_2d.py --config_path <config> --test_ratio 0.25 --test_seed 42`
3. **Test:**  
   `python NS2D_ChaoticKolmogorovFlow/test_operator_AR_rell2_2d.py --config_path <config> --test_seed 42`
4. **Post-process:** Run `aggregate_metric_table()` then `process_metric_table_to_latex()` (and optional plotting) in `post_processing_metric_table.py` after evaluating all models/seeds.

For multi-seed evaluation via SLURM, use `submit_multiseed.sh` with the desired config (and optionally test ratio).
