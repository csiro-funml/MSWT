import pandas as pd
import os
import torch
import numpy as np
import re
import numpy as np
import pandas as pd
import math
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.criterion import LpLoss
from tqdm import tqdm
from utils.compute_diagnostics import velocity_from_vorticity, compute_spectra_torch, compute_enstropy_torch



def process_metric_table_to_latex():
    # -----------------------------
    # Config
    # -----------------------------
    CSV_PATH = "logs/SW2D_PDA/avg_evaluation_metrics_periodic.csv"   # <-- set this to your local path
    STEPS = [1, 41, 81]

    # Map raw metric names -> display names used in the LaTeX header
    METRIC_DISPLAY = {
        "l2": r"Rel $L^2$",
        "SMLR": "SMLR",
        "EMLR": "EMLR",
        "SMAE": "SMAE",
        "EMAE": "EMAE",
    }

    GROUP_1 = ["l2", "SMLR", "EMLR"]
    GROUP_2 = ["SMAE", "EMAE"]

    # -----------------------------
    # Helpers
    # -----------------------------
    def _parse_float_token(tok: str) -> float:
        """Parse tokens like '0.123', 'nan', 'inf' into float."""
        t = tok.strip().lower()
        if t in {"nan", "+nan", "-nan"}:
            return float("nan")
        if t in {"inf", "+inf", "infty", "infinity", "+infty"}:
            return float("inf")
        if t in {"-inf", "-infty", "-infinity"}:
            return float("-inf")
        return float(t)

    def clean_pm_string(cell) -> str:
        """
        Apply your rules:
        - '0.2190 ± nan' -> '0.2190 ± 0.0000'
        - 'nan ± nan' or 'inf ± nan' (or any non-finite mean/std except finite±nan) -> '--'
        - otherwise: format as '%.4f ± %.4f'
        """
        if pd.isna(cell):
            return "--"

        s = str(cell).strip()
        if "±" not in s:
            return s

        parts = [p.strip() for p in s.split("±")]
        if len(parts) != 2:
            return s

        mean_s, std_s = parts
        try:
            mean = _parse_float_token(mean_s)
            std = _parse_float_token(std_s)
        except Exception:
            return "--"

        # Rule 1: finite mean, std is nan -> std := 0
        if np.isfinite(mean) and (not np.isfinite(std)) and np.isnan(std):
            std = 0.0
            return f"{mean:.4f} ± {std:.4f}"

        # Normal finite case
        if np.isfinite(mean) and np.isfinite(std):
            return f"{mean:.4f} ± {std:.4f}"

        # Rule 2: any other nan/inf combos -> '--'
        return "--"

    def to_latex_cell(s: str) -> str:
        """Convert 'a ± b' -> '$a \\pm b$' and keep '--' unchanged."""
        if s == "--":
            return "--"
        if "±" in s:
            a, b = [x.strip() for x in s.split("±")]
            return rf"${a} \pm {b}$"
        return s

    def build_multiindex_table(df: pd.DataFrame, metrics: list[str], steps: list[int]) -> pd.DataFrame:
        """
        Build a DataFrame with MultiIndex columns: (METRIC_DISPLAY[metric], f"step {step}").
        Assumes input columns look like '{metric}_step{step}' with metric possibly capitalized.
        """
        cols = []
        data = {}
        # Ensure Model column exists and get model names as list
        model_names = df["Model"].tolist()
        for m in metrics:
            for st in steps:
                colname = f"{m}_step{st}"
                if colname not in df.columns:
                    raise KeyError(f"Missing column in CSV: {colname}")
                disp_m = METRIC_DISPLAY[m]
                disp_s = f"step {st}"
                cols.append((disp_m, disp_s))
                # Map to LaTeX format - convert Series to list for proper alignment
                mapped_values = df[colname].map(to_latex_cell).tolist()
                data[(disp_m, disp_s)] = mapped_values

        # Create DataFrame with model names as index
        out = pd.DataFrame(data, index=model_names)
        out.columns = pd.MultiIndex.from_tuples(cols, names=["Metric", "Step"])
        return out

    # -----------------------------
    # Main
    # -----------------------------
    df = pd.read_csv(CSV_PATH)

    # Ensure model column name
    if "Unnamed: 0" in df.columns and "Model" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Model"})

    # Clean every metric cell (mean ± std strings)
    for c in df.columns:
        if c != "Model":
            df[c] = df[c].apply(clean_pm_string)

    # Build the two multi-header tables
    t1 = build_multiindex_table(df, GROUP_1, STEPS)  # Rel L^2, SMLR, EMLR
    t2 = build_multiindex_table(df, GROUP_2, STEPS)  # SMAE, EMAE

    # Export to LaTeX (pandas handles multicolumn headers)
    latex_1 = t1.to_latex(
        escape=False,
        multicolumn=True,
        multicolumn_format="c",
        index=True,
        column_format="l" + "c" * (len(GROUP_1) * len(STEPS)),
        bold_rows=False,
        caption=r"Long-term evaluation metrics (mean $\pm$ std) for \textbf{Rel $L^2$}, \textbf{SMLR}, and \textbf{EMLR} at selected rollout steps.",
        label="tab:metrics_relL2_smlr_emlr",
    )

    latex_2 = t2.to_latex(
        escape=False,
        multicolumn=True,
        multicolumn_format="c",
        index=True,
        column_format="l" + "c" * (len(GROUP_2) * len(STEPS)),
        bold_rows=False,
        caption=r"Long-term evaluation metrics (mean $\pm$ std) for \textbf{SMAE} and \textbf{EMAE} at selected rollout steps.",
        label="tab:metrics_smae_emae",
    )

    # Optional: swap pandas' default rules for booktabs style
    def add_booktabs(tex: str) -> str:
        tex = tex.replace(r"\toprule", r"\toprule")
        tex = tex.replace(r"\midrule", r"\midrule")
        tex = tex.replace(r"\bottomrule", r"\bottomrule")
        return tex

    latex_1 = add_booktabs(latex_1)
    latex_2 = add_booktabs(latex_2)
    save_folder = "logs/SW2D_PDA/"
    with open(os.path.join(save_folder, "table_relL2_smlr_emlr.tex"), "w") as f:
        f.write(latex_1)

    with open(os.path.join(save_folder, "table_smae_emae.tex"), "w") as f:
        f.write(latex_2)

    print("Wrote: table_relL2_smlr_emlr.tex")
    print("Wrote: table_smae_emae.tex")


def aggregate_metric_table(grid_form='linear'):
    if torch.cuda.is_available():
        save_folder = "/scratch3/wan410/operator_learning_model/SW2D_PDA/"
    else:
        save_folder = "logs/SW2D_PDA/"
   

    model_name_list = ['FNO', 'PDERefinerUNet', 'WNO', 'SAOT', 'HFS', 'MSWT_patching']
    renamed_name_list = ['FNO', 'Unet', 'WNO', 'SAOT', 'HFS', 'MSWT']
    # model_name_list = ['FNO', 'MSWT_patching']
    # renamed_name_list = ['FNO', 'MSWT']
    
    seeds = [42, 43, 44, 45, 46]
    total_df_metric = pd.DataFrame()
    for model_name in model_name_list:
        path_folder = os.path.join(save_folder, f'{model_name}2d_{grid_form}/evaluation_metrics')
        file_list = os.listdir(path_folder)
        for file in file_list:
            if file.endswith('.csv'):
                df_metric = pd.read_csv(os.path.join(path_folder, file))
                # print(df_metric.head())
                # df_metric = df.iloc[0]
                # only keep wno with seed 42 and abandon other seeeds for wno because their metrics are not stable
                # if model_name == 'WNO':
                #     print(df_metric['seed'].values[0])
                #     if df_metric['seed'].values[0] != 42:
                #         continue
                # rename model with the model_name
                df_metric['model'] = model_name
                total_df_metric = pd.concat([total_df_metric, df_metric], ignore_index=True)

    total_df_metric.to_csv(os.path.join(save_folder, f'total_evaluation_metrics_{grid_form}.csv'), index=False)
    # total_df_metric = pd.read_csv(os.path.join(save_folder, f'total_evaluation_metrics_{grid_form}.csv'))

    # group by model, and computed the mean and std of the metrics
    
    total_df_metric_grouped = total_df_metric.groupby('model')
    avg = {}
    for model_name, model_metric in total_df_metric_grouped:
        # print("model", model.shape)
        print("model_name", model_name)
        print("group shape", model_metric.shape)
        
        # Select only numeric columns (exclude 'seed' and 'model')
        numeric_cols = [col for col in model_metric.columns if col not in ['seed', 'model']]
        numeric_data = model_metric[numeric_cols]
        
        # Convert to numeric, coercing errors to NaN (handles empty strings from CSV)
        numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
        
        # Compute mean and std (pandas handles NaN properly)
        model_mean = numeric_data.mean(axis=0).values
        model_std = numeric_data.std(axis=0).values
        
        # I want to save the metrics as "mean ± std" with 4 decimal places
        model_mean_str = {metric: f"{mean:.4f} ± {std:.4f}" for metric, mean, std in zip(numeric_cols, model_mean, model_std) }
        avg[model_name] = model_mean_str
    
    # save the avg to a csv
    avg_df = pd.DataFrame(avg).T
    # order the index by the order of ['fno2d', 'refiner_unet', 'wno', 'saot', 'hfs', 'multisacle_wavelet2d_periodic_patching']
    avg_df = avg_df.reindex(index=model_name_list)
    # RENAME the index by the renamed_name_list
    avg_df.index = renamed_name_list
    # # over the columns by the orders: l2_step{1, 30, 64}, spectral_melr_step{1, 30, T}, enstropy_melr_step{1, 30, T},spectral_meape_step{1, 30, T}, enstropy_meape_step{1, 30, T}
    # # reorder the columns by the orders: l2_step1,l2_step30,l2_step64, spectral_melr_step1,spectral_melr_step30,spectral_melr_step64, enstropy_melr_step1,enstropy_melr_step30,enstropy_melr_step64,spectral_meape_step1,spectral_meape_step30,spectral_meape_step64, enstropy_meape_step1,enstropy_meape_step30,enstropy_meape_step64

    
    # avg_df = avg_df[new_columns]
    print(avg_df)
    avg_df.to_csv(os.path.join(save_folder, f'avg_evaluation_metrics_{grid_form}.csv'))


def compute_save_energy_spectra(seq):
    """
    truth_seq: (B, H, W, T)
    pred_seq: (B, H, W, T)
    time_indices: list of time indices
    save_dir: directory to save the energy spectra
    model_name: name of the model
    seed: seed for the random test split
    """

    truth_frame = seq.cpu()
    
    # Spectral energy comparison 
    ux_true, uy_true = velocity_from_vorticity(truth_frame.float())
    

    k_bins, Ek_true = compute_spectra_torch(ux_true, uy_true, 2 * math.pi, 2 * math.pi)
    
    _, Zk_true = compute_enstropy_torch(truth_frame.float(), 2 * math.pi, 2 * math.pi)
    
    k_np = k_bins.detach().cpu().numpy()
    valid_mask = range(1, min(len(k_np), min(truth_frame.shape[-1], truth_frame.shape[-2]) // 2))
    # valid_mask = range(1, len(k_np))
    k_np = k_np[valid_mask]
    Ek_true_np = Ek_true.detach().cpu().numpy()[valid_mask]
    Zk_true_np = Zk_true.detach().cpu().numpy()[valid_mask]
    return k_np, Ek_true_np, Zk_true_np


def load_pred_truth_error(model_folder_name, saved_model_name, seed, step, save_folder, grid_form):
    
    pred_path = os.path.join(save_folder, f'{model_folder_name}2d_{grid_form}/saved_plots', f'{saved_model_name}_seed{seed}_prediction_t{step}.npz')
    energy_path = os.path.join(save_folder, f'{model_folder_name}2d_{grid_form}/saved_plots', f'{saved_model_name}_seed{seed}_energy_spectra_t{step}.npz')
    
    pred_data = np.load(pred_path)
    print("pred_data shape", pred_data['initial_condition'].shape, pred_data['pred_seq_t'].shape, pred_data['truth_seq_t'].shape, pred_data['error_seq_t'].shape)
    initial_condition = pred_data['initial_condition'][..., 0 ] # (shape: N, T, H, W) get the first channel for vorticity
    pred = pred_data['pred_seq_t']
    truth = pred_data['truth_seq_t']
    error = pred_data['error_seq_t']

    print("initial_condition shape", initial_condition.shape)
    print("pred shape", pred.shape)
    print("truth shape", truth.shape)
    print("error shape", error.shape)

    # compute the l2 error
    # l2_loss = LpLoss(size_average=True)
    # l2_err = l2_loss(torch.from_numpy(pred).unsqueeze(0), torch.from_numpy(truth).unsqueeze(0)).item()
    
    return initial_condition, pred, truth, error


def plot_error():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # 2 rows ,6 columns
    # the first column is ground truth and energy plot,
    # from the secon column, shows the prediction and error plot
    # generate the color map from below
    if torch.cuda.is_available():
        save_folder = "/scratch3/wan410/operator_learning_model/SW2D_PDA/"
    else:
        save_folder = "logs/SW2D_PDA/"
    # steps = [1, 41, 87]
    # steps = range(0, 88, 20)
    steps = [0, 40, 80]
    # steps = [80]
    dataset_name = 'SW2D_PDA'
    model_name_list = ['FNO', 'PDERefinerUNet', 'SAOT', 'HFS', 'MSWT_patching']
    saved_model_name_list = ['fno2d', 'refiner_unet', 'saot', 'hfs', 'multiscale_wavelet2d_periodic_patching']
    plot_model_name_list = ['FNO', 'Unet', 'SAOT', 'HFS', 'MSWT']
    # model_name_list = ['FNO', 'HFS','MSWT_patching']
    # saved_model_name_list = ['fno2d', 'hfs', 'multiscale_wavelet2d_periodic_patching']
    # plot_model_name_list = ['FNO', 'HFS', 'MSWT']
    # full_energy_model_name_list = ['FNO', 'PDERefinerUNet', 'SAOT', 'HFS', 'MSWT_patching']
    seed = 42
    # grid_form = 'linear'
    grid_form = 'periodic'
    sample_idx = 0
    # I want to iterate over the sample indices from 0 to 99 as well
    for step in steps:

        pred_dict = {}
        error_dict = {}

        energy_dict = {}
        enstropy_dict = {}
        global_max = float('-inf')
        global_min = float('inf')
        error_max = float('-inf')
        error_min = float('inf')
        for i, model_name in enumerate(model_name_list):
            initial_condition, pred, truth, error= \
            load_pred_truth_error(model_name, saved_model_name_list[i], seed, step+1, save_folder, grid_form)
             # (N , H, W)
            pred_dict[model_name] = pred[sample_idx]
            error_dict[model_name] = error[sample_idx]

            # compute the energy spectra and enstropy spectra here
            k_np, energy_dict[model_name], enstropy_dict[model_name] = compute_save_energy_spectra(torch.from_numpy(pred[sample_idx]))

            global_max = max(global_max, initial_condition[sample_idx].max(), truth[sample_idx].max(), pred_dict[model_name].max())
            global_min = min(global_min, initial_condition[sample_idx].min(), truth[sample_idx].min(), pred_dict[model_name].min())
            error_max = max(error_max, error_dict[model_name].max())
            error_min = min(error_min, error_dict[model_name].min())
        
        initial_condition = initial_condition[sample_idx]
        truth = truth[sample_idx]
        _, energy_dict['truth'], enstropy_dict['truth'] = compute_save_energy_spectra(torch.from_numpy(truth))

        global_max = max(global_max, np.abs(global_min))
        global_min = -global_max # make it symmetrical around zero
        error_max = max(error_max, np.abs(error_min))
        error_min = -error_max # make it symmetrical around zero

        # fig, axes = plt.subplots(2, 5, figsize=(12, 3), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
        fig, axes = plt.subplots(2, 10, figsize=(12, 10), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
        # plot the truth first at axes [0, 0]
        ax = axes[0, 0]
        im = ax.imshow(initial_condition, cmap='RdBu_r', origin='lower', vmin=global_min, vmax=global_max)
        ax.set_title('Initial Condition', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = axes[0, 1]
        im = ax.imshow(truth, cmap='RdBu_r', origin='lower', vmin=global_min, vmax=global_max)
        ax.set_title('Ground Truth', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # # make the axis disapprear completly
        ax = axes[1, 0]
        ax.axis('off')

        ax = axes[1, 1]
        ax.axis('off')          

        for col_idx, model_name in enumerate(model_name_list):
            ax = axes[0, col_idx+2]
            im = ax.imshow(pred_dict[model_name], cmap='RdBu_r', origin='lower', vmin=global_min, vmax=global_max)
            ax.set_title(f'{plot_model_name_list[col_idx]} Prediction', fontsize=10, fontweight='bold')
            # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[1, col_idx+2]   
            im = ax.imshow(error_dict[model_name], cmap='RdBu_r', origin='lower', vmin=error_min, vmax=error_max)
            ax.set_title(f'{plot_model_name_list[col_idx]} Error', fontsize=10, fontweight='bold')
            # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # ax.set_title(f'(Rel $L^2$: {l2_err_dict[model_name]:.2f})', fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        

        cbar_ax = inset_axes(axes[1, 1], width="8%", height="70%", loc='center',
                             borderpad=0)
        cbar = plt.colorbar(im, cax=cbar_ax, aspect=15)
        cbar.set_label('Error', fontsize=10, fontweight='bold', rotation=90, labelpad=10)
        # Set ticks on the right side and make them bold and larger
        cbar.ax.yaxis.set_label_position('right')
        cbar.ax.yaxis.tick_left()
        cbar.ax.tick_params(labelsize=11, width=1.2, length=5)
        # Make tick labels bold
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')  
        
        plt.tight_layout(rect=[0, 0, 1, 1])
        # save_folder_error = os.path.join(save_folder, 'plot_error')
        save_folder_error = save_folder
        os.makedirs(save_folder_error, exist_ok=True)
        # plt.savefig(os.path.join(save_folder_error, f'{dataset_name}_pred_error_grid_{grid_form}_t{step}_seed{seed}_instance_{sample_idx}.png'), dpi=500, bbox_inches='tight')
        plt.close(fig)
            
               
        # plot the energy spectra and enstropy spectra
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
        # Highlight Ground Truth (first) and MSWT_patching (last) with bold colors and solid lines
        # Ground Truth: bold orange/red; MSWT_patching: bold purple
        # Middle models: muted colors with dashes
        color_list = ['#E65100', '#6BAED6', '#969696', '#FDB462', '#74C476', '#7B1FA2']  # orange-red, light blue, gray, peach, light green, bold purple
        linestyle_list = ['-', '--', '-.', ':', '--', '--']  # Solid for Ground Truth and MSWT_patching
        ax.loglog(k_np, energy_dict['truth'], label='Ground Truth', linewidth=3, color=color_list[0], linestyle=linestyle_list[0])
        for i, model_name in enumerate(model_name_list):
            ax.loglog(k_np, energy_dict[model_name], label=f'{plot_model_name_list[i]}', linewidth=2 if model_name != 'MSWT_patching' else 3, color=color_list[i+1], linestyle=linestyle_list[i+1])
        ax.set_xlabel('Wavenumber k', fontsize=20)
        ax.set_ylabel('Spectral Energy Spectrum E(k)', fontsize=20)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.legend(fontsize=20, loc='lower left')
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(os.path.join(save_folder, f'{dataset_name}_spectral_energy_spectrum_grid_{grid_form}_t{step}_seed{seed}.png'), dpi=500, bbox_inches='tight')


        fig, ax = plt.subplots(1, 1, figsize=(6, 6), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
        ax.loglog(k_np, enstropy_dict['truth'], label='Ground Truth', linewidth=2, color=color_list[0], linestyle=linestyle_list[0])
        for i, model_name in enumerate(model_name_list):
            ax.loglog(k_np, enstropy_dict[model_name], label=f'{plot_model_name_list[i]}', linewidth=2 if model_name != 'MSWT_patching' else 2, color=color_list[i+1], linestyle=linestyle_list[i+1])
        ax.set_xlabel('Wavenumber k', fontsize=20)
        ax.set_ylabel('Enstropy Z(k)', fontsize=20)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.legend(fontsize=20, loc='lower left')
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(os.path.join(save_folder, f'{dataset_name}_enstropy_spectrum_grid_{grid_form}_t{step}_seed{seed}.png'), dpi=500, bbox_inches='tight')

if __name__ == "__main__":
    # aggregate_metric_table(grid_form='linear')
    # aggregate_metric_table(grid_form='periodic')
    # process_metric_table_to_latex()

    plot_error()