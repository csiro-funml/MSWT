import pandas as pd
import os
import torch
import numpy as np
import re
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.criterion import LpLoss


def process_metric_table_to_latex(CSV_PATH):
    # -----------------------------
    # Config
    # -----------------------------
    CSV_PATH = "avg_evaluation_metrics.csv"   # <-- set this to your local path
    STEPS = [1, 30, 64]

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
        for m in metrics:
            for st in steps:
                colname = f"{m}_step{st}"
                if colname not in df.columns:
                    raise KeyError(f"Missing column in CSV: {colname}")
                disp_m = METRIC_DISPLAY[m]
                disp_s = f"step {st}"
                cols.append((disp_m, disp_s))
                data[(disp_m, disp_s)] = df[colname].map(to_latex_cell)

        out = pd.DataFrame(data, index=df["Model"])
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

    with open("table_relL2_smlr_emlr.tex", "w") as f:
        f.write(latex_1)

    with open("table_smae_emae.tex", "w") as f:
        f.write(latex_2)

    print("Wrote: table_relL2_smlr_emlr.tex")
    print("Wrote: table_smae_emae.tex")


def aggregate_metric_table(grid_form='linear'):
    if torch.cuda.is_available():
        save_folder = "/scratch3/wan410/operator_learning_model/SW2D_PDA/"
    else:
        save_folder = "logs/SW2D_PDA/"
   

    # model_name_list = ['FNO', 'PDERefinerUNet', 'WNO', 'SAOT', 'HFS', 'MSWT_patching']
    # renamed_name_list = ['FNO', 'Unet', 'WNO', 'SAOT', 'HFS', 'MSWT']
    model_name_list = ['FNO', 'MSWT_patching']
    renamed_name_list = ['FNO', 'MSWT']
    
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




def load_pred_truth_error_spectral(model_folder_name, saved_model_name, seed, step, save_folder, grid_form):
    
    pred_path = os.path.join(save_folder, f'{model_folder_name}2d_{grid_form}/saved_plots', f'{saved_model_name}_seed{seed}_prediction_t{step}.npz')
    energy_path = os.path.join(save_folder, f'{model_folder_name}2d_{grid_form}/saved_plots', f'{saved_model_name}_seed{seed}_energy_spectra_t{step}.npz')
    
    pred_data = np.load(pred_path)
    use_sample_idx = 1
    print("pred_data shape", pred_data['initial_condition'].shape, pred_data['pred_seq_t'].shape, pred_data['truth_seq_t'].shape, pred_data['error_seq_t'].shape)
    initial_condition = pred_data['initial_condition'][use_sample_idx, ..., 0] # (shape: N, C, H, W)
    pred = pred_data['pred_seq_t'][use_sample_idx]
    truth = pred_data['truth_seq_t'][use_sample_idx]
    error = pred_data['error_seq_t'][use_sample_idx]

    energy_data = np.load(energy_path)
    spectral_pred = energy_data['Ek_pred_np'][use_sample_idx]
    enstropy_pred = energy_data['Zk_pred_np'][use_sample_idx]
    spectral_true = energy_data['Ek_true_np'][use_sample_idx]
    enstropy_true = energy_data['Zk_true_np'][use_sample_idx]
    k_np = energy_data['k_np'][use_sample_idx]

    print("initial_condition shape", initial_condition.shape)
    print("pred shape", pred.shape)
    print("truth shape", truth.shape)
    print("error shape", error.shape)
    print("spectral_pred shape", spectral_pred.shape)
    print("enstropy_pred shape", enstropy_pred.shape)
    print("spectral_true shape", spectral_true.shape)
    print("enstropy_true shape", enstropy_true.shape)
    print("k_np shape", k_np.shape)

    # compute the l2 error
    l2_loss = LpLoss(size_average=True)
    l2_err = l2_loss(torch.from_numpy(pred).unsqueeze(0), torch.from_numpy(truth).unsqueeze(0)).item()
    
    return initial_condition, pred, truth, error, k_np, spectral_pred, spectral_true, enstropy_pred, enstropy_true, l2_err


def plot_error_energy():
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
    steps = [1, 41, 87]
    dataset_name = 'SW2D_PDA'
    # model_name_list = ['FNO', 'PDERefinerUNet', 'SAOT', 'HFS', 'MSWT_patching']
    # saved_model_name_list = ['fno2d', 'refiner_unet', 'saot', 'hfs', 'multiscale_wavelet2d_periodic_patching']
    # plot_model_name_list = ['FNO', 'Unet', 'SAOT', 'HFS', 'MSWT']
    model_name_list = ['FNO', 'MSWT_patching']
    saved_model_name_list = ['fno2d', 'multiscale_wavelet2d_periodic_patching']
    plot_model_name_list = ['FNO', 'MSWT']
    seed = 42
    # grid_form = 'linear'
    grid_form = 'periodic'
    for step in steps:
        fig, axes = plt.subplots(2, 7, figsize=(12, 3), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
        pred_dict = {}
        error_dict = {}
        l2_err_dict = {}
        k_np_dict = {}
        spectral_pred_dict = {}
        enstropy_pred_dict = {}
        global_max = float('-inf')
        global_min = float('inf')
        error_max = float('-inf')
        error_min = float('inf')
        truth = None  # Will be set once from first model
        k_np = None  # Will be set once from first model
        spectral_true = None  # Will be set once from first model
        
        for i, model_name in enumerate(model_name_list):

            initial_condition, pred, truth, error, k_np, spectral_pred, spectral_true, enstropy_pred, enstropy_true, l2_err = \
            load_pred_truth_error_spectral(model_name, saved_model_name_list[i], seed, step, save_folder, grid_form)
            
            pred_dict[model_name] = pred
            error_dict[model_name] = error
            k_np_dict[model_name] = k_np
            l2_err_dict[model_name] = l2_err
            spectral_pred_dict[model_name] = spectral_pred
            enstropy_pred_dict[model_name] = enstropy_pred
            
            
            error_max = max(error_max, error.max().item())
            error_min = min(error_min, error.min().item())
        

        global_max =  max(truth.max().item(), initial_condition.max().item()) * 1.1 # allow  10% more than the truth max
        global_min = min(truth.min().item(), initial_condition.min().item()) * 1.1 # allow 10% less than the truth min
        
        # plt.savefig(os.path.join(save_folder, f'pred_error_spectral_grid_{grid_form}_t{step}.png'), dpi=150, bbox_inches='tight')
        global_max = max(global_max, np.abs(global_min))
        global_min = -global_max # make it symmetrical around zero
        error_max = max(error_max, np.abs(error_min))
        error_min = -error_max # make it symmetrical around zero
        
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
        
        # make the axis disapprear completly
        ax = axes[1, 0]
        ax.axis('off')

        ax = axes[1, 1]
        ax.axis('off')
        
        
        for i, model_name in enumerate(model_name_list):
            ax = axes[0, i+2]
            im = ax.imshow(pred_dict[model_name], cmap='RdBu_r', origin='lower', vmin=global_min, vmax=global_max)
            ax.set_title(f'{plot_model_name_list[i]}', fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = axes[1, i+2]
            im = ax.imshow(error_dict[model_name], cmap='RdBu_r', origin='lower', vmin=error_min, vmax=error_max)
            # im = ax.imshow(error_dict[model_name], cmap='RdBu_r', origin='lower', vmin=error_min, vmax=error_max)
            ax.set_title(f'(Rel $L^2$: {l2_err_dict[model_name]:.2f})', fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
        # add colorbar of error min and max centered inside axes[1, 0]
        # Position colorbar in the center-right of the subplot to avoid blocking adjacent axes
        # Using inset_axes to place colorbar inside the plot area
        # put some texts "rollout %d steps" in the middle of axes[1, 0]
        # ax = axes[1, 0]
        # ax.text(0.5, 0.5, f'rollout {step} \n time step'+('s' if step > 1 else ''), fontsize=10, fontweight='bold', ha='center', va='center')
        # ax.axis('off')

        cbar_ax = inset_axes(axes[1, 1], width="4%", height="90%", loc='center',
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
        plt.savefig(os.path.join(save_folder, f'{dataset_name}_pred_error_spectral_grid_{grid_form}_t{step}_seed{seed}.png'), dpi=500, bbox_inches='tight')
        
        

        # plot the spectral energy and enstropy spectrum
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
        # Highlight Ground Truth (first) and MSWT_patching (last) with bold colors and solid lines
        # Ground Truth: bold orange/red; MSWT_patching: bold purple
        # Middle models: muted colors with dashes
        color_list = ['#E65100', '#6BAED6', '#969696', '#FDB462', '#74C476', '#7B1FA2']  # orange-red, light blue, gray, peach, light green, bold purple
        linestyle_list = ['-', '--', '-.', ':', '--', '--']  # Solid for Ground Truth and MSWT_patching
        ax.loglog(k_np, spectral_true, label='Ground Truth', linewidth=3, color=color_list[0], linestyle=linestyle_list[0])
        for i, model_name in enumerate(model_name_list):
            ax.loglog(k_np_dict[model_name], spectral_pred_dict[model_name], label=f'{plot_model_name_list[i]}', linewidth=2 if model_name != 'MSWT_patching' else 3, color=color_list[i+1], linestyle=linestyle_list[i+1])
        ax.set_xlabel('Wavenumber k', fontsize=20)
        ax.set_ylabel('Spectral Energy Spectrum E(k)', fontsize=20)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.legend(fontsize=20, loc='lower left')
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(os.path.join(save_folder, f'{dataset_name}_spectral_energy_spectrum_grid_{grid_form}_t{step}_seed{seed}.png'), dpi=500, bbox_inches='tight')


        fig, ax = plt.subplots(1, 1, figsize=(6, 6), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
        ax.loglog(k_np, enstropy_true, label='Ground Truth', linewidth=2, color=color_list[0], linestyle=linestyle_list[0])
        for i, model_name in enumerate(model_name_list):
            ax.loglog(k_np_dict[model_name], enstropy_pred_dict[model_name], label=f'{plot_model_name_list[i]}', linewidth=2 if model_name != 'MSWT_patching' else 2, color=color_list[i+1], linestyle=linestyle_list[i+1])
        ax.set_xlabel('Wavenumber k', fontsize=20)
        ax.set_ylabel('Enstropy Z(k)', fontsize=20)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.legend(fontsize=20, loc='lower left')
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(os.path.join(save_folder, f'{dataset_name}_enstropy_spectrum_grid_{grid_form}_t{step}_seed{seed}.png'), dpi=500, bbox_inches='tight')
    # pred_frame = pred[..., t_raw]
    # truth_frame = truth[..., t_raw]
    # err_frame = pred_frame - truth_frame
    # truth_min = truth_frame.min().item()
    # truth_max = truth_frame.max().item()
    # abs_lim = max(abs(truth_min), abs(truth_max))
    # vmin = -abs_lim
    # vmax = abs_lim

    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # titles = ['Truth', 'Prediction', 'Error']
    # data_to_plot = [truth_frame, pred_frame, err_frame]
    # for ax, title, data in zip(axes, titles, data_to_plot):
    #     if title in ['Truth', 'Prediction']:
    #         im = ax.imshow(data.numpy(), cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    #     else:
    #         err_abs = max(abs(data.min().item()), abs(data.max().item()), 1e-8)
    #         im = ax.imshow(data.numpy(), cmap='RdBu_r', origin='lower', vmin=-err_abs, vmax=err_abs)
    #     ax.set_title(f'{title} (T={t_raw})')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # plt.tight_layout()
    # pred_plot_path = os.path.join(pred_dir, f'ns_prediction_t{t_raw}.png')
    # fig.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
    # plt.close(fig)

    # # Spectral energy comparison
    
    # ux_pred, uy_pred = velocity_from_vorticity(pred_frame.float())
    # ux_true, uy_true = velocity_from_vorticity(truth_frame.float())
    # k_bins, Ek_pred = compute_spectra_torch(ux_pred, uy_pred, 2 * math.pi, 2 * math.pi)
    # _, Ek_true = compute_spectra_torch(ux_true, uy_true, 2 * math.pi, 2 * math.pi)

    # k_np = k_bins.cpu().numpy()
    # Ek_pred_np = Ek_pred.cpu().numpy()
    # Ek_true_np = Ek_true.cpu().numpy()

    # valid_mask = range(1, min(len(k_np), S_data // 2))
    # fig_spec, ax_spec = plt.subplots(1, 1, figsize=(6, 4))
    # ax_spec.loglog(k_np[valid_mask], Ek_true_np[valid_mask], label='Truth', linewidth=1)
    # ax_spec.loglog(k_np[valid_mask], Ek_pred_np[valid_mask], '--', label='Prediction', linewidth=1)
    # ax_spec.set_xlabel('Wavenumber k')
    # ax_spec.set_ylabel('Energy E(k)')
    # ax_spec.set_title(f'Spectral Energy (T={t_raw})')
    # ax_spec.grid(True, which='both', alpha=0.3)
    # ax_spec.legend()
    # spec_plot_path = os.path.join(spec_dir, f'ns_spectral_energy_t{t_raw}.png')
    # fig_spec.savefig(spec_plot_path, dpi=150, bbox_inches='tight')
    # plt.close(fig_spec)
    # return 



if __name__ == "__main__":
    # aggregate_metric_table(grid_form='linear')
    # aggregate_metric_table(grid_form='periodic')
    plot_error_energy()