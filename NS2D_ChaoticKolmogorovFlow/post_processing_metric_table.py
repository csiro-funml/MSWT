import pandas as pd
import os
import torch
import numpy as np
import re
import numpy as np
import pandas as pd


def process_metric_table(CSV_PATH):
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


if __name__ == "__main__":
    if torch.cuda.is_available():
        save_folder = "/scratch3/wan410/operator_learning_model/NS2D_ChaoticKolmogorovFlow/"
    else:
        save_folder = "logs/NS2D_ChaoticKolmogorovFlow/"
   

    model_name_list = ['FNO', 'PDERefinerUNet', 'WNO', 'SAOT', 'HFS', 'MSWT_patching']
    renamed_name_list = ['FNO', 'Unet', 'WNO', 'SAOT', 'HFS', 'MSWT']
    
    seeds = [42, 43, 44, 45, 46]
    total_df_metric = pd.DataFrame()
    for model_name in model_name_list:
        path_folder = os.path.join(save_folder, f'{model_name}2d_linear/evaluation_metrics')
        file_list = os.listdir(path_folder)
        for file in file_list:
            if file.endswith('.csv'):
                df_metric = pd.read_csv(os.path.join(path_folder, file))
                # print(df_metric.head())
                # df_metric = df.iloc[0]
                # only keep wno with seed 42 and abandon other seeeds for wno because their metrics are not stable
                if model_name == 'WNO':
                    print(df_metric['seed'].values[0])
                    if df_metric['seed'].values[0] != 42:
                        continue
                # rename model with the model_name
                df_metric['model'] = model_name
                total_df_metric = pd.concat([total_df_metric, df_metric], ignore_index=True)

    total_df_metric.to_csv(os.path.join(save_folder, 'total_evaluation_metrics.csv'), index=False)
    # total_df_metric = pd.read_csv(os.path.join(save_folder, 'total_evaluation_metrics.csv'))

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
    avg_df.to_csv(os.path.join(save_folder, 'avg_evaluation_metrics.csv'))
