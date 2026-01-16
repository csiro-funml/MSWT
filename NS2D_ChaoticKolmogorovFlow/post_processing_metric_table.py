import pandas as pd
import os
import torch
import numpy as np

if __name__ == "__main__":
    if torch.cuda.is_available():
        save_folder = "/scratch3/wan410/operator_learning_model/NS2D_ChaoticKolmogorovFlow/"
    else:
        save_folder = "logs/NS2D_ChaoticKolmogorovFlow/"
   

    model_name_list = ['FNO', 'PDERefinerUNet', 'WNO', 'SAOT', 'HFS', 'MSWT_patching']
    
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
                # only keep wno with seed 42 and abandon other seeeds for wno
                if model_name == 'WNO':
                    print(df_metric['seed'].values[0])
                    if df_metric['seed'].values[0] != 42:
                        continue
                # rename model with the model_name
                df_metric['model'] = model_name
                total_df_metric = pd.concat([total_df_metric, df_metric], ignore_index=True)

    total_df_metric.to_csv(os.path.join(save_folder, 'total_evaluation_metrics.csv'), index=False)
    total_df_metric = pd.read_csv(os.path.join(save_folder, 'total_evaluation_metrics.csv'))

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
    avg_df = avg_df.reindex(index=['fno2d', 'refiner_unet', 'wno', 'saot', 'hfs', 'multiscale_wavelet2d_periodic_patching'])

    # over the columns by the orders: l2_step{1, 30, 64}, spectral_melr_step{1, 30, T}, enstropy_melr_step{1, 30, T},spectral_meape_step{1, 30, T}, enstropy_meape_step{1, 30, T}
    df = pd.DataFrame()
    for col in ['l2', 'spectral_melr', 'enstropy_melr', 'spectral_meape', 'enstropy_meape']:
        for t in [1, 30, 64]:
            df[f'{col}_step{t}'] = avg_df[f'{col}_step{t}']
    df.to_csv(os.path.join(save_folder, 'avg_evaluation_metrics.csv'))
