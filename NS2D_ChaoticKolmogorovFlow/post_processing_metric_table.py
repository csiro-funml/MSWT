import pandas as pd
import os

if __name__ == "__main__":
    save_folder = "/scratch3/wan410/operator_learning_model/NS2D_ChaoticKolmogorovFlow/"

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
                total_df_metric = pd.concat([total_df_metric, df_metric], ignore_index=True)

    total_df_metric.to_csv(os.path.join(save_folder, 'total_evaluation_metrics.csv'), index=False)


    # group by model, and computed the mean and std of the metrics

    # total_df_metric_grouped = total_df_metric.groupby('model')
    # for model, group in total_df_metric_grouped:
        # print("model", model)
        # print("group shape", group.shape)
        # print("mean", group.mean(axis=0))
        # print("std", group.std(axis=0))
        # exit(-1)
    
    exit(-1)