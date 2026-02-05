import os
import numpy as np
import pandas as pd
import torch
import math
from utils.compute_diagnostics import compute_spectra_torch, compute_enstropy_torch

# Optional: cartopy for coastlines / labeled gridlines. Falls back to plain matplotlib if unavailable.
try:  # pragma: no cover
    import cartopy.crs as ccrs  # type: ignore
    import cartopy.feature as cfeature  # type: ignore

    HAVE_CARTOPY = True
except Exception:  # pragma: no cover
    HAVE_CARTOPY = False

# Shared channel units (reused in multiple plots)
channel_unit_list = ['K', 'g/kg', 'm/s', 'm/s', 'hPa', 'mm/d']




def table_metric():
    model_list = ['MSWT_patch_sphere', 'HFS_sphere', 'LUCIE']
    # model_list = ['LUCIE']

    channel_list = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'precipitation']
    seeds = [42, 43, 44, 45, 46]

    save_dir = '/scratch3/wan410/operator_learning_model/ERA5/'
    total_df = pd.DataFrame()
    for idx, model in enumerate(model_list):
        for seed in seeds:
            save_path = os.path.join(save_dir, model, f'evaluation_metrics_seed{seed}.csv')
            df_model = pd.read_csv(save_path, index_col=0)
            # convert the index of df_model to be a new column and name it as "channel"
            df_model.reset_index(inplace=True)
            df_model.rename(columns={'index': 'channel'}, inplace=True)
            # add a new column to df_model to indicate the model name
            df_model['model'] = model
            # add a new column to df_model to indicate the seed
            df_model['seed'] = seed
            total_df = pd.concat([total_df, df_model], axis=0)
            
    print(total_df)
    total_df.to_csv(os.path.join(save_dir, 'total_evaluation_metrics_raw.csv'))
    # times humidity by 1000, and divide surface pressure by 100
    new_df = pd.DataFrame()
    for channel_value, df_channel in total_df.groupby('channel'):
        for model_value, df_model in df_channel.groupby('model'):
            if channel_value == 'humidity':
                df_model.iloc[:, 1:5] = df_model.iloc[:, 1:5] * 1000 # (min, max, bias)
            elif channel_value == 'surface_pressure':
                df_model.iloc[:, 1:5] = df_model.iloc[:, 1:5] / 100 # (min, max, bias)
            elif channel_value == 'precipitation':
                df_model.iloc[:, 1:5] = df_model.iloc[:, 1:5] * 4 * 1000 # 6hourly data to daily data (min, max, bias)
            print("df_model.shape: ", df_model.shape)
            df_model_mean = df_model.iloc[:, 1:5].mean(axis=0) # from column 1 to 4 
            df_model_std = df_model.iloc[:, 1:5].std(axis=0)
            print("df_model_mean.shape, df_model_std.shape: ", df_model_mean.shape, df_model_std.shape)
            
            # Create a new DataFrame with mean +/- std format for each column
            metric_columns = df_model.iloc[:, 1:5].columns.tolist()  # Get column names
            df_model_mean_std = pd.DataFrame()
            # Format each column as "mean +/- std"
            for col in metric_columns:
                mean_val = df_model_mean[col]
                std_val = df_model_std[col]
                df_model_mean_std[col] = [f"{mean_val:.4f} ± {std_val:.4f}"]
            
            # Add model and channel columns
            df_model_mean_std['model'] = model_value
            df_model_mean_std['channel'] = channel_value
            new_df = pd.concat([new_df, df_model_mean_std], axis=0)
    print(new_df)
    new_df.to_csv(os.path.join(save_dir, 'total_evaluation_metrics.csv'))
    

def plot_bias():
    import matplotlib.pyplot as plt

    # model_list = ['MSWT_sphere', 'HFS_sphere', 'LUCIE']
    # saved_model_name =['mswt_sphere', 'hfs', 'lucie']
    # plot_model_name = ['MSWT', 'HFS', 'LUCIE']
    model_list = ['LUCIE', 'MSWT_patch_sphere']
    saved_model_name =['lucie', 'mswt_patch_sphere']
    plot_model_name = ['LUCIE', 'MSWT']
    channel_list = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'precipitation']
    # channel_unit_list = ['K', 'g/kg', 'm/s', 'm/s', 'hPa', 'mm/d']
    seeds = [42, 43, 44, 45, 46]

    save_dir = '/scratch3/wan410/operator_learning_model/ERA5/'
    true_clim_data = torch.load(os.path.join(save_dir, 'LUCIE', 'true_clim.pt'))
    true_clim_data[1] = true_clim_data[1] * 1000 # (C, H, W) humidity
    true_clim_data[4] = true_clim_data[4] / 100 # (C, H, W) surface pressure 
    true_clim_data[5] = true_clim_data[5] * 4 * 1000 # 6hourly data to daily data (C, H, W) precipitation
               
    # use different diverging color maps for each channel (all with a light/white-ish center)
    color_map_list = ['coolwarm', 'BrBG', 'PiYG', 'PRGn', 'PuOr', 'RdBu_r']


    # load data once for all channels:
    pred_dict = {'temperature': {}, 'humidity': {}, 'u_wind': {}, 'v_wind': {}, 'surface_pressure': {}, 'precipitation': {}}
    error_dict = {'temperature': {}, 'humidity': {}, 'u_wind': {}, 'v_wind': {}, 'surface_pressure': {}, 'precipitation': {}}
    for model_idx, model in enumerate(model_list):
        pred_seed_list = []
        error_seed_list = []
        for seed in seeds:
            pred_dict_model_seed = torch.load(os.path.join(save_dir, model, f'rollout_model_{saved_model_name[model_idx]}_seed{seed}.pt')).mean(dim=0) # (C, H, W)
            pred_dict_model_seed[1] = pred_dict_model_seed[1] * 1000 # (C, H, W) humidity
            pred_dict_model_seed[4] = pred_dict_model_seed[4] / 100 # (C, H, W) surface pressure 
            pred_dict_model_seed[5] = pred_dict_model_seed[5] * 4 * 1000 # (C, H, W) precipitation

            error_dict_model_seed = pred_dict_model_seed - true_clim_data # (C, H, W)
            pred_seed_list.append(pred_dict_model_seed)
            error_seed_list.append(error_dict_model_seed)
        
        pred_mean_all = torch.stack(pred_seed_list).mean(dim=0) # average over seeds(C, H, W)
        print(f'{model} pred mean all shape: {pred_mean_all.shape}')
        error_mean_all = torch.stack(error_seed_list).mean(dim=0) # average over seeds(C, H, W)
        print(f'{model} error mean all shape: {error_mean_all.shape}')
        
        for channel_idx, channel in enumerate(channel_list):
            pred_dict[channel][plot_model_name[model_idx]] = pred_mean_all[channel_idx].cpu().numpy() # (H, W)
            error_dict[channel][plot_model_name[model_idx]] = error_mean_all[channel_idx].cpu().numpy() # (H, W)
            print(f'{channel} pred shape: {pred_dict[channel][plot_model_name[model_idx]].shape}')
            print(f'{channel} error shape: {error_dict[channel][plot_model_name[model_idx]].shape}')
            
    # iterate overall all channels and models to get the error range (90 percentile)
    error_range_dict = {'temperature': {}, 'humidity': {}, 'u_wind': {}, 'v_wind': {}, 'surface_pressure': {}, 'precipitation': {}}
    for channel in channel_list:
        error_range_list = []
        for model in plot_model_name:
            error_range_list.append(error_dict[channel][model].ravel()) # (H, W) -> (H*W,)
        error_range_list = np.stack(error_range_list).reshape(-1) # (n_models, H*W)
        error_range_dict[channel] = np.percentile(np.abs(error_range_list), 95)
        print(f'{channel} error range: {error_range_dict[channel]}')
    

    for c_idx,channel in enumerate(channel_list):
        # load ground truth data
        true_clim_channel = true_clim_data[c_idx] # (H, W)
        # use central percentiles to avoid extreme outliers in climatology plots
        tc_np = true_clim_channel.cpu().numpy()
        # global_min, global_max = np.percentile(tc_np, [5, 95])  # 80% central range
        global_min = tc_np.min()
        global_max = tc_np.max()

       

        subplot_kw = {"projection": ccrs.PlateCarree()} if HAVE_CARTOPY else None
        fig, axes = plt.subplots(2, 3, figsize=(7, 3), subplot_kw=subplot_kw)
        # plot the ground truth first
        axes[0, 0].imshow(
            true_clim_channel.cpu().numpy(),
            cmap=color_map_list[c_idx],
            origin='lower',
            vmin=global_min,
            vmax=global_max,
            extent=(0, 360, -90, 90) if HAVE_CARTOPY else None,
            transform=ccrs.PlateCarree() if HAVE_CARTOPY else None,
        )
        axes[0, 0].set_title(f'ERA5 {channel}')
        if HAVE_CARTOPY:
            gl = axes[0, 0].gridlines(
                draw_labels=True,
                xlocs=np.arange(0, 361, 90),  # 0, 90, 180, 270, 360
                ylocs=np.arange(-90, 91, 30),  # -90, -60, -30, 0, 30, 60, 90
                linewidth=0.5,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            # Show labels on both sides
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}
            axes[0, 0].coastlines(linewidth=0.5)
        else:
            axes[0, 0].set_xticks(np.linspace(0, true_clim_channel.shape[1], 5))
            axes[0, 0].set_xticklabels([0, 90, 180, 270, 360], fontsize=8)
            axes[0, 0].set_yticks(np.linspace(0, true_clim_channel.shape[0], 7))
            axes[0, 0].set_yticklabels([-90, -60, -30, 0, 30, 60, 90], fontsize=8)

        axes[1, 0].set_axis_off()
        
        # draw the data and bias for each model
        for model_idx, model in enumerate(plot_model_name):
            
            # plot the prediction
            axes[0, 1+model_idx].imshow(
                pred_dict[channel][model],
                cmap=color_map_list[c_idx],
                origin='lower',
                vmin=global_min,
                vmax=global_max,
                extent=(0, 360, -90, 90) if HAVE_CARTOPY else None,
                transform=ccrs.PlateCarree() if HAVE_CARTOPY else None,
            )
            axes[0, 1+model_idx].set_title(f'{plot_model_name[model_idx]} Prediction')
            if HAVE_CARTOPY:
                gl = axes[0, 1+model_idx].gridlines(
                    draw_labels=True,
                    xlocs=np.arange(0, 361, 90),  # 0, 90, 180, 270, 360
                    ylocs=np.arange(-90, 91, 30),  # -90, -60, -30, 0, 30, 60, 90
                    linewidth=0.5,
                    color="gray",
                    alpha=0.5,
                    linestyle="--",
                )
                # Show labels on left and bottom sides
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 8}
                gl.ylabel_style = {'size': 8}
                axes[0, 1+model_idx].coastlines(linewidth=0.5)
            else:
                axes[0, 1+model_idx].set_xticks(np.linspace(0, pred_dict[channel][model].shape[1], 5))
                axes[0, 1+model_idx].set_xticklabels([0, 90, 180, 270, 360], fontsize=8)
                axes[0, 1+model_idx].set_yticks(np.linspace(0, pred_dict[channel][model].shape[0], 7))
                axes[0, 1+model_idx].set_yticklabels([-90, -60, -30, 0, 30, 60, 90], fontsize=8)

            # plot the bias
            axes[1, 1+model_idx].imshow(
                error_dict[channel][model],
                cmap=color_map_list[c_idx],
                origin='lower',
                vmin=-error_range_dict[channel],
                vmax=error_range_dict[channel], # positive bias is red, negative bias is blue
                extent=(0, 360, -90, 90) if HAVE_CARTOPY else None,
                transform=ccrs.PlateCarree() if HAVE_CARTOPY else None,
            )
            axes[1, 1+model_idx].set_title(f'{plot_model_name[model_idx]} Bias')
            if HAVE_CARTOPY:
                gl = axes[1, 1+model_idx].gridlines(
                    draw_labels=True,
                    xlocs=np.arange(0, 361, 90),  # 0, 90, 180, 270, 360
                    ylocs=np.arange(-90, 91, 30),  # -90, -60, -30, 0, 30, 60, 90
                    linewidth=0.5,
                    color="gray",
                    alpha=0.5,
                    linestyle="--",
                )
                # Show labels on both sides
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 8}
                gl.ylabel_style = {'size': 8}
                axes[1, 1+model_idx].coastlines(linewidth=0.5)
            else:
                axes[1, 1+model_idx].set_xticks(np.linspace(0, error_dict[channel][model].shape[1], 5))
                axes[1, 1+model_idx].set_xticklabels([0, 90, 180, 270, 360], fontsize=8)
                axes[1, 1+model_idx].set_yticks(np.linspace(0, error_dict[channel][model].shape[0], 7))
                axes[1, 1+model_idx].set_yticklabels([-90, -60, -30, 0, 30, 60, 90], fontsize=8)
        
        # Use tight_layout with rect to leave space for colorbars on the right
        # rect=[left, bottom, right, top] in figure coordinates
        # Leave space on right (0.78) for colorbars, tighter margins elsewhere
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        
        # Colorbar for prediction row (top row: truth + predictions)
        # Position: right side, aligned with top row (shorter height)
        cax_pred = fig.add_axes([0.90, 0.55, 0.015, 0.35])  # [left, bottom, width, height] in figure coords
        cbar_pred = fig.colorbar(axes[0, 1].images[0], cax=cax_pred, orientation='vertical')
        # cbar_pred.set_label(f'{channel_unit_list[c_idx]}', rotation=270, labelpad=15)
        
        # Colorbar for bias row (bottom row: biases only)
        # Position: right side, aligned with bottom row (shorter height)
        cax_bias = fig.add_axes([0.90, 0.1, 0.015, 0.35])  # [left, bottom, width, height] in figure coords
        cbar_bias = fig.colorbar(axes[1, 1].images[0], cax=cax_bias, orientation='vertical')
        # cbar_bias.set_label(f'{channel_unit_list[c_idx]} Bias', rotation=270, labelpad=15)
        plt.savefig(os.path.join(save_dir, f'bias_{channel}.png'))
        plt.close()
 

def plot_bias_demo():
    import matplotlib.pyplot as plt

    # model_list = ['MSWT_sphere', 'HFS_sphere', 'LUCIE']
    # saved_model_name =['mswt_sphere', 'hfs', 'lucie']
    # plot_model_name = ['MSWT', 'HFS', 'LUCIE']
    model_list = ['LUCIE', 'MSWT_patch_sphere']
    saved_model_name =['lucie', 'mswt_patch_sphere']
    plot_model_name = ['LUCIE', 'MSWT']
    # channel_list = ['surface_pressure']
    # channel_list = ['precipitation']
    # plot_channel_name = ['Surface Pressure']
    channel_list =['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'precipitation']
    plot_channel_name = ['Temperature', 'Humidity', 'U-Wind', 'V-Wind', 'Surface Pressure', 'Precipitation']
    channel_unit_list = ['K', 'g/kg', 'm/s', 'm/s', 'hPa', 'mm/d']
    seeds = [42, 43, 44, 45, 46]

    save_dir = '/scratch3/wan410/operator_learning_model/ERA5/'
    true_clim_data = torch.load(os.path.join(save_dir, 'LUCIE', 'true_clim.pt'))
    true_clim_data[1] = true_clim_data[1] * 1000 # (C, H, W) humidity
    true_clim_data[4] = true_clim_data[4] / 100 # (C, H, W) surface pressure 
    true_clim_data[5] = true_clim_data[5] * 4 * 1000 # 6hourly data to daily data (C, H, W) precipitation
               
    # use different diverging color maps for each channel (all with a light/white-ish center)
    color_map_list = ['coolwarm', 'BrBG', 'PiYG', 'PRGn', 'PuOr', 'RdBu_r']


    # load data once for all channels:
    # pred_dict = {'surface_pressure': {}}
    # error_dict = {'surface_pressure': {}}
    pred_dict = {'temperature': {}, 'humidity': {}, 'u_wind': {}, 'v_wind': {}, 'surface_pressure': {}, 'precipitation': {}}
    error_dict = {'temperature': {}, 'humidity': {}, 'u_wind': {}, 'v_wind': {}, 'surface_pressure': {}, 'precipitation': {}}
    for model_idx, model in enumerate(model_list):
        pred_seed_list = []
        error_seed_list = []
        for seed in seeds:
            pred_dict_model_seed = torch.load(os.path.join(save_dir, model, f'rollout_model_{saved_model_name[model_idx]}_seed{seed}.pt')).mean(dim=0) # (C, H, W)
            pred_dict_model_seed[1] = pred_dict_model_seed[1] * 1000 # (C, H, W) humidity
            pred_dict_model_seed[4] = pred_dict_model_seed[4] / 100 # (C, H, W) surface pressure 
            pred_dict_model_seed[5] = pred_dict_model_seed[5] * 4 * 1000 # (C, H, W) precipitation

            error_dict_model_seed = pred_dict_model_seed - true_clim_data # (C, H, W)
            pred_seed_list.append(pred_dict_model_seed)
            error_seed_list.append(error_dict_model_seed)
        
        pred_mean_all = torch.stack(pred_seed_list).mean(dim=0) # average over seeds(C, H, W)
        print(f'{model} pred mean all shape: {pred_mean_all.shape}')
        error_mean_all = torch.stack(error_seed_list).mean(dim=0) # average over seeds(C, H, W)
        print(f'{model} error mean all shape: {error_mean_all.shape}')
        
        for channel_idx, channel in enumerate(channel_list):
            # channel_idx = 4 # surface pressure
            # channel_idx = 5 # precipitation
            pred_dict[channel][plot_model_name[model_idx]] = pred_mean_all[channel_idx].cpu().numpy() # (H, W)
            error_dict[channel][plot_model_name[model_idx]] = error_mean_all[channel_idx].cpu().numpy() # (H, W)
            print(f'{channel} pred shape: {pred_dict[channel][plot_model_name[model_idx]].shape}')
            print(f'{channel} error shape: {error_dict[channel][plot_model_name[model_idx]].shape}')
            
    # iterate overall all channels and models to get the error range (90 percentile)
    error_range_dict = {'temperature': {}, 'humidity': {}, 'u_wind': {}, 'v_wind': {}, 'surface_pressure': {}, 'precipitation': {}}
    for channel in channel_list:
        error_range_list = []
        for model in plot_model_name:
            error_range_list.append(error_dict[channel][model].ravel()) # (H, W) -> (H*W,)
        error_range_list = np.stack(error_range_list).reshape(-1) # (n_models, H*W)
        error_range_dict[channel] = np.percentile(np.abs(error_range_list), 95)
        print(f'{channel} error range: {error_range_dict[channel]}')
    

    for c_idx,channel in enumerate(channel_list):
        # load ground truth data
        true_clim_channel = true_clim_data[c_idx] # (H, W)
        # use central percentiles to avoid extreme outliers in climatology plots
        tc_np = true_clim_channel.cpu().numpy()
        # global_min, global_max = np.percentile(tc_np, [5, 95])  # 80% central range
        global_min = tc_np.min()
        global_max = tc_np.max()

       

        subplot_kw = {"projection": ccrs.PlateCarree()} if HAVE_CARTOPY else None
        fig, axes = plt.subplots(3, 1, figsize=(3.7, 5), subplot_kw=subplot_kw, gridspec_kw={'hspace': 0.4})
        # plot the ground truth first
        axes[0].imshow(
            true_clim_channel.cpu().numpy(),
            cmap=color_map_list[c_idx],
            origin='lower',
            vmin=global_min,
            vmax=global_max,
            extent=(0, 360, -90, 90) if HAVE_CARTOPY else None,
            transform=ccrs.PlateCarree() if HAVE_CARTOPY else None,
        )
        axes[0].set_title(f'ERA5 {plot_channel_name[c_idx] + " (" + channel_unit_list[c_idx] + ")"}', fontsize=10)
        if HAVE_CARTOPY:
            gl = axes[0].gridlines(
                draw_labels=True,
                xlocs=np.arange(0, 361, 90),  # 0, 90, 180, 270, 360
                ylocs=np.arange(-90, 91, 30),  # -90, -60, -30, 0, 30, 60, 90
                linewidth=0.5,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            # Show labels on both sides
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}
            axes[0].coastlines(linewidth=0.5)
            # set the colorbar of the axes[0]
            cbar = plt.colorbar(axes[0].images[0], orientation='vertical')

        else:
            axes[0].set_xticks(np.linspace(0, true_clim_channel.shape[1], 5))
            axes[0].set_xticklabels([0, 90, 180, 270, 360], fontsize=8)
            axes[0].set_yticks(np.linspace(0, true_clim_channel.shape[0], 7))
            axes[0].set_yticklabels([-90, -60, -30, 0, 30, 60, 90], fontsize=8)
        
        # draw the data and bias for each model
        for model_idx, model in enumerate(plot_model_name):
            # plot the bias
            axes[model_idx+1].imshow(
                error_dict[channel][model],
                cmap=color_map_list[c_idx],
                origin='lower',
                vmin=-error_range_dict[channel],
                vmax=error_range_dict[channel], # positive bias is red, negative bias is blue
                extent=(0, 360, -90, 90) if HAVE_CARTOPY else None,
                transform=ccrs.PlateCarree() if HAVE_CARTOPY else None,
            )
            axes[model_idx+1].set_title(plot_model_name[model_idx] + " Bias (" + channel_unit_list[c_idx] + ")", fontsize=10)
            cbar = plt.colorbar(axes[model_idx+1].images[0], orientation='vertical')
            if HAVE_CARTOPY:
                gl = axes[model_idx+1].gridlines(
                    draw_labels=True,
                    xlocs=np.arange(0, 361, 90),  # 0, 90, 180, 270, 360
                    ylocs=np.arange(-90, 91, 30),  # -90, -60, -30, 0, 30, 60, 90
                    linewidth=0.5,
                    color="gray",
                    alpha=0.5,
                    linestyle="--",
                )
                # Show labels on both sides
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 8}
                gl.ylabel_style = {'size': 8}
                axes[model_idx+1].coastlines(linewidth=0.5)
            else:
                axes[model_idx+1].set_xticks(np.linspace(0, error_dict[channel][model].shape[1], 5))
                axes[model_idx+1].set_xticklabels([0, 90, 180, 270, 360], fontsize=8)
                axes[model_idx+1].set_yticks(np.linspace(0, error_dict[channel][model].shape[0], 7))
                axes[model_idx+1].set_yticklabels([-90, -60, -30, 0, 30, 60, 90], fontsize=8)
        
        # fig.tight_layout()
        fig.subplots_adjust(left=0.02, right=0.95, top=0.95, bottom=0.05)
        # Use tight_layout with rect to leave space for colorbars on the right
        # rect=[left, bottom, right, top] in figure coordinates
        # Leave space on right for colorbars, minimize left margin
        # fig.tight_layout(rect=[0, 0, 0.85, 1])
        # Further reduce left margin explicitly (cartopy can add extra space)
        # Set right margin close to colorbar position to minimize gap
        # fig.subplots_adjust(left=0.02, right=0.9, top=0.93, bottom=0.05)
        
        # # Colorbar for prediction row (top row: truth + predictions)
        # # Position: right side, aligned with top row (closer to subplots)
        # cax_truth = fig.add_axes([0.85, 0.75, 0.02, 0.15])  # [left, bottom, width, height] in figure coords
        # cbar_truth = fig.colorbar(axes[0].images[0], cax=cax_truth, orientation='vertical')

        # cax_bias = fig.add_axes([0.85, 0.4, 0.02, 0.15])  # [left, bottom, width, height] in figure coords
        # cbar_bias = fig.colorbar(axes[1].images[0], cax=cax_bias, orientation='vertical')
        
        # cax_bias_2 = fig.add_axes([0.85, 0.1, 0.02, 0.15])  # [left, bottom, width, height] in figure coords
        # cbar_bias_2 = fig.colorbar(axes[2].images[0], cax=cax_bias_2, orientation='vertical')

        plt.savefig(os.path.join(save_dir, f'bias_demo_{channel}.pdf'), dpi=500)
        plt.close()

def plot_zonal_mean():
    import matplotlib.pyplot as plt
    # return 
    model_list = ['LUCIE', 'MSWT_patch_sphere']
    saved_model_name =['lucie','mswt_patch_sphere']
    plot_model_name = ['LUCIE',  'MSWT']
    channel_list = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'precipitation']
    channel_unit_list = ['K', 'g/kg', 'm/s', 'm/s', 'hPa', 'mm/d']
    seeds = [42, 43, 44, 45, 46]
    save_dir = '/scratch3/wan410/operator_learning_model/ERA5/'
    true_clim_data = torch.load(os.path.join(save_dir, 'LUCIE', 'true_clim.pt'))

    # conver several variables to match with the lucie paper:
    # humidity  x1000, pressure / 100, precipitation x1000
    true_clim_data[1] = true_clim_data[1] * 1000
    true_clim_data[4] = true_clim_data[4] / 100
    true_clim_data[5] = true_clim_data[5] * 1000
    true_clim_zonal_mean = torch.mean(true_clim_data, dim=-1) # (C, H) # true climatology zonal mean
    # get average over all seeds for each model
    model_zonal_dict = {'temperature': {}, 'humidity': {}, 'u_wind': {}, 'v_wind': {}, 'surface_pressure': {}, 'precipitation': {}}
    for model_idx, model in enumerate(model_list):
        avg_pred_zonal_mean_list = []
        for seed in seeds:
            data_pred = torch.load(os.path.join(save_dir, model, f'rollout_model_{saved_model_name[model_idx]}_seed{seed}.pt')).mean(dim=0) # (C, H, W)
            data_pred[1] = data_pred[1] * 1000 # humidity
            data_pred[4] = data_pred[4] / 100 # surface pressure
            data_pred[5] = data_pred[5] * 1000 # precipitation
            avg_pred_zonal_mean_list.append(data_pred)
        
        seed_avg = torch.stack(avg_pred_zonal_mean_list).mean(dim=0) # (C, H, W)
        zonal_model_mean = torch.mean(seed_avg, dim=-1) # (C, H)
        for c_idx, channel in enumerate(channel_list):
            model_zonal_dict[channel][plot_model_name[model_idx]] = zonal_model_mean[c_idx].cpu().numpy() # (H,)
        print(f'{plot_model_name[model_idx]} zonal mean shape: {zonal_model_mean.shape}')


    for c_idx, channel in enumerate(channel_list):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        # latitude axis in degrees
        nlat = true_clim_zonal_mean[c_idx].shape[0]
        print("nlat: ", nlat)
        lats = np.linspace(-90, 90, nlat)
        xticks = [-90, -60, -30, 0, 30, 60, 90]
        xtick_labels = ['90S', '60S', '30S', 'EQ', '30N', '60N', '90N']

        # styles and colors for curves (ERA5 + models)
        colors = ['black', '#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b']
        linestyles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]

        # ERA5
        ax.plot(lats, true_clim_zonal_mean[c_idx].cpu().numpy(),
                label='ERA5', color=colors[0], linestyle=linestyles[0], linewidth=2.0)
        
        for model_idx, model in enumerate(plot_model_name):
            data_pred_zonal_mean = model_zonal_dict[channel][model]
            ax.plot(lats, data_pred_zonal_mean, label=model, color=colors[(model_idx + 1) % len(colors)], linestyle=linestyles[(model_idx + 1) % len(linestyles)], linewidth=2.0)

        ax.legend()
        ax.set_xlabel('Latitude')
        ax.set_ylabel(f'{channel_unit_list[c_idx]}')
        ax.set_title(f'Zonal Mean of {channel}')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'zonal_mean_{channel}.png'), dpi=300)


def plot_spectrum():
    import matplotlib.pyplot as plt
    # return 
    model_list = ['LUCIE', 'MSWT_patch_sphere']
    saved_model_name =['lucie', 'mswt_patch_sphere']
    plot_model_name = ['LUCIE', 'MSWT']
    channel_list = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'precipitation']
    # channel_unit_list = ['K', 'g/kg', 'm/s', 'm/s', 'hPa', 'mm/d']
    seeds = [42, 43, 44, 45, 46]

    save_dir = '/scratch3/wan410/operator_learning_model/ERA5/'
    true_clim_data = torch.load(os.path.join(save_dir, 'LUCIE', 'true_clim.pt'))
    true_clim_data[1] = true_clim_data[1] * 1000 # (C, H, W) humidity
    true_clim_data[4] = true_clim_data[4] / 100 # (C, H, W) surface pressure 
    true_clim_data[5] = true_clim_data[5] * 1000 # (C, H, W) precipitation
               

    # load data once for all channels:
    pred_dict = {'temperature': {}, 'humidity': {}, 'u_wind': {}, 'v_wind': {}, 'surface_pressure': {}, 'precipitation': {}}

    for model_idx, model in enumerate(model_list):
        pred_seed_list = []
        for seed in seeds:
            pred_dict_model_seed = torch.load(os.path.join(save_dir, model, f'rollout_model_{saved_model_name[model_idx]}_seed{seed}.pt')).mean(dim=0) # (C, H, W)
            pred_dict_model_seed[1] = pred_dict_model_seed[1] * 1000 # (C, H, W) humidity
            pred_dict_model_seed[4] = pred_dict_model_seed[4] / 100 # (C, H, W) surface pressure 
            pred_dict_model_seed[5] = pred_dict_model_seed[5] * 1000 # (C, H, W) precipitation

            pred_seed_list.append(pred_dict_model_seed)
        
        pred_mean_all = torch.stack(pred_seed_list).mean(dim=0) # average over seeds(C, H, W)
        print(f'{model} pred mean all shape: {pred_mean_all.shape}')
        
        for channel_idx, channel in enumerate(channel_list):
            pred_dict[channel][plot_model_name[model_idx]] = pred_mean_all[channel_idx].cpu().numpy() # (H, W)
            print(f'{channel} pred shape: {pred_dict[channel][plot_model_name[model_idx]].shape}')

    print("true_clim_data.shape: ", true_clim_data.shape) #(C, H, W), (6, 48, 96)
    H, W = true_clim_data.shape[1], true_clim_data.shape[2]
    
    # Define styling for each model
    style_dict = {
        'ERA5': {'color': 'red', 'linestyle': '-', 'linewidth': 2},
        'LUCIE': {'color':  'green',  'linestyle': '--', 'linewidth': 1.0},  # blue, dashed
        'MSWT': {'color':  'black','linestyle': '-.', 'linewidth': 1.5},  # orange, dash-dot
    }
    
    for c_idx, channel in enumerate(channel_list):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        # compute the ground truth energy spectrum for channel c_idx
        # if channel == 'u_wind':
        #     ux = true_clim_data[c_idx]
        #     uy = true_clim_data[c_idx+1]
        #     k_bins, Ek_true = compute_spectra_torch(ux, uy, 2 * math.pi, 2 * math.pi)
        #     k_np = k_bins.detach().cpu().numpy()
        #     valid_mask = range(1, min(len(k_np), min(H, W) // 2))
        #     k_np = k_np[valid_mask]
        #     Ek_true_np = Ek_true.detach().cpu().numpy()[valid_mask]
        #     ax.loglog(k_np, Ek_true_np, label='ERA5')
        #     ax.set_xlabel('Wavenumber')
        #     ax.set_ylabel('Energy E(k)')
        #     ax.set_title(f'Energy Spectrum of {channel}')
        # elif channel == 'v_wind':
        #     continue

        # else:
        k_bins, Zk_true = compute_enstropy_torch(true_clim_data[c_idx], 2 * math.pi, 2 * math.pi)
        k_np = k_bins
        valid_mask = range(1, min(len(k_np), min(H, W) // 2))
        k_np = k_np[valid_mask]
        Zk_true_np = Zk_true.detach().cpu().numpy()[valid_mask]
        ax.loglog(k_np, Zk_true_np, label='ERA5', 
                 color=style_dict['ERA5']['color'],
                 linestyle=style_dict['ERA5']['linestyle'],
                 linewidth=style_dict['ERA5']['linewidth'])
            
    
        for model_idx, model in enumerate(model_list):
            data_pred = torch.tensor(pred_dict[channel][plot_model_name[model_idx]]) # (H, W)
            # if channel == 'u_wind':
            #     ux_pred = data_pred
            #     uy_pred = torch.tensor(pred_dict['v_wind'][plot_model_name[model_idx]])
            #     _, Ek_pred = compute_spectra_torch(ux_pred, uy_pred, 2 * math.pi, 2 * math.pi)
            #     Ek_pred_np = Ek_pred.detach().cpu().numpy()[valid_mask]
            #     ax.loglog(k_np, Ek_pred_np, label=plot_model_name[model_idx])
            # elif channel == 'v_wind':
            #     continue
            # else:
            _, Zk_pred = compute_enstropy_torch(data_pred, 2 * math.pi, 2 * math.pi)
            Zk_pred_np = Zk_pred.detach().cpu().numpy()[valid_mask]
            model_name = plot_model_name[model_idx]
            ax.loglog(k_np, Zk_pred_np, label=model_name,
                     color=style_dict[model_name]['color'],
                     linestyle=style_dict[model_name]['linestyle'],
                     linewidth=style_dict[model_name]['linewidth'], zorder=2+model_idx)
        plt.legend()
        plt.xlabel('Wavenumber')
        plt.ylabel('Energy E(k)')
        plt.title(f'Energy Spectrum of {channel}')
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        plt.savefig(os.path.join(save_dir, f'spectrum_{channel}.png'), dpi=300)
        plt.close()

if __name__ == '__main__':
    # table_metric()
    # plot_bias()
    plot_bias_demo()
    # plot_zonal_mean()
    plot_spectrum()
    # value_list = evaluation_metrics_model_seed.inde
    # plot the avg bias and

