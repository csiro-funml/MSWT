import numpy as np
import torch
import os


def load_data(fname):
    data = np.load(fname)
    data_list = []
    for file in data.files:
        data_list.append(data[file])

    np_data = np.asarray(data_list)
    np_data = np.transpose(np_data, (1, 2, 3, 0))

    return np_data


def load_data_era5(device, demo_index=None, load_high_res=False):
    folder = '/scratch3/wan410/operator_learning_data/LUCIE' if torch.cuda.is_available() else 'saved_data'
    if load_high_res:
        data_file = os.path.join(folder, 'era5_512gg_1985-2004_regridded.npz')
        mean_file = os.path.join(folder, 'era5_512gg_1985-clim.npz')
    else:
        data_file = os.path.join(folder, 'era5_T30_regridded.npz')
        mean_file = os.path.join(folder, 'era5_T30_clim.npz')
    if not os.path.exists(mean_file):
        data = load_data(data_file)[...,:6]
        
        true_clim = torch.tensor(np.mean(data, axis=0)).to(device).permute(2,0,1)
        np.savez(mean_file, true_clim=true_clim.cpu().numpy())
        print("True clim saved")
        # exit(-1)
    else:
        true_clim = torch.tensor(np.load(mean_file)['true_clim']).to(device)
        print("True clim loaded")

    if load_high_res:
        data = np.load(os.path.join(folder, "era5_512gg_1985-2004_preprocessed.npz"))     # standardized data with mean and stds generated from dataset_generator.py
    else:
        data = np.load(os.path.join(folder, "era5_T30_preprocessed.npz"))     # standardized data with mean and stds generated from dataset_generator.py
     # the first 10 time steps (T, C, H, W)
    print("Data loaded")
    data_inp = torch.tensor(data["data_inp"],dtype=torch.float32)     # input data 
    data_tar = torch.tensor(data["data_tar"],dtype=torch.float32)
    raw_means = torch.tensor(data["raw_means"],dtype=torch.float32).reshape(1,-1,1,1).to(device)
    raw_stds = torch.tensor(data["raw_stds"],dtype=torch.float32).reshape(1,-1,1,1).to(device)
    prog_means = raw_means[:,:5]
    prog_stds = raw_stds[:,:5]
    diag_means = torch.tensor(data["diag_means"],dtype=torch.float32).reshape(1,-1,1,1).to(device)
    diag_stds = torch.tensor(data["diag_stds"],dtype=torch.float32).reshape(1,-1,1,1).to(device)
    diff_means = torch.tensor(data["diff_means"],dtype=torch.float32).reshape(1,-1,1,1).to(device)
    diff_stds = torch.tensor(data["diff_stds"],dtype=torch.float32).reshape(1,-1,1,1).to(device)

    return data_inp, data_tar, true_clim, prog_means, prog_stds, diag_means, diag_stds, diff_stds




def print_range():
    folder = 'saved_data'
    data_file = os.path.join(folder, 'era5_T30_regridded.npz')
    data = load_data(data_file)[...,:6] # (T, H, W, C)
    true_clim = torch.tensor(np.mean(data, axis=0)).permute(2,0,1) # (C, H, W) (6, 48, 96)
    channel_list = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'precipitation']
    print(true_clim.shape)

    # compute the zonal meanof the true clim (diff), average over latitude (the long side)
    true_clim_zonal_mean = torch.mean(true_clim, dim=-1) # (C, H)
    for i in range(true_clim_zonal_mean.shape[0]):
        print(f'{channel_list[i]} min %.4f and max %.4f'%(torch.min(true_clim_zonal_mean[i]).item(), torch.max(true_clim_zonal_mean[i]).item()))


if __name__ == '__main__':
    print_range()