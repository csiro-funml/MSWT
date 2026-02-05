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


def print_zonal_mean_range():
    gadi_path = 'saved_data'
    # gadi_path = '/scratch/v14/mac599/Neural_Climate/LUCIE/' 
    data_file = os.path.join(gadi_path, 'era5_T30_regridded.npz')
    data = load_data(data_file)[...,:6] # (T, H, W, C)
    true_clim = torch.tensor(np.mean(data, axis=0)).permute(2,0,1) # (C, H, W) (6, 48, 96)
    channel_list = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'precipitation']
    print(true_clim.shape)

    # compute the zonal meanof the true clim (diff), average over latitude (the long side)
    true_clim_zonal_mean = torch.mean(true_clim, dim=-1) # (C, H)
    for i in range(true_clim_zonal_mean.shape[0]):
        print(f'{channel_list[i]} min %.4f and max %.4f'%(torch.min(true_clim_zonal_mean[i]).item(), torch.max(true_clim_zonal_mean[i]).item()))


if __name__ == '__main__':
    print_zonal_mean_range()