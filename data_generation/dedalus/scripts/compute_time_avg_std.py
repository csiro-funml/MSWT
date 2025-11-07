import os
import h5py
import numpy as np

folder = '/datasets/work/oa-tcch/work/forXuesong/with-forcing/realisation_0000/snapshots'
data_file = os.path.join(folder, 'snapshots_s1.h5')

with h5py.File(data_file, 'r') as f:
    for key in f['tasks'].keys():
        print(key)
        var = np.array(f['tasks'][key][:])
        # print(f"shape: {var.shape}")
        if len(var.shape) == 4: # (T, C, H, W)
            for c in range(var.shape[1]):
                var_c = var[:, c, :, :]
                print(f"shape: {var_c.shape} for channel {c}")
                print(f"mean: {var_c.mean()}")
                print(f"std: {var_c.std()}")
        else: # (T, H, W)
            print(f"shape: {var.shape}")
            print(f"mean: {var.mean()}")
            print(f"std: {var.std()}")
    