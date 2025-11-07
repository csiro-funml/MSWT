import os
import h5py
folder = '/datasets/work/oa-tcch/work/forXuesong/with-forcing/realisation_0000/snapshots'
data_file = os.path.join(folder, 'snapshots_s1.h5')

with h5py.File(data_file, 'r') as f:
    for key in f['tasks'].keys():
        var = f['tasks'][key]
        print(var)
        print(var.shape)
        print(var.mean())
        print(var.std())
    