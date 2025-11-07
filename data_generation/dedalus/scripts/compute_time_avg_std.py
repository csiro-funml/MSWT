import os
import h5py
folder = '/datasets/work/oa-tcch/work/forXuesong/with-forcing/realisation_0000/snapshots'
data_file = os.path.join(folder, 'snapshots_s1.h5')

data = h5py.File(data_file, 'r')
for key in data.keys():
    print(key)
    var = data[key][:]
    print(var.shape)
    print(var.mean())
    print(var.std())
    