import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.stats as stats
import os


gadi_path = '/scratch/v14/mac599/Neural_Climate/LUCIE/' 
virga_path = '/scratch3/wan410/operator_learning_data/LUCIE/'
if os.path.exists(gadi_path):
    print("on the gadi server")
    path = gadi_path
elif os.path.exists(virga_path):
    path = virga_path
else:
    print("on mac")
    path = './'

# data_name = 'era5_T30_regridded.npz'  # atmospheric data (small one)
data_name = 'era5_512gg_1985-2004_regridded.npz'  # atmospheric data/



data = np.load(path + data_name)


for key in data.keys():
    print(key, "shape",data[key].shape)