import os

folder_path = '/scratch3/wan410/operator_learning_data/pdearena/sw2d_pda'
train_path = os.path.join(folder_path, 'train')
val_path = os.path.join(folder_path, 'val')


# move the last 100 files (data_6899.hdf5 to data_6999.hdf5) from train to val and reindex from the 0 to 99.hdf5
for i in range(100):
    os.system(f'mv {os.path.join(train_path, f"data_{6899 + i}.hdf5")} {os.path.join(val_path, f"data_{i}.hdf5")}')
    print(f'moved data_{6899 + i}.hdf5 to val folder')


