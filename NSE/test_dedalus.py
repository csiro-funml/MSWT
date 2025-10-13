import sys
import os
# Add parent directory to Python path to access utils and models
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from utils.griddataset import DedalusDataset2D



train_dataset = DedalusDataset2D(data_name='ns2d_dedalus', train='train')
x_train, y_train = train_dataset[0]
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)

test_dataset = DedalusDataset2D(data_name='ns2d_dedalus', train='test')
x_test, y_test = test_dataset[0]
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)

val_dataset = DedalusDataset2D(data_name='ns2d_dedalus', train='val')
x_val, y_val = val_dataset[0]
print("x_val shape", x_val.shape)
print("y_val shape", y_val.shape)