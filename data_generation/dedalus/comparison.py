"""
Compare the prediciton of the FNO model with the ground truth
# (250 snapshots in total, the first 7 steps were used as input to FNO, the rest 243 were obtained by One step ahead prediction)
"""
import numpy as np


def get_groundtruth_from_h5(h5_path, snapshot_idx, downsample_spatial):
    pass


def load_data(pred_path, h5_path=None, load_both=True):
    data = np.load(pred_path, allow_pickle=True)
    print("data keys: ",data.keys())
    pred_vorticity = data['pred_vorticity'] # (T, Nx, Ny) (243, 128, 128)
    pred_streanfunction = data['pred_streamfunction'] # (T, Nx, Ny) (243, 128, 128)
    if load_both:    
        true_vorticity = data['output_vorticity']
        true_streanfunction = data['output_streamfunction']
    else:
        start_idx = 7500
        end_idx = 8500
        stride = 4 # the model was trained to predict the solution after 4 steps
        downsample_spatial = (2, 2) # downsample the spatial resolution from 256 to 128 
        skip_steps = 7 # the first 7 steps were used as input to FNO
        snapshot_idx = np.arange(start_idx, end_idx, stride)[skip_steps:] # 243 snapshots
        true_vorticity, true_streanfunction = get_groundtruth_from_h5(h5_path, snapshot_idx, downsample_spatial)
    print("pred vorticity shape: ", pred_vorticity.shape, "pred streamfunction shape: ", pred_streanfunction.shape)
    print("true vorticity shape: ", true_vorticity.shape, "true streamfunction shape: ", true_streanfunction.shape)
    return pred_vorticity, pred_streanfunction, true_vorticity, true_streanfunction



if __name__ == "__main__":
    pred_path = '/scratch3/wan410/operator_learning_model/FNO_ns2d_dedalus_ntrain4968/test_data_prediction.npz'
    load_data(pred_path, h5_path=None, load_both=True)