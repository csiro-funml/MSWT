"""
Compare the prediciton of the FNO model with the ground truth
# (250 snapshots in total, the first 7 steps were used as input to FNO, the rest 243 were obtained by One step ahead prediction)
"""
import numpy as np
import h5py


def downsample_x(u, downsample_spatial=(1, 1)):
    import torch
    """
    Downsample a real-valued input using FFT, matching DedalusDataset2D.
    Args:
        u: Input tensor of shape (T, C, H, W)
        N: Target size for downsampling (assumes square target N x N)
    Returns:
        Downsampled tensor of shape (T, C, N, N)
    """
    T, C, H, W = u.shape
    N = H/downsample_spatial[0]
    u_hat = torch.fft.rfft2(u, norm='forward')
    freqs_h = torch.fft.fftfreq(H, d=1/H)
    freqs_w = torch.fft.rfftfreq(W, d=1/W)
    sel_h = torch.logical_and(freqs_h >= -N/2, freqs_h <= N/2-1)
    sel_w = torch.logical_and(freqs_w >= -N/2, freqs_w <= N/2-1)
    u_hat_down = u_hat[:, :, sel_h][:, :, :, sel_w]
    u_down = torch.fft.irfft2(u_hat_down, s=(N, N), norm='forward')
    return u_down



def get_groundtruth_from_h5(h5_path, snapshot_idx, downsample_spatial=(1, 1)):
    h5_path ='/datasets/work/oa-tcch/work/forXuesong/new/realisation_0000/snapshots/snapshots_s1.h5'
    true_vorticity = []
    true_streanfunction = []
    with h5py.File(h5_path, 'r') as f:
        for idx in snapshot_idx:
            vorticity = np.array(f['tasks/vorticity'][idx])
            streamfunction = np.array(f['tasks/streamfunction'][idx])
            true_vorticity.append(vorticity)
            true_streanfunction.append(streamfunction)
    true_vorticity = np.array(true_vorticity)
    true_streanfunction = np.array(true_streanfunction)
    print("true vorticity shape: ", true_vorticity.shape, "true streamfunction shape: ", true_streanfunction.shape)
    data = np.concatenate([true_vorticity, true_streanfunction], axis=0)
    if downsample_spatial[0] != (1,1):
        data = downsample_x(data, downsample_spatial)
        true_vorticity, true_streanfunction = data[:len(true_vorticity)], data[len(true_vorticity):]
    return true_vorticity, true_streanfunction
    


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
    # compute MSE between ground truth and prediction
    mse_vorticity = np.mean((pred_vorticity - true_vorticity)**2)
    mse_streamfunction = np.mean((pred_streanfunction - true_streanfunction)**2)
    print("MSE between vorticity: ", mse_vorticity, "MSE between streamfunction: ", mse_streamfunction)
    return pred_vorticity, pred_streanfunction, true_vorticity, true_streanfunction



if __name__ == "__main__":
    pred_path = '/scratch3/wan410/operator_learning_model/FNO_ns2d_dedalus_ntrain4968/test_data_prediction.npz'
    load_data(pred_path, h5_path=None, load_both=False)