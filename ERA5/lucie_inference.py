import torch
import numpy as np
from tqdm import tqdm


def inference(model, steps, initial_frame, forcing, initial_forcing_idx, prog_means, prog_stds, diag_means, diag_stds, diff_stds):
    inf_data = []
    model.eval()
    with torch.no_grad():
        inp_val = initial_frame
        for i in tqdm(range(steps)):
            forcing_idx = (initial_forcing_idx + i) % 1460      # tisr is repeating and orography is 
            previous = inp_val[:,:5,:,:]

            pred = model(inp_val)
            pred[:,:5,:,:] = pred[:,:5,:,:] * diff_stds         # denormalize the predicted tendency

            # demornalzie the previous time step and add to the tendecy to reconstruct the current field
            pred[:,:5,:,:] += previous[:,:5,:,:] * prog_stds + prog_means   
            tp_frame = pred[:,5:,:,:] * diag_stds + diag_means
            # pred_frame += (previous_frame + 1) / 2 * (input_maxs - input_mins) + input_mins
            raw = torch.cat((pred[:,:5,:,:],tp_frame), 1)

            inp_val = (raw[:,:5,:,:] - prog_means) / prog_stds      # normalize the current time step for autoregressive prediction
            inp_val = torch.cat((inp_val, forcing[forcing_idx,:,:,:].reshape(1,2,48,96)), dim=1)
            raw = raw.cpu().clone().detach().numpy()
            inf_data.append(raw[0])

    inf_data = np.array(inf_data)
    inf_data[:,5,:,:] = (np.exp(inf_data[:,5,:,:]) - 1) * 1e-2      # denormalzie precipitation that was normalized in log space
    return inf_data


def my_inference(model, steps, initial_frame, forcing, initial_forcing_idx, 
                 prog_means, prog_stds,
                 diag_means, diag_stds, 
                 diff_means, diff_stds):
    inf_data = []
    model.eval()
    # move the means and stds to the same device as the model
    prog_stds = torch.tensor(prog_stds).to(initial_frame.device)
    diag_stds = torch.tensor(diag_stds).to(initial_frame.device)
    diff_means = torch.tensor(diff_means)[None, :, None, None].to(initial_frame.device)
    diff_stds = torch.tensor(diff_stds).to(initial_frame.device)
    print("prog_stds shape", prog_stds.shape)
    print("diag_stds shape", diag_stds.shape)
    print("diff_means shape", diff_means.shape)
    print("diff_stds shape", diff_stds.shape)
    with torch.no_grad():
        inp_val = initial_frame
        for i in tqdm(range(steps)):
            forcing_idx = (initial_forcing_idx + i) % 1460      # tisr is repeating and orography is 
            previous = inp_val[:,:5,:,:] #normalized inputs from the previous time step: data/raw_stds

            pred = model(inp_val) # predicted normalized difference [(y_t - y_t-1)-diff_means]/ diff_stds
            # print("pred shape", pred.shape)
            # handle the first 5 variables (diff_vars)
            pred[:,:5,:,:] = pred[:,:5,:,:] * diff_stds + diff_means         # denormalize the predicted tendency (y_t - y_t-1)
            # demornalzie the previous time step and add to the tendecy to reconstruct the current field
            # y_t = pred + y_{t-1} = pred + previous * raw_stds[:5] 
            pred[:,:5,:,:] += previous[:,:5,:,:] * prog_stds #(didn't add prog_means because diff=False in data_preprocessing.py)   


            # handle the last variable (precipitation):   y_t = log(p_t/1e-2 + 1) / diag_stds
            tp_frame = pred[:,5:,:,:] * diag_stds
            tp_frame = (torch.exp(tp_frame) - 1 )* 1e-2 # unnormalized precipitation
            
            raw = torch.cat((pred[:,:5,:,:],tp_frame), 1) # unnormalized raw data

            # normalize the prediction for autoregressive prediction
            # handle the diff_vars (devided by )
            inp_val = raw[:,:5,:,:] / prog_stds      # normalize the current time step for autoregressive prediction
            inp_val = torch.cat((inp_val, forcing[forcing_idx,:,:,:].reshape(1,2,48,96)), dim=1) # the ground truth normalized forcing variables are provided
            
            raw = raw.cpu().clone().detach().numpy()
            inf_data.append(raw[0])

    inf_data = np.array(inf_data) # should correspond to raw unnormalized data
    # inf_data[:,5,:,:] = (np.exp(inf_data[:,5,:,:]) - 1) * 1e-2      # denormalzie precipitation that was normalized in log space
    return inf_data