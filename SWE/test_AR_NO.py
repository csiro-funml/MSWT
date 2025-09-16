# TEST THE Predictions of the model

import sys
import os
# Add parent directory to Python path to access utils and models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# os.environ['OMP_NUM_THREADS'] = '16'
import warnings
import json
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange


from timeit import default_timer
from torch.utils.tensorboard import SummaryWriter
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint, resume_training_from_checkpoint

from utils.griddataset import MixedTemporalDataset, TemporalDataset2D, TemporalDataset2D_multiscale, LocalTemporalDataset2D
from utils.make_master_file import DATASET_DICT
from models.fno import FNO2d
from models.wavelet_transform import CrossWaveletTransformer
from models.high_frequency_scaling import ResUNet
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils.criterion import RelL2Norm, RMSE, BoundaryRMSE, MaxAbsError, GlobalMaxAbsError, SpectralError

warnings.filterwarnings("ignore")

################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='FNO') # FNO, ViT, UNO, CNO, Oformer, Transolver, DPOT, Crossformer, wavelet_transformer
parser.add_argument('--dataset',type=str, default='ns2d_pda') # ['ns2d_fno_1e-3', 'ns2d_pda', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 'sw2d_pda'], note: pdb is the pde bench
parser.add_argument('--resume_path',type=int, default=0 if not torch.cuda.is_available() else 1) # use random weights if not cuda available
parser.add_argument('--use_writer', action='store_true',default=False)



# ### dataset details
parser.add_argument('--T_in', type=int, default=7)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--pad', type=int, default=0)
parser.add_argument('--normalize',type=int, default=1)


# ### FNO/UNO params 
parser.add_argument('--n_layers',type=int, default=8)
parser.add_argument('--modes', type=int, default=16)
# parser.add_argument('--modes', type=int, default=32)
parser.add_argument('--width', type=int, default=64)
# parser.add_argument('--use_ln',type=int, default=0)
parser.add_argument('--act',type=str, default='gelu')


# ### DPOT
# parser.add_argument('--patch_size',type=int, default=8)
parser.add_argument('--n_blocks',type=int, default=8)
parser.add_argument('--mlp_ratio',type=int, default=1)
parser.add_argument('--out_layer_dim', type=int, default=32)

# ### ViT
parser.add_argument('--patch_size',type=int, default=16)
# parser.add_argument('--coord',type=str, default='fourier', choices=['fourier', 'cartesian', 'learnable', 'spherical'])
# parser.add_argument('--prenorm',type=str, default='standard', choices=['standard', 'lognorm', 'instancenorm', 'batchnorm'])


###### optimizer and training setups
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--save_everyepoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb'])
parser.add_argument('--beta1',type=float,default=0.9)
parser.add_argument('--beta2',type=float,default=0.9)
parser.add_argument('--lr_method',type=str, default='cossin') # cyclic for ViT perhaps
parser.add_argument('--grad_clip',type=float, default=10000.0)
parser.add_argument('--step_size', type=int, default=20)
parser.add_argument('--step_gamma', type=float, default=0.5)
parser.add_argument('--warmup_epochs',type=int, default=100)

parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='/scratch3/wan410/operator_learning_model/')

args = parser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Current working directory: {os.getcwd()}")


def load_data_model(just_load_path=False):
    ################################################################
    # load some toy data to run locally
    if not torch.cuda.is_available():
        train_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, n_channels=3, normalize=args.normalize, train='train')
        test_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels=3, normalize=args.normalize, train='test')
    else:
        # load data and dataloader
        train_dataset = TemporalDataset2D(args.dataset, t_in = args.T_in, t_ar = args.T_ar, train='train', normalize=args.normalize)
        test_dataset = TemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels = train_dataset.n_channels, train='test', normalize=args.normalize)


    
    ntrain, ntest = len(train_dataset), len(test_dataset)
    ntrain = 5200 if args.dataset == 'ns2d_pda' else ntrain # for testing (with my local folder)
    print(args.dataset)
    print('Train num {}, Test num {}'.format(ntrain, ntest))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
    # val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)

    # ntrain, ntest = len(train_dataset), len(test_dataset)

    if not args.pad:
        args.res = train_dataset.res  # use original dataset  resolution to train the model

    comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
    log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
    if not os.path.exists(log_path):# running tests locallt
        log_path = './logs/' + comment
    # model_path = log_path + '/model.pth' # for testing, note: to be deleted later
    model_path = log_path + f'/model_epochs_{args.epochs}.pth'
    print(model_path)
    
    # if just_load_path, return the log_path
    if just_load_path:
        return None, test_loader,log_path
    
    if args.use_writer:
        writer = SummaryWriter(log_dir=log_path)
        # write params (usually you only do this once)
        json.dump(vars(args),
            open(os.path.join(log_path, 'params.json'), 'w'),
            indent=4)
        # open the log file in append mode, line‑buffered
        fp = open(os.path.join(log_path, 'logs.txt'), 'a', buffering=1)
        # redirect stdout there (so all prints go into logs.txt)
        sys.stdout = fp
    else:
        writer = None

    print('args',args)
    # find the maximum number of steps in the test dataset
    max_steps = test_dataset[0][1].shape[-2]
    print('Train num {} train len {} test num {} test max steps {}'.format(train_dataset.n_size, ntrain, ntest, max_steps))



    ################################################################
    # load model
    ################################################################
    if args.model == "FNO":
        model = FNO2d(args.modes, args.modes, width=args.width,
                    n_channels=train_dataset.n_channels,
                    in_timesteps = args.T_in, out_timesteps=1, 
                    n_layers = args.n_layers).to(device)
    elif args.model == 'wavelet_transformer':
        model = CrossWaveletTransformer(wave='haar', n_channels=train_dataset.n_channels, in_timesteps = args.T_in, dim=512, depth=8).to(device)
    elif args.model == 'HFS':
        model =  ResUNet(in_c = train_dataset.n_channels * args.T_in + 2 ,out_c = train_dataset.n_channels, 
                 bottleneck_feature=512, 
                 device=device).to(device)
    else:
        raise NotImplementedError


    print(model)
    count_parameters(model)

    start_epoch = 0
    best_loss_epoch = 0
    if args.resume_path:
        print('Loading models and resume from {}'.format(model_path))
        args.resume_path = model_path
        model, optimizer, scheduler, start_epoch = resume_training_from_checkpoint(model, args.resume_path, device, optimizer=None, scheduler=None)
        print("resume training from epoch:", start_epoch)
        best_loss_epoch = start_epoch

    return model, test_loader, log_path

################################################################
# Function 1 Report Average step, step-wise, and full prediction relative l2 norm
################################################################
def predict_and_save(model, test_loader, save=False, log_path=None, max_steps=None):
    """
    test_error(model, test_loader, save=False)
    Args:
        model: the model to test
        test_loader: the test loader
        save: whether to save the results to a pthfile
    Returns:
        save_data = {'input': torch.Tensor, 'output': torch.Tensor, 'pred': torch.Tensor} for computing the error
    """
    with torch.no_grad():
        model.eval()
        test_dataset = test_loader.dataset
        max_steps = test_dataset[0][1].shape[-2] if max_steps is None else max_steps

        save_data = {'input': [], 'output': [], 'pred': []}
        # autoregressive computing  
        for i, (xx, yy) in tqdm(enumerate(test_loader), total=len(test_loader)):
            save_data['input'].append(xx)
            xx_raw = xx.to(device)
            yy = yy.to(device)
            xx = test_dataset.normalize_x(xx_raw) # normalize the input before the autoregressive predicting
            if i == 0:
                # print the range of the xx_raw
                for c in range(xx_raw.shape[-1]):
                    print("xx range before normalization", c, xx_raw[:, :, :, :, c].max().item(), xx_raw[:, :, :, :, c].min().item())
                    print("xx range after normalization", c, xx[:, :, :, :, c].max().item(), xx[:, :, :, :, c].min().item())
                # print the total number of steps in dataset
                print("total number of steps in dataset", yy.shape[-2])

                # preint 
            for t in range(0, yy.shape[-2], args.T_bundle):
                if t > 30:
                    break
                im = model(xx)
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -2)
                xx = torch.cat((xx[..., args.T_bundle:,:], im), dim=-2)
            # denormalize the pred at the final step (get better results)   
            pred = test_dataset.denormalize_x(pred)    
        
            # # save the data to np_data
            # # print("save input and output shape", xx.shape, yy.shape)
            save_data['output'].append(yy[..., :pred.shape[-2], :]) # I didn't save the entire trajectory
            save_data['pred'].append(pred)

        # organzie np_data
        save_data['input'] = torch.cat(save_data['input'], axis=0)
        save_data['output'] = torch.cat(save_data['output'], axis=0)
        save_data['pred'] = torch.cat(save_data['pred'], axis=0)

        # print the shape of the np_data
        print("save_data shape", save_data['input'].shape, save_data['output'].shape, save_data['pred'].shape)

        # save to npz
        if save:
            torch.save(save_data, f'{log_path}/test_data_prediction.pth')

        return save_data



################################################################
# Function 2 Compute evaluation metrics
################################################################
def compute_evalutation_metrics(save_data, model_name='', log_path=''):
    pred, target = save_data['pred'], save_data['output'] # shape: (B, H, W, T, C)
    
    # for step in [0, -1]: # first step and last step
        # for c in range(pred.shape[-1]):
            # low_err, mid_err, high_err = SpectralError()(pred[..., step, c][:, :, :, None, None], target[..., step, c][:, :, :, None, None])
            

    loss_dict = {}
    loss_dict['rel_l2_loss'] = RelL2Norm() # rel L2 loss
    loss_dict['rmse'] = RMSE()
    loss_dict['boundary_rmse'] = BoundaryRMSE()
    loss_dict['max_avg'] = MaxAbsError()
    loss_dict['max_global'] = GlobalMaxAbsError()
    loss_dict['spectral_error'] = SpectralError(model_name=model_name, save_path=log_path, low_percentile=0.70, high_percentile=0.97)
    
    if 'ns2d' in log_path: # NS equation
        step_dict = {0: "t=1", -1: "t=T"} # just plot two steps
    else:
        total_steps_to_compute = 5
        step_dict = {t: f"t={t+1}" for t in range(0, pred.shape[-2], pred.shape[-2]//total_steps_to_compute)}
        print("steps to compute",step_dict.keys())
    # Standard error metrics
    print("\n=== Channel-wise Error Metrics ===")
    save_df = pd.DataFrame(columns=["step", "channel", "metric", f"{model_name}"])
    
    for step in step_dict.keys(): # first step and last step
        print("evaluating step .....", step)
        for c in range(pred.shape[-1]):
            # evaluate different metrics per channel
            for key, loss_func in loss_dict.items():
                if key == 'spectral_error':
                    # (B, H, W, T, C)
                    loss_metric = loss_func(pred[..., step, c][:, :, :, None, None], target[..., step, c][:, :, :, None, None],
                                             channel=c, time_step=step, save_plot=True)
                    # loss metric is a dict with keys: 'low_err', 'mid_err', 'high_err', 'k_low', 'k_high'
                    print("frequency bands", loss_metric['k_low'], loss_metric['k_high'])
                    for band_key, val in loss_metric.items():
                        if band_key == 'k_low' or band_key == 'k_high':
                            continue
                        print(f"Channel {c} {step_dict[step]} {band_key}: {val:.6f}")
                        new_row = pd.Series({"step": step_dict[step], "channel": c, "metric": band_key, f"{model_name}": val,
                                             "k_low": loss_metric['k_low'], "k_high": loss_metric['k_high']}).to_frame().T
                        save_df = pd.concat([save_df, new_row], ignore_index=True)
                    
                else:
                    
                    loss_metric = loss_func(pred[..., step, c][:, :, :, None, None], target[..., step, c][:, :, :, None, None])
                    print(f"Channel {c} {step_dict[step]} {key}: {loss_metric.item():.6f}")   
                    new_row = pd.Series({"step": step_dict[step], "channel": c, "metric": key, f"{model_name}": loss_metric.item()}).to_frame().T
                    save_df = pd.concat([save_df, new_row], ignore_index=True)
    print(save_df.head(n=16))
    save_df.to_csv(f"{log_path}/evalutation_metrics_{model_name}_epochs_{args.epochs}.csv", index=False)
    return loss_dict
    

################################################################
# Function 3  save the data for diffusion training
################################################################
def no_postprocessing_pred_save_data(args):

    ################################################################
    # load some toy data to run locally
    if not torch.cuda.is_available():
        train_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, n_channels=3, normalize=args.normalize, train='train')
        test_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels=3, normalize=args.normalize, train='test')
        val_dataset = test_dataset
    else: 
        # load data and dataloader
        train_dataset = TemporalDataset2D(args.dataset, t_in = args.T_in, t_ar = args.T_ar, train='train', normalize=args.normalize)
        val_dataset =  TemporalDataset2D(args.dataset, n_train=260, t_in = args.T_in, t_ar =-1, train='val', normalize=args.normalize)
        test_dataset = TemporalDataset2D(args.dataset, n_train=260, t_in=args.T_in, t_ar=-1, n_channels = train_dataset.n_channels, train='test', normalize=args.normalize)


    
    ntrain, ntest = len(train_dataset), len(test_dataset)
    ntrain = 5200 if args.dataset == 'ns2d_pda' else ntrain # for testing


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
    val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)

    loaderdict = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    if not args.pad:
        args.res = train_dataset.res  # use original dataset  resolution to train the model

    comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
    log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
    if not os.path.exists(log_path):# running tests locallt
        log_path = './logs/' + comment
    model_path = log_path + f'/model_epochs_{args.epochs}.pth'
    print(model_path)
    
    # Load pretrained neural operator
    if args.model == "FNO":
        model = FNO2d(args.modes, args.modes, width=args.width,
                    n_channels=train_dataset.n_channels,
                    in_timesteps = args.T_in, out_timesteps=1, 
                    n_layers = args.n_layers).to(device)
    elif args.model == 'UNO':
        model = UNO( width=args.width, n_channels=train_dataset.n_channels, in_timesteps = args.T_in,  out_timesteps=1).to(device)
    elif args.model == 'wavelet_transformer':
        model = CrossWaveletTransformer(wave='haar', n_channels=train_dataset.n_channels, in_timesteps = args.T_in, dim=512, depth=8).to(device)
    else:
        raise NotImplementedError

    # load weights
    model, optimizer, scheduler, start_epoch = resume_training_from_checkpoint(model, model_path, device, optimizer=None, scheduler=None)
    print(f"Loaded {args.model} neural operator")


    # iterate throught train/val/test loader and save the predictions
    loaderdict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    for key, loader in loaderdict.items():
        model.eval()
        with torch.no_grad():
            save_data = {'output': [], 'pred': []}
            # autoregressive computing  
            for xx, yy in tqdm(loader):
                xx = xx.to(device)
                yy = yy.to(device)
                xx = loader.dataset.normalize_x(xx) # normalize the input before the autoregressive predicting
                for t in range(0, yy.shape[-2], args.T_bundle):
                    im = model(xx)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -2)
                    xx = torch.cat((xx[..., args.T_bundle:,:], im), dim=-2)
                # denormalize the pred at the final step (get better results)   
                pred = loader.dataset.denormalize_x(pred)    

                # # save the data to np_data
                save_data['output'].append(yy)
                save_data['pred'].append(pred)

            # organzie np_data
            save_data['output'] = torch.cat(save_data['output'], axis=0).cpu().numpy()
            save_data['pred'] = torch.cat(save_data['pred'], axis=0).cpu().numpy()
            
            print("save_data shape", save_data['output'].shape, save_data['pred'].shape)
            if not os.path.exists(f'{log_path}/diffusion'):
                os.makedirs(f'{log_path}/diffusion')
            np.savez(f'{log_path}/diffusion/{key}_pred.npz', **save_data)
            print(f"Saved {key} predictions to {log_path}/diffusion/{key}_pred.npz")

    print("Successfully saved the predictions for diffusion training")






if __name__ == '__main__':
    
    #### 1. predict and save the data
    model, test_loader, log_path = load_data_model(just_load_path=False)
    save_data = predict_and_save(model, test_loader, save=True, log_path=log_path, max_steps=None)
    
    #### 2. load the save_data
    # save_data = torch.load(f'{log_path}/test_data_prediction.pth', map_location=device)
    
    #### 3. compute different types of metrics
    # compute_evalutation_metrics(save_data, model_name=args.model, log_path=log_path)

    #### 4. postprocessing save the data for diffusion training
    # no_postprocessing_pred_save_data(args)