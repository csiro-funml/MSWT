"""
Train a neural operator to predict the next step of the PDE
This code is developed with reference to the following GitHub repo: DPOT: https://github.com/HaoZhongkai/DPOT/
"""

import sys
import os
# Add parent directory to Python path to access utils and models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# os.environ['OMP_NUM_THREADS'] = '16'

import json
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from timeit import default_timer
from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR,  CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer import Adam, Lamb
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint, resume_training_from_checkpoint
from utils.criterion import RelL2Norm, compute_error_fft, RMSE, BoundaryRMSE, MaxAbsError, GlobalMaxAbsError, SpectralError
from utils.griddataset import MixedTemporalDataset, TemporalDataset2D, LocalTemporalDataset2D
from utils.make_master_file import DATASET_DICT
from models.fno import FNO2d
from models.uno import UNO
from models.wavelet_transform import CrossWaveletTransformer
from models.high_frequency_scaling import ResUNet
from models.unet import UNet_with_BottleneckHFS, UNet_withoutHFS
from models.hano import HANO2d
import pickle
from tqdm import tqdm
from lion_pytorch import Lion


################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='FNO') # FNO, wavelet_transformer, HFS, UNet, HANO, UNO 
parser.add_argument('--dataset',type=str, default='sw2d_pda') # ['ns2d_fno_1e-3', 'ns2d_pda', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 'sw2d_pda'], note: pdb is the pde bench
parser.add_argument('--resume_path',type=str, default='')
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
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--save_everyepoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb','lion'])
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



################################################################
# load some toy data to run locally
if not torch.cuda.is_available():
    train_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, n_channels=3, normalize=args.normalize, train='train')
    test_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels=3, normalize=args.normalize, train='test')
    val_dataset= test_dataset
else:
    # load data and dataloader
    train_dataset = TemporalDataset2D(args.dataset, t_in = args.T_in, t_ar = args.T_ar, train='train', normalize=args.normalize)
    val_dataset =  TemporalDataset2D(args.dataset, n_train=260, t_in = args.T_in, t_ar =-1, train='val', normalize=args.normalize)
    test_dataset = TemporalDataset2D(args.dataset, n_train=260, t_in=args.T_in, t_ar=-1, n_channels = train_dataset.n_channels, train='test', normalize=args.normalize)



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0 if not torch.cuda.is_available() else 8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=0 if not torch.cuda.is_available() else 8)
val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=0 if not torch.cuda.is_available() else 8)

ntrain, ntest = len(train_dataset), len(test_dataset)
if not args.pad:
    args.res = train_dataset.res  # use original dataset  resolution to train the model

comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
# model_path = log_path + '/model.pth'
# model_path = log_path + f'/model_epochs_{args.epochs}.pth' # I will test a longer training epoch
model_path = log_path + f'/model_epochs_{args.epochs}.pth'
print(model_path)
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
print('Train num {} train len {} test num {}'.format(train_dataset.n_size, ntrain, ntest))

################################################################
# load model
################################################################
if args.model == "FNO":
    model = FNO2d(args.modes, args.modes, width=args.width,
                  n_channels=train_dataset.n_channels,
                  in_timesteps = args.T_in, out_timesteps=1, 
                  n_layers = args.n_layers, 
                # normalize=args.normalize, 
                 ).to(device)
elif args.model == 'UNO':
    model = UNO( width=args.width, n_channels=train_dataset.n_channels, in_timesteps = args.T_in,  out_timesteps=1).to(device)
elif args.model == 'wavelet_transformer':
    model = CrossWaveletTransformer(wave='haar', n_channels=train_dataset.n_channels, in_timesteps = args.T_in, dim=512, depth=8).to(device)
elif args.model == 'HFS':
    model =  ResUNet(in_c = train_dataset.n_channels * args.T_in + 2 ,out_c = train_dataset.n_channels, 
                     bottleneck_feature=512, 
                     device=device).to(device)
elif args.model == 'UNet_withoutHFS':
    model = UNet_withoutHFS(in_c = train_dataset.n_channels * args.T_in + 2 ,out_c = train_dataset.n_channels, 
                            bottleneck_feature=512, 
                            device=device).to(device)
elif args.model == 'UNet':
    model = UNet_with_BottleneckHFS(in_c = train_dataset.n_channels * args.T_in + 2 ,out_c = train_dataset.n_channels, 
                                   bottleneck_feature=512, 
                                   device=device).to(device)
elif args.model == 'HANO':
    model = HANO2d(T_in=args.T_in, T_out=args.T_ar, res_output=train_dataset.res[0],  res_att=train_dataset.res[0],
                   in_dim=train_dataset.n_channels, out_dim=train_dataset.n_channels, feature_dim=256).to(device)
else:
    raise NotImplementedError


#### set optimizer
if args.opt == 'lamb':
    optimizer = Lamb(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2), adam=True, debias=False,weight_decay=1e-4)
elif args.opt == 'lion':
    optimizer = Lion(model.parameters(), lr=args.lr, weight_decay = 0.01)
else:
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-6)


if args.lr_method == 'cycle':
    print('Using cycle learning rate schedule')
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, div_factor=1e4, pct_start=(args.warmup_epochs / args.epochs), final_div_factor=1e4, steps_per_epoch=len(train_loader), epochs=args.epochs)
elif args.lr_method == 'step':
    print('Using step learning rate schedule')
    scheduler = StepLR(optimizer, step_size=args.step_size * len(train_loader), gamma=args.step_gamma)
elif args.lr_method == 'warmup':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: min((steps + 1) / (args.warmup_epochs * len(train_loader)), np.power(args.warmup_epochs * len(train_loader) / float(steps + 1), 0.5)))
elif args.lr_method == 'linear':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: (1 - steps / (args.epochs * len(train_loader))))
elif args.lr_method == 'restart':
    print('Using cos anneal restart')
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * args.lr_step_size, eta_min=0.)
elif args.lr_method == 'cyclic':
    scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=args.lr_step_size * len(train_loader),mode='triangular2', cycle_momentum=False)
elif args.lr_method == 'cossin':
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader)) 
else:
    raise NotImplementedError

start_epoch = 0
best_loss_epoch = 0
print(model)
count_parameters(model)

if args.resume_path:
    print('Loading models and resume from {}'.format(model_path))
    args.resume_path = model_path
    model, optimizer, scheduler, start_epoch = resume_training_from_checkpoint(model, args.resume_path, device, optimizer, scheduler)
    print("resume training from epoch:", start_epoch)
    best_loss_epoch = start_epoch


################################################################
# Main function for pretraining
################################################################
myloss = RelL2Norm(size_average=False)
loss_dict = {} # for testing
loss_dict['rel_l2_loss'] = RelL2Norm() # rel L2 loss
loss_dict['rmse'] = RMSE()
loss_dict['boundary_rmse'] = BoundaryRMSE()
loss_dict['max_avg'] = MaxAbsError()
loss_dict['max_global'] = GlobalMaxAbsError()
loss_dict['spectral_error'] = SpectralError(model_name=args.model, save_path=log_path, low_percentile=0.70, high_percentile=0.97)


best_loss = np.inf
# Store a grayscale heatmap of error spectrum over epochs (C=1, W= num_saved_epochs, H=num_bins)
num_bins = min(train_dataset.res[0], train_dataset.res[1]) // 2

pbar = tqdm(range(start_epoch, args.epochs))
for ep in pbar:
    model.train()

    t1 = t_1 = default_timer()
    t_load, t_train = 0., 0.
    train_l2_norm = 0
    train_l2_denorm = 0

    loss_previous = 10000

    for batch_id, (xx, yy) in enumerate(train_loader):
        t_load += default_timer() - t_1
        t_1 = default_timer()
        loss = 0.
        xx = xx.to(device)  ## B, n, n, T_in, C
        yy = yy.to(device)  ## B, n, n, T_ar, C
        # range check
        if ep == 0 and batch_id == 0:
            C = xx.shape[-1]
            x_temp = xx.reshape((-1, C))
            for c_i in range(C):
                print("channel: %s range before normalization "%c_i, x_temp[:, c_i].max().item(), x_temp[:, c_i].min().item())
        
        # normalize it before the autoregressive predicting
        xx = train_dataset.normalize_x(xx)
        yy_norm = train_dataset.normalize_x(yy)
        for t in range(0, yy_norm.shape[-2], args.T_bundle):
            y = yy_norm[..., t:t + args.T_bundle, :]
            # print('input shape', xx.shape)
            pred = model(xx)  # give the normalized output to the autoregressive predicting
            loss += myloss(pred, y)

        train_l2_norm += loss.item() * y.shape[0]

        pbar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{ep}/{args.epochs}")
        # print("train input shape", xx.shape, "output shape", yy.shape, "pred shape", pred.shape, "mask shape", msk.shape)
        pred_denorm = train_dataset.denormalize_x(pred)
        loss_denorm = myloss(pred_denorm, yy)
        train_l2_denorm += loss_denorm.item() * y.shape[0]

        optimizer.zero_grad()
        total_loss = loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        # break # todo : to remove
    train_l2_norm_avg, train_l2_denorm_avg = train_l2_norm/ntrain, train_l2_denorm/ntrain


    if args.use_writer:
        writer.add_scalar("train_loss_norm", train_l2_norm_avg, ep)
        writer.add_scalar("train_loss_denorm", train_l2_denorm_avg, ep)

    t_train += default_timer() -  t_1
    t_1 = default_timer()

    lr = optimizer.param_groups[0]['lr']
    if ep % args.save_everyepoch!=0:
        print('epoch {}, best epoch: {}, lr {:.2e}, train l2 norm {:.5f} train l2 denorm {:.5f}, time train avg {:.5f}'
                .format(ep, best_loss_epoch, lr, train_l2_norm_avg, train_l2_denorm_avg,  t_train / len(train_loader)))
    else:
        with torch.no_grad():
            model.eval()
            # compute spectrum once per epoch (first test batch)
            pred, target = [], []
            for xx, yy in val_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                # normalize it before the autoregressive predicting
                xx = train_dataset.normalize_x(xx)
                yy_norm = train_dataset.normalize_x(yy)
                for t in range(0, yy_norm.shape[-2], args.T_bundle):
                    # print("t", t)
                    y = yy_norm[..., t:t + args.T_bundle, :]
                    pred_step = model(xx)
                    
                    break # just test one step

                pred.append(pred_step)
                target.append(y)

            pred = torch.cat(pred, dim=0)
            target = torch.cat(target, dim=0)
            
            # denormalize the pred and target
            pred_denorm = train_dataset.denormalize_x(pred)
            target_denorm = train_dataset.denormalize_x(target)

            # print("pred shape", pred.shape, "target shape", target.shape)
            test_rel_l2_loss = loss_dict['rel_l2_loss'](pred_denorm, target_denorm)           

            # print("test_l2_step_avg", test_l2_step_avg.item())
            # print("test_l2_full_avg", test_l2_full_avg.item())
            if args.use_writer:
                for key, loss_func in loss_dict.items():
                    loss_metric = loss_func(pred_denorm, target_denorm)
                    if key != 'spectral_error':
                        writer.add_scalar(f"test_{key}", loss_metric.item(), ep)                
                    else:
                        for band_key in list(loss_metric.keys()): # only save  spec_low, spec_mid, spec_high
                            writer.add_scalar(f"test_{key}_{band_key}", loss_metric[band_key], ep)

        if test_rel_l2_loss < best_loss:
            best_loss = test_rel_l2_loss
            best_loss_epoch = ep
            if args.use_writer:
                # save error fft as well:
                torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': ep, 'scheduler': scheduler.state_dict(),
                            }, model_path,
                           )

        t_test = default_timer() - t_1
        t2 = t_1 = default_timer()
        
        # log a compact summary of the spectrum row (mean of first 20 bins)
        print('epoch {}, best epoch: {}, time {:.5f}, lr {:.2e}, train l2 norm {:.5f} train l2 denorm {:.5f}, test rel l2 loss {:.5f}, time train avg {:.5f} load avg {:.5f} test {:.5f}'
            .format(ep, best_loss_epoch, t2 - t1, lr, train_l2_norm_avg,  train_l2_denorm_avg, test_rel_l2_loss, t_train / len(train_loader), t_load / len(train_loader), t_test))




