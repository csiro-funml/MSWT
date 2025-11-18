"""
Train a neural operator to predict the next step of the PDE
This code is developed with reference to the following GitHub repo: DPOT: https://github.com/HaoZhongkai/DPOT/
"""

import sys
import os
# Add parent directory to Python path to access utils and models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint, resume_training_from_checkpoint, log_tensorboard_images_and_spectra
from utils.criterion import RelL2Norm, compute_error_fft, RMSE, BoundaryRMSE, MaxAbsError, GlobalMaxAbsError, SpectralError, FourierLoss1D, FourierLoss2D
from utils.griddataset import MixedTemporalDataset, TemporalDataset2D, LocalTemporalDataset2D, MemmapDedalusDataset2D, MemmapDedalusBigDataset2D
from utils.make_master_file import DATASET_DICT
# from models.fno import FNO2d
from models.fno import FNO2d_Tin1_Tout1 as FNO2d
from models.wavelet_transform import CrossWaveletTransformer, CrossWaveletTransSkipConnection
from models.wavelet_transform_exploration import WaveletTransformer
from models.high_frequency_scaling import ResUNet
# from models.unet import UNet_with_BottleneckHFS, UNet_withoutHFS
from models.hano import HANO2d
import pickle
from tqdm import tqdm


################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='FNO') # FNO, wavelet_transformer, HFS, UNet, HANO, UNO 
parser.add_argument('--dataset',type=str, default='ns2d_dedalus_big') # ['ns2d_fno_1e-3', 'ns2d_pda', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 'sw2d_pda'], note: pdb is the pde bench
parser.add_argument('--resume_path',type=str, default='')
parser.add_argument('--use_writer', action='store_true',default=False)
parser.add_argument('--form',type=str, default='vorticity', choices=['vorticity', 'velocity'])


# ### dataset details
parser.add_argument('--T_in', type=int, default=1)
parser.add_argument('--T_out', type=int, default=1, help='Number of steps ahead to predict (1 for one-step, 5 for five-step ahead)')
parser.add_argument('--T_bundle', type=int, default=1)

parser.add_argument('--pad', type=int, default=0)
parser.add_argument('--normalize',type=int, default=1)
parser.add_argument('--normalize_strategy',type=str, default='zscore')

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
parser.add_argument('--loss_type', type=str, default='rel_l2', choices=['fourier', 'fourier2d', 'rel_l2'])
parser.add_argument('--fourier_logscale', type=str, default='False', choices=['True', 'False'])

parser.add_argument('--save_everyepoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb','lion'])
parser.add_argument('--beta1',type=float,default=0.9)
parser.add_argument('--beta2',type=float,default=0.9)
parser.add_argument('--lr_method',type=str, default='cossin') # cyclic for ViT perhaps
parser.add_argument('--grad_clip',type=float, default=10000.0)
parser.add_argument('--step_size', type=int, default=20)
parser.add_argument('--step_gamma', type=float, default=0.5)
parser.add_argument('--warmup_epochs',type=int, default=0)

# Performance optimization arguments
parser.add_argument('--use_compile', action='store_true', default=False, help='Use torch.compile for faster training (PyTorch 2.0+)')
parser.add_argument('--num_workers', type=int, default=None, help='Number of DataLoader workers (default: auto-detect)')
parser.add_argument('--pin_memory', action='store_true', default=True, help='Pin memory for faster CPU->GPU transfers')
parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches to prefetch')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps (effective batch size = batch_size * steps)')

parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='/scratch3/wan410/operator_learning_model/')

args = parser.parse_args()
args.fourier_logscale = args.fourier_logscale == 'True'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Performance optimizations
if torch.cuda.is_available():
    # Enable cuDNN benchmark for consistent input sizes (faster convolutions)
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for faster training on Ampere+ GPUs (H100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Clear cache at start to reduce fragmentation
    torch.cuda.empty_cache()
    print(f"CUDA optimizations enabled: benchmark={torch.backends.cudnn.benchmark}, TF32={torch.backends.cuda.matmul.allow_tf32}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB total")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

print(f"Current working directory: {os.getcwd()}")



################################################################
# load some toy data to run locally
if not torch.cuda.is_available() and args.dataset != 'ns2d_dedalus':
    train_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_out, n_channels=3, normalize=args.normalize, train='train')
    test_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_out, n_channels=3, normalize=args.normalize, train='test')
    val_dataset= test_dataset
elif args.dataset == 'ns2d_dedalus_small':
    # train_dataset = DedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form='vorticity', normalize=args.normalize, train='train')
    # test_dataset = DedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, form='vorticity', normalize=args.normalize, train='test')
    # val_dataset= DedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, form='vorticity', normalize=args.normalize, train='val')
    train_dataset = MemmapDedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_out, form='vorticity', normalize=args.normalize, train='train', strategy=args.normalize_strategy)
    test_dataset = MemmapDedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_out, form='vorticity', normalize=args.normalize, train='test', strategy=args.normalize_strategy)
    val_dataset= MemmapDedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_out, form='vorticity', normalize=args.normalize, train='val', strategy=args.normalize_strategy)
    # Mean/STD statistics are more consitent between training/validation/test sets
    train_dataset.predict_normalizing_statistics()
    test_dataset.predict_normalizing_statistics()
    val_dataset.predict_normalizing_statistics()
elif args.dataset == 'ns2d_dedalus_big':
    train_dataset = MemmapDedalusBigDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_out, form=args.form, normalize=args.normalize, train='train', strategy=args.normalize_strategy)
    test_dataset = MemmapDedalusBigDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_out, form=args.form, normalize=args.normalize, train='test', strategy=args.normalize_strategy)
    val_dataset = MemmapDedalusBigDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_out, form=args.form, normalize=args.normalize, train='val', strategy=args.normalize_strategy)
else:
    # load data and dataloader
    train_dataset = TemporalDataset2D(args.dataset, t_in = args.T_in, t_ar = args.T_out, train='train', normalize=args.normalize)
    val_dataset =  TemporalDataset2D(args.dataset,  t_in = args.T_in, t_ar =-1, train='val', normalize=args.normalize)
    test_dataset = TemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels = train_dataset.n_channels, train='test', normalize=args.normalize)



# Determine number of workers
if args.num_workers is None:
    if torch.cuda.is_available():
        # Auto-detect: use min(CPU count, 16) for optimal performance
        import os
        num_workers = min(os.cpu_count() or 8, 16)
    else:
        num_workers = 0
else:
    num_workers = args.num_workers

print(f"DataLoader settings: num_workers={num_workers}, pin_memory={args.pin_memory}, prefetch_factor={args.prefetch_factor}")

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=args.pin_memory if torch.cuda.is_available() else False,
    prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
    persistent_workers=num_workers > 0  # Keep workers alive between epochs
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=args.batch_size, 
    shuffle=False,
    num_workers=num_workers,
    pin_memory=args.pin_memory if torch.cuda.is_available() else False,
    prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
    persistent_workers=num_workers > 0
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=args.batch_size, 
    shuffle=False,
    num_workers=num_workers,
    pin_memory=args.pin_memory if torch.cuda.is_available() else False,
    prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
    persistent_workers=num_workers > 0
)

ntrain, ntest = len(train_dataset), len(test_dataset)
# if not args.pad:
#     args.res = train_dataset.res  # use original dataset  resolution to train the model
testing_mode = 'FNO_testing'
if testing_mode == 'FNO_testing':
    # comment = args.comment + '{}_{}_mod{}_wid{}_lay{}_ntrain{}_normalizer_{}_form_{}'.format(args.dataset, args.model, args.modes, args.width, args.n_layers, ntrain, args.normalize_strategy, args.form)
    comment = args.comment + f'{args.dataset}_{args.model}_mod{args.modes}_wid{args.width}_lay{args.n_layers}_ntrain{ntrain}_form{args.form}_loss{args.loss_type}_logscale{args.fourier_logscale}_warmup{args.warmup_epochs}_Tout{args.T_out}'
    log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
    # model_path = log_path + '/model.pth'
    model_path = log_path + f'/model_epochs_{args.epochs}.pth' # I will test a longer training epoch
else:
    comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
    log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
    # model_path = log_path + '/model.pth'
    model_path = log_path + f'/model_epochs_{args.epochs}.pth' # I will test a longer training epoch
# model_path = log_path + f'/model_epochs_{args.epochs}_patchsize_{args.patch_size}.pth'
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
    # if args.dataset == 'ns2d_dedalus':
    # model = FNO2d(args.modes, args.modes, width=args.width,
    #             n_channels=train_dataset.n_channels,
    #             in_timesteps = args.T_in, out_timesteps=1, 
    #             n_layers = args.n_layers, 
    #         #   normalize=args.normalize
    #             ).to(device)
    model = FNO2d(args.modes, args.modes, width=args.width,
                img_size=train_dataset.res,
                in_channels=train_dataset.n_channels_in[args.form],out_channels=train_dataset.n_channels_out[args.form],
                in_timesteps = args.T_in, out_timesteps=1, 
                n_layers = args.n_layers,
                use_ln=True,
                # normalize=args.normalize, 
                 ).to(device)
elif args.model == 'wavelet_transformer':
    model = CrossWaveletTransformer(wave='haar', n_channels=train_dataset.n_channels, in_timesteps = args.T_in, dim=512, depth=8).to(device)
elif args.model == 'HFS':
    model =  ResUNet(in_c = train_dataset.n_channels * args.T_in + 2 ,out_c = train_dataset.n_channels, 
                     bottleneck_feature=512, 
                     device=device).to(device)
elif args.model == 'wavelet_transformer_skip':
    model = CrossWaveletTransSkipConnection(wave='haar', n_channels=train_dataset.n_channels, in_timesteps = args.T_in, dim=512, depth=8).to(device)
elif args.model == 'WaveletTransV2':
    model = WaveletTransformer(in_timesteps = args.T_in, in_chans=train_dataset.n_channels, out_chans=train_dataset.n_channels
                              ,output_size=(train_dataset.res[0], train_dataset.res[1])).to(device)
elif args.model == 'HANO':
    model = HANO2d(T_in=args.T_in, T_out=args.T_ar, res_output=train_dataset.res[0],  res_att=train_dataset.res[0],
                   in_dim=train_dataset.n_channels, out_dim=train_dataset.n_channels, feature_dim=256).to(device)
else:
    raise NotImplementedError


#### set optimizer
if args.opt == 'lamb':
    optimizer = Lamb(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2), adam=True, debias=False,weight_decay=1e-4)
else:
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-6)


# Calculate effective number of steps (accounts for gradient accumulation)
# With gradient accumulation, we step the optimizer less frequently
effective_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
total_effective_steps = args.epochs * effective_steps_per_epoch

if args.lr_method == 'cycle':
    print('Using cycle learning rate schedule')
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, div_factor=1e4, pct_start=(args.warmup_epochs / args.epochs), final_div_factor=1e4, steps_per_epoch=effective_steps_per_epoch, epochs=args.epochs)
elif args.lr_method == 'step':
    print('Using step learning rate schedule')
    scheduler = StepLR(optimizer, step_size=args.step_size * effective_steps_per_epoch, gamma=args.step_gamma)
elif args.lr_method == 'warmup':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: min((steps + 1) / (args.warmup_epochs * effective_steps_per_epoch), np.power(args.warmup_epochs * effective_steps_per_epoch / float(steps + 1), 0.5)))
elif args.lr_method == 'linear':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: (1 - steps / total_effective_steps))
elif args.lr_method == 'restart':
    print('Using cos anneal restart')
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=effective_steps_per_epoch * args.lr_step_size, eta_min=0.)
elif args.lr_method == 'cyclic':
    scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=args.lr_step_size * effective_steps_per_epoch, mode='triangular2', cycle_momentum=False)
elif args.lr_method == 'cossin':
    scheduler = CosineAnnealingLR(optimizer, T_max=total_effective_steps)
else:
    raise NotImplementedError

print(f"Scheduler steps per epoch: {effective_steps_per_epoch} (with gradient_accumulation_steps={args.gradient_accumulation_steps})")

# Model compilation (PyTorch 2.0+)
# Note: Requires Triton to be installed. If not available, compilation will be skipped.
if args.use_compile:
    try:
        # Compile model for faster training
        model = torch.compile(model, mode='reduce-overhead')  # or 'max-autotune' for best performance
        print("Model compiled with torch.compile (mode=reduce-overhead)")
    except Exception as e:
        print(f"Warning: torch.compile failed ({e})")
        print("Continuing without compilation. This is normal if Triton is not installed.")
        print("To install Triton: pip install triton")
        args.use_compile = False

start_epoch = 0
best_loss_epoch = 0
print(model)
count_parameters(model)

# Print memory usage after model initialization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Memory after model init: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.2f} GB total")
    print(f"Available memory: {total - reserved:.2f} GB")

if args.resume_path:
    print('Loading models and resume from {}'.format(model_path))
    args.resume_path = model_path
    model, optimizer, scheduler, start_epoch = resume_training_from_checkpoint(model, args.resume_path, device, optimizer, scheduler)
    print("resume training from epoch:", start_epoch)
    best_loss_epoch = start_epoch


################################################################
# Main function for pretraining
################################################################
# Initialize loss functions based on args
# For warmup: use RelL2Norm during warmup, then switch to specified loss
final_loss = None  # Will be set if warmup is used
if args.warmup_epochs == 0:
    # No warmup: use specified loss from the start
    if args.loss_type == 'rel_l2':
        myloss = RelL2Norm(size_average=False)
    elif args.loss_type == 'fourier':
        myloss = FourierLoss1D(log_scale=args.fourier_logscale)
    elif args.loss_type == 'fourier2d':
        myloss = FourierLoss2D(log_scale=args.fourier_logscale)
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")
else:
    # With warmup: start with RelL2Norm, will switch to specified loss after warmup
    myloss = RelL2Norm(size_average=False)
    # Pre-create the final loss function for after warmup
    if args.loss_type == 'fourier':
        final_loss = FourierLoss1D(log_scale=args.fourier_logscale)
    elif args.loss_type == 'fourier2d':
        final_loss = FourierLoss2D(log_scale=args.fourier_logscale)
    elif args.loss_type == 'rel_l2':
        final_loss = RelL2Norm(size_average=False)
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")

loss_dict = {} # for testing
loss_dict['rel_l2_loss'] = RelL2Norm() # rel L2 loss
loss_dict['rmse'] = RMSE()
loss_dict['boundary_rmse'] = BoundaryRMSE()
loss_dict['max_avg'] = MaxAbsError()
loss_dict['max_global'] = GlobalMaxAbsError()
loss_dict['spectral_error'] = SpectralError(model_name=args.model, save_path=log_path, low_percentile=0.70, high_percentile=0.97)

print(f"Loss configuration: warmup_epochs={args.warmup_epochs}, loss_type={args.loss_type}, fourier_logscale={args.fourier_logscale}")
if args.warmup_epochs > 0:
    print(f"  Using RelL2Norm for epochs 0-{args.warmup_epochs-1}, then switching to {args.loss_type}")
else:
    print(f"  Using {args.loss_type} from the start")


best_loss = np.inf
# Store a grayscale heatmap of error spectrum over epochs (C=1, W= num_saved_epochs, H=num_bins)
num_bins = min(train_dataset.res[0], train_dataset.res[1]) // 2

pbar = tqdm(range(start_epoch, args.epochs))
for ep in pbar:
    model.train()

    # Track which loss function is currently being used for training
    # This helps determine which loss to use for model saving
    current_loss_type = 'rel_l2'  # Default during warmup
    if args.warmup_epochs == 0:
        # No warmup: use specified loss from the start
        current_loss_type = args.loss_type
    elif ep >= args.warmup_epochs:
        # After warmup: switch to specified loss
        current_loss_type = args.loss_type
        if ep == args.warmup_epochs:
            myloss = final_loss
            print(f"\nSwitching to {args.loss_type} loss at epoch {ep}")
    
    # Determine T_out for this epoch: use 1 during warmup for training, then switch to specified T_out
    # For evaluation, always use args.T_out if > 1 (regardless of warmup)
    current_T_out = 1 if (args.warmup_epochs > 0 and ep < args.warmup_epochs) else args.T_out
    eval_T_out = args.T_out if args.T_out > 1 else 1  # For evaluation, use T_out if > 1, start from ep=0
    if ep == args.warmup_epochs and args.warmup_epochs > 0:
        print(f"\nSwitching to T_out={current_T_out} (multi-step ahead prediction) at epoch {ep}")

    t1 = t_1 = default_timer()
    t_load, t_train = 0., 0.
    train_l2_norm = 0
    train_l2_denorm = 0
    train_pred_loss = 0  # For tracking pred_loss from FourierLoss
    train_fft_loss = 0  # For tracking fft_loss from FourierLoss

    loss_previous = 10000

    # Zero gradients at the start of each epoch (for gradient accumulation)
    optimizer.zero_grad()
    
    for batch_id, data_batch in enumerate(train_loader):
        # Handle both old format (xx, yy) and new format (xx, yy, forcing_y)
        if len(data_batch) == 3:
            xx, yy, forcing_y = data_batch  # (B, H, W, T_in, C), (B, H, W, T_ar, C), (B, H, W, T_out-1, 2), forcing is used as inputs for the next step
            forcing_y = forcing_y.to(device, non_blocking=True) if forcing_y is not None else None
            forcing_y = train_dataset.normalize_forcing_tensor(forcing_y, args.form)
        else:
            xx, yy = data_batch
            forcing_y = None  # Will extract from xx if not provided
        
        t_load += default_timer() - t_1
        t_1 = default_timer()
        loss = 0.
        batch_pred_loss = 0.
        batch_fft_loss = 0.
        
        # Non-blocking transfer for faster data loading
        xx = xx.to(device, non_blocking=True)  ## B, n, n, T_in, C
        yy = yy.to(device, non_blocking=True)  ## B, n, n, T_ar, C (ground truth unnormalized output)
        
        # range check
        if ep == 0 and batch_id == 0:
            C = xx.shape[-1]
            x_temp = xx.reshape((-1, C))
            for c_i in range(C):
                print("channel: %s range before normalization "%c_i, x_temp[:, c_i].max().item(), x_temp[:, c_i].min().item())
        
        # normalize it before the autoregressive predicting
        xx = train_dataset.normalize_x(xx)
        yy_norm = train_dataset.normalize_x(yy) # ground truth normalized output
        
        if current_T_out == 1:
            # One-step ahead: original behavior
            time_indices = range(0, yy_norm.shape[-2], args.T_bundle)
            pred_norm_list = []

            for t in time_indices:
                y = yy_norm[..., t:t + args.T_bundle, :]  # (B, H, W, 1, C)
                pred_norm = model(xx)  # (B, H, W, 1, C_out) - normalized
                pred_norm_list.append(pred_norm)
            pred_norm = torch.cat(pred_norm_list, dim=-2)  # (B, H, W, T_bundle, C_out)
            # Compute loss on normalized predictions
            loss_output = myloss(pred_norm, yy_norm)
        else:
            # Multi-step ahead prediction: autoregressive for T_out steps
            pred_norm_list = []
            x_current = xx  # Start with normalized input (B, H, W, T_in, C_in)
            
            for step_idx in range(current_T_out):
                # Use last timestep if T_in > 1
                if x_current.shape[-2] > 1:
                    x_input = x_current[..., -1:, :]  # (B, H, W, 1, C_in)
                else:
                    x_input = x_current  # (B, H, W, 1, C_in)
                
                # Predict one step (x_input is already normalized)
                pred_step_norm = model(x_input)  # (B, H, W, 1, C_out) - normalized output
                pred_norm_list.append(pred_step_norm)
                
                # Prepare input for next step: combine predicted main variables with forcing
                if step_idx < current_T_out - 1:  # Don't need to prepare next input for last step
                    # Get forcing from ground truth for the next step
                    # forcing_y shape: (B, H, W, T_out-1, 2) - forcing for steps 1 to T_out-1
                    forcing = forcing_y[..., step_idx:step_idx+1, :]  # (B, H, W, 1, 2) - already normalized
                    
                    # Concatenate predicted main variables (normalized) with forcing (normalized)
                    x_next = torch.cat((pred_step_norm, forcing), dim=-1)  # (B, H, W, 1, C_in)
                    # Update x_current for next iteration (use last T_in timesteps)
                    x_current = x_next
            
            # Stack all predictions: (B, H, W, T_out, C_out)
            pred_norm = torch.cat(pred_norm_list, dim=-2)  # Normalized for loss computation
            # Compute loss on normalized predictions vs normalized targets
            loss_output = myloss(pred_norm, yy_norm[..., :current_T_out, :])
        
        # Handle loss output (tuple for FourierLoss, scalar for RelL2Norm)
        if isinstance(loss_output, tuple):
            loss += loss_output[0]  # Total loss
            batch_pred_loss += loss_output[1].item()  # pred_loss
            batch_fft_loss += loss_output[2].item()  # fft_loss
        else:
            loss += loss_output
            batch_pred_loss += loss_output.item()
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / args.gradient_accumulation_steps
        # Get batch size from xx (input) for logging
        batch_size = xx.shape[0]
        train_l2_norm += loss.item() * batch_size  # Store unscaled loss for logging
        train_pred_loss += batch_pred_loss * batch_size
        train_fft_loss += batch_fft_loss * batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{ep}/{args.epochs}")
        # print("train input shape", xx.shape, "output shape", yy.shape, "pred shape", pred.shape, "mask shape", msk.shape)
        
        # Denormalize for monitoring (pred is already denormalized from the loop)
        with torch.no_grad():
            # pred is already denormalized, yy is the original (denormalized) target
            # Use the same number of steps as pred
            pred = train_dataset.denormalize_x(pred_norm)
            yy_subset = yy[..., :pred.shape[-2], :]  # Match pred's time dimension
            loss_denorm_output = myloss(pred, yy_subset)
            if isinstance(loss_denorm_output, tuple):
                loss_denorm = loss_denorm_output[1]  # pred_loss component from FourierLoss
            else:
                loss_denorm = loss_denorm_output  # RelL2Norm returns scalar
            train_l2_denorm += loss_denorm.item() * batch_size

        # Backward pass (use scaled loss for gradient accumulation)
        scaled_loss.backward()

        # Gradient accumulation: only step optimizer every N steps
        if (batch_id + 1) % args.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            # Clear cache periodically to reduce memory fragmentation
            if torch.cuda.is_available() and (batch_id + 1) % (args.gradient_accumulation_steps * 10) == 0:
                torch.cuda.empty_cache()
        # break # todo : to remove
    
    # Handle remaining gradients if gradient_accumulation_steps doesn't divide evenly
    if len(train_loader) % args.gradient_accumulation_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    
    # Clear cache at end of epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    train_l2_norm_avg, train_l2_denorm_avg = train_l2_norm/ntrain, train_l2_denorm/ntrain
    train_pred_loss_avg = train_pred_loss / ntrain
    train_fft_loss_avg = train_fft_loss / ntrain

    if args.use_writer:
        writer.add_scalar("train_loss_norm", train_l2_norm_avg, ep)
        writer.add_scalar("train_loss_denorm", train_l2_denorm_avg, ep)
        # Log pred_loss and fft_loss (pred_loss should match rel_l2_loss for comparison)
        writer.add_scalar("train_pred_loss", train_pred_loss_avg, ep)  # Same as rel_l2_loss
        writer.add_scalar("train_fft_loss", train_fft_loss_avg, ep)

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
            pred_list, target_list = [], []
            for data_batch in val_loader:
                # Handle both old format (xx, yy) and new format (xx, yy, forcing_y)
                if len(data_batch) == 3:
                    xx, yy, forcing_y = data_batch
                    forcing_y = forcing_y.to(device)
                    forcing_y = val_dataset.normalize_forcing_tensor(forcing_y, args.form)
                else:
                    xx, yy = data_batch
                    forcing_y = None
                
                xx = xx.to(device)
                yy = yy.to(device)
                # normalize it before the autoregressive predicting
                xx = train_dataset.normalize_x(xx)
                yy_norm = train_dataset.normalize_x(yy)
                
                # Use eval_T_out for evaluation (multi-step if T_out > 1, regardless of warmup)
                if eval_T_out == 1:
                    # One-step ahead evaluation
                    for t in range(0, yy_norm.shape[-2], args.T_bundle):
                        y = yy_norm[..., t:t + args.T_bundle, :]
                        pred_step = model(xx)
                        break # just test one step
                    pred_list.append(pred_step) # (B, H, W, T_bundle, C_out) normalized prediction
                    target_list.append(y) # (B, H, W, T_bundle, C_out) normalized ground truth
                else:                
                    # Multi-step ahead evaluation
                    pred_steps = []
                    x_current = xx  # (B, H, W, T_in, C_in)
                    
                    for step_idx in range(eval_T_out):
                        # Use last timestep if T_in > 1
                        if x_current.shape[-2] > 1:
                            x_input = x_current[..., -1:, :]  # (B, H, W, 1, C_in)
                        else:
                            x_input = x_current  # (B, H, W, 1, C_in)
                        
                        pred_step = model(x_input)  # (B, H, W, 1, C_out)
                        pred_steps.append(pred_step)
                        
                        # Prepare input for next step
                        if step_idx < eval_T_out - 1:
                            # Get forcing from ground truth for the next step
                            if forcing_y is not None:
                                # forcing_y shape: (B, H, W, T_out-1, 2) - forcing for steps 1 to T_out-1
                                forcing = forcing_y[..., step_idx:step_idx+1, :]  # (B, H, W, 1, 2)
                                x_next = torch.cat((pred_step, forcing), dim=-1)  # (B, H, W, 1, C_in)
                            else:
                                x_next = pred_step
                            x_current = x_next
                    
                    pred = torch.cat(pred_steps, dim=-2)  # (B, H, W, T_out, C_out) normalized prediction
                    pred_list.append(pred)
                    target_list.append(yy_norm[..., :pred.shape[-2], :])  # (B, H, W, T_out, C_out) normalized ground truth

            pred = torch.cat(pred_list, dim=0)
            target = torch.cat(target_list, dim=0)
            
            # denormalize the pred and target
            pred_denorm = train_dataset.denormalize_x(pred)
            target_denorm = train_dataset.denormalize_x(target)

            # print("pred shape", pred.shape, "target shape", target.shape)
            # Always compute RelL2Loss for comparison
            test_rel_l2_loss = loss_dict['rel_l2_loss'](pred_denorm, target_denorm)
            
            # Always compute FourierLoss for comparison (regardless of current training loss)
            # This allows comparing different training configurations
            val_fourier_loss_1d = FourierLoss1D(log_scale=args.fourier_logscale)
            fourier_output_1d = val_fourier_loss_1d(pred_denorm, target_denorm)
            test_fourier_l2loss_1d = fourier_output_1d[0].item()  # Total FourierLoss1D (pred_loss + beta * fft_loss)
            test_fourier_pred_loss_1d = fourier_output_1d[1].item()  # pred_loss component from FourierLoss1D
            test_fourier_fft_loss_1d = fourier_output_1d[2].item()
            
            # Also compute FourierLoss2D for comparison
            val_fourier_loss_2d = FourierLoss2D(log_scale=args.fourier_logscale)
            fourier_output_2d = val_fourier_loss_2d(pred_denorm, target_denorm)
            test_fourier_l2loss_2d = fourier_output_2d[0].item()  # Total FourierLoss2D (pred_loss + beta * fft_loss)
            test_fourier_pred_loss_2d = fourier_output_2d[1].item()  # pred_loss component from FourierLoss2D
            test_fourier_fft_loss_2d = fourier_output_2d[2].item()
            
            # Determine which loss to use for model saving based on current training loss
            # Use the loss that matches what we're currently training with
            if current_loss_type == 'fourier':
                loss_for_saving = test_fourier_l2loss_1d
            elif current_loss_type == 'fourier2d':
                loss_for_saving = test_fourier_l2loss_2d
            else:
                loss_for_saving = test_rel_l2_loss.item()

            # print("test_l2_step_avg", test_l2_step_avg.item())
            # print("test_l2_full_avg", test_l2_full_avg.item())
            if args.use_writer:
                # Log scalar metrics
                for key, loss_func in loss_dict.items():
                    loss_metric = loss_func(pred_denorm, target_denorm)
                    if key != 'spectral_error':
                        writer.add_scalar(f"test_{key}", loss_metric.item(), ep)                
                    else:
                        for band_key in list(loss_metric.keys()): # only save  spec_low, spec_mid, spec_high
                            writer.add_scalar(f"test_{key}_{band_key}", loss_metric[band_key], ep)
                
                # Always log all losses for comparison across different training configurations
                # Log pred_loss with same name for comparison (from FourierLoss1D, which is RelL2Norm component)
                writer.add_scalar("test_pred_loss_1d", test_fourier_pred_loss_1d, ep)
                
                # Always log FourierLoss1D components (for comparison even when not using fourier loss)
                writer.add_scalar("test_fourier_l2loss_1d", test_fourier_l2loss_1d, ep)
                writer.add_scalar("test_fft_loss_1d", test_fourier_fft_loss_1d, ep)
                
                # Always log FourierLoss2D components (for comparison even when not using fourier2d loss)
                writer.add_scalar("test_pred_loss_2d", test_fourier_pred_loss_2d, ep)
                writer.add_scalar("test_fourier_l2loss_2d", test_fourier_l2loss_2d, ep)
                writer.add_scalar("test_fft_loss_2d", test_fourier_fft_loss_2d, ep)
                
                # Log images and spectra using utility function
                log_tensorboard_images_and_spectra(
                    writer=writer,
                    pred_denorm=pred_denorm,
                    target_denorm=target_denorm,
                    epoch=ep,
                    form=args.form,
                    model_name=args.model
                )
        # Use the appropriate loss for model saving based on current training loss
        if loss_for_saving < best_loss:
            best_loss = loss_for_saving
            best_loss_epoch = ep
            if args.use_writer:
                # save error fft as well:
                torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': ep, 'scheduler': scheduler.state_dict(),
                            }, model_path,
                           )

        t_test = default_timer() - t_1
        t2 = t_1 = default_timer()
        
        # log a compact summary of the spectrum row (mean of first 20 bins)
        # Always log all losses for comparison
        log_str = 'epoch {}, best epoch: {}, time {:.5f}, lr {:.2e}, train l2 norm {:.5f} train l2 denorm {:.5f}, test rel l2 loss {:.5f}, test pred loss 1d {:.5f}, test fourier l2loss 1d {:.5f}, test fft loss 1d {:.5f}, test pred loss 2d {:.5f}, test fourier l2loss 2d {:.5f}, test fft loss 2d {:.5f}'.format(
            ep, best_loss_epoch, t2 - t1, lr, train_l2_norm_avg, train_l2_denorm_avg, 
            test_rel_l2_loss, test_fourier_pred_loss_1d, test_fourier_l2loss_1d, test_fourier_fft_loss_1d,
            test_fourier_pred_loss_2d, test_fourier_l2loss_2d, test_fourier_fft_loss_2d)
        
        log_str += ', time train avg {:.5f} load avg {:.5f} test {:.5f}'.format(
            t_train / len(train_loader), t_load / len(train_loader), t_test)
        
        print(log_str)




