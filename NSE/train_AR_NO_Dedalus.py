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
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint, resume_training_from_checkpoint
from utils.criterion import RelL2Norm, compute_error_fft, RMSE, BoundaryRMSE, MaxAbsError, GlobalMaxAbsError, SpectralError
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
# helper functions
################################################################

def _to_rgb_minmax(image_2d: torch.Tensor) -> torch.Tensor:
    """Convert single-channel 2D field to 3-channel RGB with per-image min-max normalization.
    
    Args:
        image_2d: (H, W) tensor
        
    Returns:
        (3, H, W) tensor with RGB channels
    """
    img = image_2d.detach().float()
    min_val = torch.amin(img)
    max_val = torch.amax(img)
    if torch.isfinite(min_val) and torch.isfinite(max_val) and (max_val > min_val):
        img = (img - min_val) / (max_val - min_val)
    else:
        img = torch.zeros_like(img)
    return img.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)


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
parser.add_argument('--T_ar', type=int, default=1)
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

# Performance optimization arguments
parser.add_argument('--use_amp', action='store_true', default=False, help='Use automatic mixed precision (FP16/BF16)')
parser.add_argument('--use_compile', action='store_true', default=False, help='Use torch.compile for faster training (PyTorch 2.0+)')
parser.add_argument('--num_workers', type=int, default=None, help='Number of DataLoader workers (default: auto-detect)')
parser.add_argument('--pin_memory', action='store_true', default=True, help='Pin memory for faster CPU->GPU transfers')
parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches to prefetch')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps (effective batch size = batch_size * steps)')

parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='/scratch3/wan410/operator_learning_model/')

args = parser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Performance optimizations
if torch.cuda.is_available():
    # Enable cuDNN benchmark for consistent input sizes (faster convolutions)
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for faster training on Ampere+ GPUs (H100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"CUDA optimizations enabled: benchmark={torch.backends.cudnn.benchmark}, TF32={torch.backends.cuda.matmul.allow_tf32}")

print(f"Current working directory: {os.getcwd()}")



################################################################
# load some toy data to run locally
if not torch.cuda.is_available() and args.dataset != 'ns2d_dedalus':
    train_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, n_channels=3, normalize=args.normalize, train='train')
    test_dataset = LocalTemporalDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, n_channels=3, normalize=args.normalize, train='test')
    val_dataset= test_dataset
elif args.dataset == 'ns2d_dedalus_small':
    # train_dataset = DedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form='vorticity', normalize=args.normalize, train='train')
    # test_dataset = DedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, form='vorticity', normalize=args.normalize, train='test')
    # val_dataset= DedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=-1, form='vorticity', normalize=args.normalize, train='val')
    train_dataset = MemmapDedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form='vorticity', normalize=args.normalize, train='train', strategy=args.normalize_strategy)
    test_dataset = MemmapDedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form='vorticity', normalize=args.normalize, train='test', strategy=args.normalize_strategy)
    val_dataset= MemmapDedalusDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form='vorticity', normalize=args.normalize, train='val', strategy=args.normalize_strategy)
    # Mean/STD statistics are more consitent between training/validation/test sets
    train_dataset.predict_normalizing_statistics()
    test_dataset.predict_normalizing_statistics()
    val_dataset.predict_normalizing_statistics()
elif args.dataset == 'ns2d_dedalus_big':
    train_dataset = MemmapDedalusBigDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form=args.form, normalize=args.normalize, train='train', strategy=args.normalize_strategy)
    test_dataset = MemmapDedalusBigDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form=args.form, normalize=args.normalize, train='test', strategy=args.normalize_strategy)
    val_dataset = MemmapDedalusBigDataset2D(args.dataset, t_in=args.T_in, t_ar=args.T_ar, form=args.form, normalize=args.normalize, train='val', strategy=args.normalize_strategy)
else:
    # load data and dataloader
    train_dataset = TemporalDataset2D(args.dataset, t_in = args.T_in, t_ar = args.T_ar, train='train', normalize=args.normalize)
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
    comment = args.comment + '{}_{}_mod{}_wid{}_lay{}_ntrain{}_normalizer_{}'.format(args.dataset, args.model, args.modes, args.width, args.n_layers, ntrain, args.normalize_strategy)
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
elif args.opt == 'lion':
    optimizer = Lion(model.parameters(), lr=args.lr, weight_decay = 0.01)
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

# Mixed precision training setup
if args.use_amp:
    # Use BF16 on H100 (better than FP16 for training stability)
    # Check if BF16 is supported (H100 supports it)
    try:
        if torch.cuda.is_bf16_supported():
            scaler = None  # BF16 doesn't need gradient scaling
            dtype = torch.bfloat16
            use_bf16 = True
            print("Using BF16 mixed precision training (no gradient scaling needed)")
        else:
            raise AttributeError("BF16 not supported")
    except (AttributeError, RuntimeError):
        scaler = torch.cuda.amp.GradScaler()
        dtype = torch.float16
        use_bf16 = False
        print("Using FP16 mixed precision training (with gradient scaling)")
else:
    scaler = None
    dtype = torch.float32
    use_bf16 = False
    print("Using FP32 training (no mixed precision)")

# Model compilation (PyTorch 2.0+)
if args.use_compile:
    try:
        # Compile model for faster training
        model = torch.compile(model, mode='reduce-overhead')  # or 'max-autotune' for best performance
        print("Model compiled with torch.compile (mode=reduce-overhead)")
    except Exception as e:
        print(f"Warning: torch.compile failed ({e}), continuing without compilation")
        args.use_compile = False

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

    # Zero gradients at the start of each epoch (for gradient accumulation)
    optimizer.zero_grad()
    
    for batch_id, (xx, yy) in enumerate(train_loader):
        t_load += default_timer() - t_1
        t_1 = default_timer()
        loss = 0.
        
        # Non-blocking transfer for faster data loading
        xx = xx.to(device, non_blocking=True)  ## B, n, n, T_in, C
        yy = yy.to(device, non_blocking=True)  ## B, n, n, T_ar, C
        
        # range check
        if ep == 0 and batch_id == 0:
            C = xx.shape[-1]
            x_temp = xx.reshape((-1, C))
            for c_i in range(C):
                print("channel: %s range before normalization "%c_i, x_temp[:, c_i].max().item(), x_temp[:, c_i].min().item())
        
        # normalize it before the autoregressive predicting
        xx = train_dataset.normalize_x(xx)
        yy_norm = train_dataset.normalize_x(yy)
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=dtype if args.use_amp else None):
            for t in range(0, yy_norm.shape[-2], args.T_bundle):
                y = yy_norm[..., t:t + args.T_bundle, :]
                # print('input shape', xx.shape)
                pred = model(xx)  # give the normalized output to the autoregressive predicting
                # print("pred shape", pred.shape, "y shape", y.shape)
                loss += myloss(pred, y)
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / args.gradient_accumulation_steps
        train_l2_norm += loss.item() * y.shape[0]  # Store unscaled loss for logging

        pbar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{ep}/{args.epochs}")
        # print("train input shape", xx.shape, "output shape", yy.shape, "pred shape", pred.shape, "mask shape", msk.shape)
        
        # Denormalize for monitoring (keep in full precision)
        with torch.no_grad():
            pred_denorm = train_dataset.denormalize_x(pred.float() if args.use_amp else pred)
            loss_denorm = myloss(pred_denorm, yy)
            train_l2_denorm += loss_denorm.item() * y.shape[0]

        # Backward pass with mixed precision (use scaled loss for gradient accumulation)
        if scaler is not None:
            # FP16 with gradient scaling
            scaler.scale(scaled_loss).backward()
        else:
            # BF16 or FP32: no scaling needed
            scaled_loss.backward()

        # Gradient accumulation: only step optimizer every N steps
        if (batch_id + 1) % args.gradient_accumulation_steps == 0:
            if scaler is not None:
                # FP16: unscale gradients before clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # BF16 or FP32: clip and step directly
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        # break # todo : to remove
    
    # Handle remaining gradients if gradient_accumulation_steps doesn't divide evenly
    if len(train_loader) % args.gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
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
                # write the pred and target as RGB images for TensorBoard
                pred_img = _to_rgb_minmax(pred_denorm[0, :, :, 0, 0])
                target_img = _to_rgb_minmax(target_denorm[0, :, :, 0, 0])
                error_img = _to_rgb_minmax(pred_denorm[0, :, :, 0, 0] - target_denorm[0, :, :, 0, 0])
                writer.add_image("model pred", pred_img, ep)
                writer.add_image("ground truth", target_img)
                writer.add_image("error", error_img, ep)
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




