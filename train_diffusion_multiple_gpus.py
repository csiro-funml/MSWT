# ---------------------------------------------------------------------------------------------
# Author: Vivek Oommen
# Date: 08/01/2024
# ---------------------------------------------------------------------------------------------

import os
import sys

import math
import time
import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import argparse
from torch.utils.tensorboard import SummaryWriter
from models.diff_Unet import Unet
from models.diffusion import ElucidatedDiffusion

# Distributed training imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

torch.manual_seed(23)
import pickle

DTYPE = torch.float32

scaler = GradScaler()

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def get_device(rank=None):
    """Get the appropriate device for the current process"""
    if rank is not None:
        return torch.device(f"cuda:{rank}")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='FNO') # FNO, ViT, UNO, CNO, Oformer, Transolver, DPOT, Crossformer, 
parser.add_argument('--dataset',type=str, default='ns2d_pda') # ['ns2d_fno_1e-3', 'ns2d_pda', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 'sw2d_pda'], note: pdb is the pde bench
parser.add_argument('--use_writer', action='store_true', default=False)
parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='/scratch3/wan410/operator_learning_model/')
# Distributed training arguments
parser.add_argument('--distributed', action='store_true', default=False, help='Use distributed training')
parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs to use')
parser.add_argument('--rank', type=int, default=0, help='Rank of current process')

args = parser.parse_args()


def error_metric(pred,true, Par):
    #re-normalize
    # true = true*Par['out_scale'] + Par['out_shift']
    # true = true*Par['out_scale'] + Par['out_shift']
    return torch.norm(true-pred, p=2)/torch.norm(true, p=2)

class MyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample, y_sample = self.transform(x_sample, y_sample)

        return x_sample, y_sample
    
def preprocess(x,y, Par):
    x = sliding_window_view(x[:,Par['lb']-1:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])
    y = sliding_window_view(y[:,Par['lb']-1:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])

    print('x: ', x.shape)
    print('y: ', y.shape)
    print()
    return x,y

def main_worker(rank, world_size, args):
    """Main training function for each GPU process"""
    
    # Initialize distributed training if enabled
    if args.distributed:
        setup_distributed(rank, world_size)
        device = get_device(rank)
        print(f"[GPU {rank}] Using device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
    
    # Only print on main process to avoid cluttered output
    is_main_process = not args.distributed or rank == 0
    
    res = 128
    begin_time = time.time()

    ntrain = 5200 if args.dataset == 'ns2d_pda' else 0
    comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
    log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)

    
    # inp = np.load(f"/oscar/data/gk/voommen/no_diffusion/kolmogrov/res_{res}/matcho/Y_PRED.npy") #low-fidelity
    # out = np.load(f"/oscar/data/gk/voommen/no_diffusion/kolmogrov/res_{res}/matcho/Y_TRUE.npy") #high-fidelity

    x_train = np.load(f"{log_path}/train_pred.npz")['pred'][...,0].transpose(0, 3, 1, 2) # (N, H, W, T) -> (N, T, H, W)
    y_train = np.load(f"{log_path}/train_pred.npz")['output'][...,0].transpose(0, 3, 1, 2) # (N, H, W, T) -> (N, T, H, W)

    x_val = np.load(f"{log_path}/val_pred.npz")['pred'][...,0].transpose(0, 3, 1, 2) # (N, H, W, T) -> (N, T, H, W)
    y_val = np.load(f"{log_path}/val_pred.npz")['output'][...,0].transpose(0, 3, 1, 2) # (N, H, W, T) -> (N, T, H, W)

    x_test = np.load(f"{log_path}/test_pred.npz")['pred'][...,0].transpose(0, 3, 1, 2) # (N, H, W, T) -> (N, T, H, W)
    y_test = np.load(f"{log_path}/test_pred.npz")['output'][...,0].transpose(0, 3, 1, 2) # (N, H, W, T) -> (N, T, H, W)

    if is_main_process:
        print("x_train shape", x_train.shape, "y_train shape", y_train.shape)
        print("x_val shape", x_val.shape, "y_val shape", y_val.shape)
        print("x_test shape", x_test.shape, "y_test shape", y_test.shape)
        print(f"Data Loading Time: {time.time() - begin_time:.1f}s")


    inp_min = np.min(x_train, axis=(0,2,3)).reshape(1,-1,1,1) # (1, T, 1, 1)
    inp_max = np.max(x_train, axis=(0,2,3)).reshape(1,-1,1,1)
    out_min = np.min(y_train, axis=(0,2,3)).reshape(1,-1,1,1)
    out_max = np.max(y_train, axis=(0,2,3)).reshape(1,-1,1,1)

    Par = {"inp_shift" : torch.tensor(inp_min, dtype=DTYPE, device=device),
           "inp_scale" : torch.tensor(inp_max - inp_min, dtype=DTYPE, device=device),
           "out_shift" : torch.tensor(out_min, dtype=DTYPE, device=device),
           "out_scale" : torch.tensor(out_max - out_min, dtype=DTYPE, device=device),
           "nx"        : x_train.shape[2],
           "ny"        : x_train.shape[3],
           "nf"        : 1,
           "lb"        : 1,
           "lf"        : 1,
           "num_epochs": 10000
           }

    # Normalizing the data to [0,1]
    shift = Par['inp_shift'].detach().cpu().numpy()
    scale = Par['inp_scale'].detach().cpu().numpy()
    x_train = (x_train - shift)/scale
    x_val = (x_val - shift)/scale
    x_test = (x_test - shift)/scale

    shift = Par['out_shift'].detach().cpu().numpy()
    scale = Par['out_scale'].detach().cpu().numpy()
    y_train = (y_train - shift)/scale
    y_val = (y_val - shift)/scale
    y_test = (y_test - shift)/scale

    Par["sigma_data"] = np.std(y_train)

    # Traj splitting
    begin_time = time.time()
    if is_main_process:
        print('\nTrain Dataset')
    x_train, y_train = preprocess(x_train, y_train, Par)
    if is_main_process:
        print('\nValidation Dataset')
    x_val, y_val = preprocess(x_val, y_val, Par)
    if is_main_process:
        print('\nTest Dataset')
    x_test, y_test = preprocess(x_test, y_test, Par)
    if is_main_process:
        print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

    Par.update({"channels"       : x_train.shape[1],
                "self_condition" : True
                })

    if is_main_process:
        print("Par")
        with open(log_path + '/Par.pkl', 'wb') as f:
            pickle.dump(Par, f)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    x_val_tensor   = torch.tensor(x_val,   dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)

    x_test_tensor  = torch.tensor(x_test,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

    train_dataset = MyDataset(x_train_tensor, y_train_tensor)
    val_dataset = MyDataset(x_val_tensor, y_val_tensor)
    test_dataset = MyDataset(x_test_tensor, y_test_tensor)

    # Define data loaders with distributed sampling
    train_batch_size = 100 #16
    val_batch_size   = 100
    test_batch_size  = 100
    
    if args.distributed:
        # Adjust batch size for distributed training
        train_batch_size = train_batch_size // world_size
        val_batch_size = val_batch_size // world_size
        test_batch_size = test_batch_size // world_size
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, sampler=val_sampler, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, sampler=test_sampler, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

    # Define Network Architecture
    net = Unet(
        dim = 16,
        dim_mults = (1, 2, 4, 8, 8),
        channels = Par["channels"],
        self_condition = Par["self_condition"],
        flash_attn = True
    ).to(device).to(torch.float32)

    model = ElucidatedDiffusion(net,
                                    channels = Par["channels"],
                                    image_size=Par["nx"],
                                    sigma_data=Par["sigma_data"])
    
    model = model.to(device)
    
    # Wrap model with DistributedDataParallel if using distributed training
    if args.distributed:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        if is_main_process:
            print(f"Model wrapped with DDP on {world_size} GPUs")

    # Adjust the dimensions as per your model's input size
    dummy_x = torch.tensor(torch.randn(1, Par["channels"], Par["nx"], Par["ny"]),   dtype=DTYPE, device=device)
    dummy_input = (dummy_x, dummy_x)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

    # Learning rate scheduler (Cosine Annealing)
    scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * len(train_loader) )  # Adjust T_max as needed

    # Training loop
    num_epochs = Par['num_epochs']
    best_val_loss = float('inf')
    best_model_id = 0

    if args.use_writer and is_main_process:
        writer = SummaryWriter(log_dir=log_path)

    t0 = time.time()
    for epoch in range(num_epochs):  # Remove tqdm to avoid conflicts with distributed training
        begin_time = time.time()
        model.train()
        
        # Set epoch for distributed sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss = 0.0
        train_time = time.time()
        
        # Use tqdm only on main process
        iterator = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}') if is_main_process else train_loader
        
        for l_fidel, h_fidel in iterator:
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                loss = model(h_fidel.to(device), l_fidel.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            # Update learning rate
            # scheduler.step()

        train_loss /= len(train_loader)
        
        # Synchronize training loss across all processes for distributed training
        if args.distributed:
            train_loss_tensor = torch.tensor(train_loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = train_loss_tensor.item() / world_size
        
        train_time = time.time()-train_time
        if args.use_writer and is_main_process:
            writer.add_scalar("train_loss", train_loss, epoch)

        # Validation
        if epoch !=0 and epoch % 10 == 0:
            val_time = time.time()
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for l_fidel, h_fidel in val_loader:
                    with autocast(device_type=device.type):
                        # Handle DDP model for sampling
                        if args.distributed:
                            pred = model.module.sample(l_fidel.to(device))
                        else:
                            pred = model.sample(l_fidel.to(device))
                        loss = error_metric(pred, h_fidel.to(device), Par)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            
            # Synchronize validation loss across all processes
            if args.distributed:
                val_loss_tensor = torch.tensor(val_loss, device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                val_loss = val_loss_tensor.item() / world_size

            # Save the model if validation loss is the lowest so far (only on main process)
            if val_loss < best_val_loss and is_main_process:
                best_val_loss = val_loss
                best_model_id = epoch+1
                # Save the actual model without DDP wrapper
                model_to_save = model.module if args.distributed else model
                torch.save(model_to_save.state_dict(), f'{log_path}/best_model.pt')

            val_time = time.time() - val_time
            if args.use_writer and is_main_process:
                writer.add_scalar("val_loss", val_loss, epoch)
                
            if is_main_process:
                time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
                elapsed_time = time.time() - begin_time
                print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, train time: {train_time:.2f}, val time: {val_time:.2f}')

        else:
            if is_main_process:
                time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
                print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, LR: {scheduler.get_last_lr()[0]:.4e}, train time: {train_time:.2f}')

    if is_main_process:
        print('Training finished.')
        print(f"Training Time: {time.time() - t0:.1f}s")

    # Testing loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for l_fidel, h_fidel in test_loader:
            with autocast(device_type=device.type):
                # Handle DDP model for sampling
                if args.distributed:
                    pred = model.module.sample(l_fidel.to(device))
                else:
                    pred = model.sample(l_fidel.to(device))
                loss = error_metric(pred, h_fidel.to(device), Par)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    
    # Synchronize test loss across all processes
    if args.distributed:
        test_loss_tensor = torch.tensor(test_loss, device=device)
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        test_loss = test_loss_tensor.item() / world_size
    
    if is_main_process:
        print(f'Test Loss: {test_loss:.4e}')
    
    # Clean up distributed training
    if args.distributed:
        cleanup_distributed()


def main():
    """Main function to handle distributed training launch"""
    if args.distributed:
        # Use torch.multiprocessing to spawn processes for each GPU
        mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
    else:
        # Single GPU training
        main_worker(0, 1, args)


if __name__ == '__main__':
    main()