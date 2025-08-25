"""
Train a ResNet to predict the next step of the PDE
This code is developed with reference to the following GitHub repo: HFSNO
https://github.com/SiaK4/HFS_ResUNet/
"""



import os
import json
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import time

from models.high_frequency_scaling import ResUNet

from lion_pytorch import Lion
import json
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.griddataset import TemporalDataset2D, LocalTemporalDataset2D
from utils.utilities import count_parameters
from tqdm import tqdm

torch.manual_seed(1234)
np.random.seed(1234)

torch.cuda.empty_cache()
batch_size = 8


################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining on multiple PDE datasets')

parser.add_argument('--model', type=str, default='HFS') # HFS: high frequency scaling
parser.add_argument('--dataset',type=str, default='ns2d_pda') # ['ns2d_fno_1e-3', 'ns2d_pda', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 'sw2d_pda'], note: pdb is the pde bench
parser.add_argument('--resume_path',type=str, default='')
parser.add_argument('--use_writer', action='store_true',default=False)


# ### dataset details
parser.add_argument('--T_in', type=int, default=7)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--pad', type=int, default=0)
parser.add_argument('--normalize',type=int, default=1)


###### optimizer and training setups
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=2000)
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

comment = args.comment + '{}_{}_ntrain{}'.format(args.model, args.dataset, ntrain)
log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
# model_path = log_path + '/model.pth'
model_path = log_path + f'/model.pth' # I will test a longer training epoch
print(model_path)

def train_step(model,x,y):

    y_pred = model(x)
    loss = torch.mean((y_pred-y)**2)
    return loss

def warmup_lr(optimizer, scheduler1, scheduler2, current_step, warmup_steps, initial_lr,target_lr):
    if current_step <= warmup_steps:
        lr = initial_lr + (target_lr - initial_lr)*(current_step/warmup_steps)
        optimizer.param_groups[0]['lr'] = lr
        scheduler1.base_lrs = [group['lr'] for group in optimizer.param_groups]
        scheduler2.base_lrs = [group['lr'] for group in optimizer.param_groups]

def train(model, epoch_number, learning_rate, target_lr,model_path, display_every=10, checkpoint_interval=50,warmup_steps=10):
    best_loss = float('inf')
    best_loss_epoch = 0
    train_loss = []
    val_loss = []
    model.to(device)
    optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay = 0.01)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=5e-06)
    epoch_time = 0

    epoc_bar = range(epoch_number)
    for epoch in tqdm(epoc_bar): # the last item of train_loss to display
        ## Optionally call warmup learning rate function
        # warmup_lr(optimizer,scheduler1,scheduler2,epoch,warmup_steps,initial_lr=learning_rate,target_lr=target_lr)
        epoch_start = time.time()
        batch_loss = []
        for i, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            loss = train_step(model, x_train, y_train)
            optimizer.zero_grad()
            loss.backward()
            #Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            optimizer.step()
            
            batch_loss.append(loss.item())
        time_for_epoch = time.time() - epoch_start
        epoch_time+=time_for_epoch
        train_loss.append(np.mean(batch_loss))
        print('epoch {}, best epoch: {}, train loss {:.5f}'
            .format(epoch, best_loss_epoch, train_loss[-1]))
        if epoch % 10 == 0:
            with torch.no_grad():
                batch_val_loss = []
                for j, (x_val, y_val) in enumerate(val_loader):
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    loss_val = train_step(model, x_val, y_val)
                    batch_val_loss.append(loss_val.item())
                val_loss.append(np.mean(batch_val_loss))

            if val_loss[-1] < best_loss:
                best_loss = val_loss[-1]
                best_loss_epoch = epoch
                print(f"New best val loss: {best_loss} at epoch {epoch+1}")
                torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                            'scheduler1': scheduler1.state_dict(),
                            'scheduler2': scheduler2.state_dict(),
                            }, model_path,
                            )
            # best_loss = save_checkpoint(model, epoch, loss_val, best_loss, checkpoint_name=checkpoint_name)
        
        lr_old = optimizer.param_groups[0]['lr']
        if (epoch+1)>=120 and (epoch+1)<=200:
            if (epoch+1) ==120:
                scheduler1.base_lrs = [group['lr'] for group in optimizer.param_groups]
            scheduler1.step()
        elif (epoch+1)>250:
            if (epoch+1)== 251:
                scheduler2.base_lrs = [group['lr'] for group in optimizer.param_groups]
            scheduler2.step()
        lr_new = optimizer.param_groups[0]['lr']
        if lr_old != lr_new:
            print(f"Learning rate has changed {lr_old:.8f}--->{lr_new:.8f} at epoch {epoch+1}")
        
        if ((epoch+1)%display_every==0) or epoch==0:
            print(f"Training loss is {(np.mean(batch_loss)).item()} at epoch {epoch+1} <><><><><> Validation loss is {(np.mean(batch_val_loss)).item()}")
            print("epoch time:", time_for_epoch)

        ### Save the losses every 20 epochs in case training is left incomplete
        if (epoch+1)%20 == 0:
            with open('train_loss.json','w') as file:
                json.dump(train_loss, file)
            with open('val_loss.json','w') as file2:
                json.dump(val_loss, file2)
    
    ### Print out the average epoch time
    print("Average per epoch time:",epoch_time/epoch_number)
    
    #Save last epoch
    best_loss = float('inf')
    # save_checkpoint(model, epoch, loss_val, best_loss=best_loss, checkpoint_name=checkpoint_name)
    
model = ResUNet(in_c = 3 * args.T_in + 2 ,out_c = 3, features = [32,64,64,128,128], bottleneck_feature=256, device=device)

# todo: change this
checkpoint_name = args.log_path + 'kolmo_HFS'

print(model)
count_parameters(model)

train(model, epoch_number=2000, learning_rate=8e-4,target_lr=8e-4,model_path=model_path,display_every=10,checkpoint_interval=5,warmup_steps=10)
