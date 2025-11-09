
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import pickle
from tqdm import tqdm

################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter( self.scale * torch.rand(2, in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(2, in_channels, out_channels, self.modes1, self.modes2))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        out_real = torch.einsum("bixy,ioxy->boxy", input.real, weights[0]) - torch.einsum("bixy,ioxy->boxy", input.imag, weights[1])
        out_imag = torch.einsum("bixy,ioxy->boxy", input.real, weights[1]) + torch.einsum("bixy,ioxy->boxy", input.imag, weights[0])
        return torch.view_as_complex(torch.stack([out_real, out_imag],dim=-1))
        # return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, img_size = 64, n_channels=1,in_timesteps = 10, out_timesteps=1, n_layers=4, patch_size = 1, use_ln=False, multi_channel=True, normalize=False, n_cls=0, meanstd=False):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.img_size = img_size
        self.use_ln = use_ln
        self.padding = 2  # pad the domain if input is non-periodic
        # self.patch_size = patch_size

        self.normalize = normalize
        self.n_cls = n_cls
        self.meanstd = meanstd
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.fc0 = nn.Linear(in_timesteps*n_channels + 2, self.width)

        self.spectral_convs = nn.ModuleList([SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])
        self.convs = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        # self.ln_layers = nn.ModuleList([nn.LayerNorm([ self.in_shape[0], self.in_shape[1]]) for _ in range(self.n_layers)])
        if self.normalize:
            self.scale_feats = nn.Linear(2 * n_channels, width)
        if self.use_ln:
            self.ln_layers = nn.ModuleList([nn.GroupNorm(4, self.width) for _ in range(self.n_layers)])

        self.fc1 = nn.Linear(self.width, self.width)

        if self.meanstd:
            self.fc2 = nn.Linear(self.width, n_channels * out_timesteps * 2)
        else:
            self.fc2 = nn.Linear(self.width, n_channels * out_timesteps)

    def input_proj(self, x):
        T, C = x.shape[-2], x.shape[-1]
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
            x = (x - mu)/ sigma
            scale_feats = self.scale_feats(torch.cat([mu, sigma],dim=-1)).squeeze(-2).contiguous()   # B, 1, 1, C
        else:
            scale_feats = 0.0
        x = x.view(*x.shape[:-2], -1)           #### B, X, Y, T*C
        grid = self.get_grid(x)
        x = torch.cat((x, grid), dim=-1)        #### B, X, Y, T*C +2
        x = self.fc0(x) + scale_feats
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.normalize:
            return x, (mu, sigma)
        else:
            return x, (None, None)

    def forward(self, x):
        """
        x: (B, H, W, T, C)
        """
        x, (mu, sigma) = self.input_proj(x)

        # print(x.shape, scale_feats.shape, self.normalize)
        # x = self.patch_embed(x) + scale_feats
        # x = x + scale_feats

        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.convs[i](x)
            x = x1 + x2
            x = F.gelu(x)
            if self.use_ln:
                x = self.ln_layers[i](x)


        # classification
        # cls_token = x.mean(dim=(2, 3), keepdim=False)
        # cls_pred = self.cls_head(cls_token)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x) # mlp

        x = F.gelu(x)
        x = self.fc2(x)

        x = x.reshape(*x.shape[:3], self.out_timesteps, -1)

        if self.normalize:
            x = x * sigma  + mu

        # return x, cls_pred
        return x


    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid


    def get_latent_by_index(self, x, start_block_index=0):
        x, (mu, sigma) = self.input_proj(x)  # encoder
        if start_block_index == 0:
            # (B, C, H, W) -> (B, H, W, 1, C)
            x = x.permute(0, 2, 3, 1).unsqueeze(-2)
            return x
        for i in range(start_block_index):
            x = self.spectral_convs[i](x)
            x = self.convs[i](x)
            x = F.gelu(x)
            if self.use_ln:
                x = self.ln_layers[i](x) # (B, C, H, W)
        # (B, C, H, W) -> (B, H, W, 1, C)
        x = x.permute(0, 2, 3, 1).unsqueeze(-2)
        return x
    
    def get_testing_block_by_index(self, index, x):
        # the input comes in shape (B, H, W, 1, C), change it to (B, C, H, W)
        x = x.squeeze(-2).permute(0, 3, 1, 2).contiguous()
        x = self.spectral_convs[index](x)
        x = self.convs[index](x)
        x = F.gelu(x)
        if self.use_ln:
            x = self.ln_layers[index](x)
        # (B, C, H, W) -> (B, H, W, 1, C)
        x = x.permute(0, 2, 3, 1).unsqueeze(-2)
        return x

class FNO2d_Tin1_Tout1(nn.Module): # only use one timestep input and one timestep output
    def __init__(self, modes1, modes2, width, img_size = 64, in_channels=1,out_channels=1,in_timesteps = 1, out_timesteps=1, n_layers=4, patch_size = 1, use_ln=True, multi_channel=True, normalize=False, n_cls=0, meanstd=False):
        super(FNO2d_Tin1_Tout1, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.img_size = img_size
        self.use_ln = use_ln
        self.padding = 2  # pad the domain if input is non-periodic
        # self.patch_size = patch_size

        self.normalize = normalize
        self.n_cls = n_cls
        self.meanstd = meanstd
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.fc0 = nn.Linear(in_timesteps*in_channels, self.width)

        self.spectral_convs = nn.ModuleList([SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])
        self.convs = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        # self.ln_layers = nn.ModuleList([nn.LayerNorm([ self.in_shape[0], self.in_shape[1]]) for _ in range(self.n_layers)])
        if self.normalize:
            self.scale_feats = nn.Linear(2 * in_channels, width)
        if self.use_ln:
            self.ln_layers = nn.ModuleList([nn.GroupNorm(4, self.width) for _ in range(self.n_layers)])

        self.fc1 = nn.Linear(self.width, self.width)

        if self.meanstd:
            self.fc2 = nn.Linear(self.width, out_channels * out_timesteps * 2)
        else:
            self.fc2 = nn.Linear(self.width, out_channels * out_timesteps)

    def input_proj(self, x):
        if len(x.shape) == 5: #(B, H, W, 1, C) -> (B, H, W, C)
            x = x.squeeze(-2) # remove the time dimension
        else:# (B, C, H, W)
            x = x.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)
        x = self.fc0(x) 
        x = x.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        return x

    def forward(self, x):
        """
        x: (B, H, W, 1, C)
        
        """
        origial_shape = x.shape
        x = self.input_proj(x)

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.convs[i](x)
            x = x1 + x2
            if self.use_ln:
                x = self.ln_layers[i](x)
            x = F.gelu(x)

        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.fc1(x) # mlp

        x = F.gelu(x)
        x = self.fc2(x)    # (B, H, W, C_out)
        if len(origial_shape) == 5:
            x = x.unsqueeze(-2) # (B, H, W, 1, C_out)
        else:
            x = x.permute(0, 3, 1, 2) # (B, C_out, H, W)
        return x

if __name__ == "__main__":
    # x = torch.rand(1, 128, 128, 10, 1)
    # model = FNO2d(12, 12, 32, 128)

    x = torch.rand(2, 96, 192, 10, 1)
    model = FNO2d(12, 12, 32, img_size=(96, 192), normalize=True)
    y = model(x)
    print(y.shape)
