"""
Fourierformer model
two versions: w/wo temporal attention
Attention is implemented in the Fourier domain
Goal: model dependencies between multi-scale features while preserving high frequency details
Enc -> FFormerBlock -> Dec
FFormerBlocl(x):    IFT(Transformer(FFT(x))) 
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
# Add parent directory to Python path to access utils and models
sys.path.append(os.path.join(os.path.dirname(__file__)))
from wavelet_transform import Transformer, RelativePositionBias
from einops import rearrange

class FourierTransformer(Transformer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super(FourierTransformer, self).__init__(dim, depth, heads, dim_head, mlp_dim)
        self.pos_emb = RelativePositionBias(dim=dim)

    def norm_with_reshape(self, x):
        x = rearrange(x, 'b d h w -> b h w d')
        x = self.norm(x)
        x = rearrange(x, 'b h w d -> b d h w')
        return x
    
    def get_pos_emb(self, x):
        pos_emb = self.pos_emb(x.shape[-2], x.shape[-1]//2 +1, x.device)
        pos_emb = torch.cat((pos_emb, pos_emb), dim=0) # include the real and imaginary parts with the same position embedding
        return pos_emb.unsqueeze(0)
    
    def forward(self, x):
        x_raw = x
        # maybe add position embedding here
        pos_emb = self.get_pos_emb(x)

        # only learn attention in the attention block but not in the skip connection
        for layer_idx, (attn, ff) in enumerate(self.layers):
            # fourier transform
            x = torch.fft.rfft2(x) # (B, D, H, W//2+1)
            x = torch.cat((x.real, x.imag), dim=-1) # include the real and imaginary parts
            h, w= x.shape[-2], x.shape[-1]
            x = rearrange(x, 'b d h w -> b (h w) d')
            # attention block in the fourier domain
            if layer_idx == 0:
                x = x + pos_emb
            x = attn(x) + x
            x = ff(x) + x
            x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
            x = torch.complex(x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]).to(x_raw.device)
            # inverse fourier transform
            x = torch.fft.irfft2(x) # (B, D, H, W)
            # only modelling the residual, i.e., the velocity
            x = x + x_raw
        x = self.norm_with_reshape(x)
        return x

class FFormer(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels,
            in_timesteps=10, 
            out_timesteps=1,
            n_layers=4, 
            dim=1024,
            patch_size=(4, 4),
        ):
        super(FFormer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.n_layers = n_layers
        self.dim = dim
        self.patch_size = patch_size

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels*in_timesteps+2, dim, kernel_size=patch_size, stride=patch_size, padding=0),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.transformer = FourierTransformer(dim=dim, depth=n_layers, heads=8, dim_head=64, mlp_dim=dim*4)

        self.output_layer = nn.ConvTranspose2d(dim, out_channels, kernel_size=patch_size, stride=patch_size, padding=0)
    
    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid
   
    def input_proj(self, x):
        B, H, W, T, C = x.shape
        x = x.view(B, H, W, T*C)
        grid = self.get_grid(x)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous() 
        x = self.input_layer(x)
        return x


    def output_proj(self, x):
        x = self.output_layer(x)
        x = rearrange(x, 'b d h w -> b h w 1 d')
        return x
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    x = torch.randn(2, 128, 128, 7, 3)
    model = FFormer(in_channels=3, out_channels=3, in_timesteps=7, out_timesteps=1, n_layers=3, dim=1024, patch_size=(4, 4))
    print(model.count_parameters())
    y = model(x)
    print(y.shape)
    