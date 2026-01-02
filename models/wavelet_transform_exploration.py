import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange
import os
import sys
sys.path.append(os.path.dirname(__file__))
from wavelet_transform import WaveletAttentionBlock, FeedForward, DWT_2D, IDWT_2D


class MultiscaleWaveletTransformer2DDecoderNoAttention(nn.Module):
    def __init__(self, wave='haar', input_dim=3, output_dim=3, dim=None, dims=[], use_efficient_attention=False,
                       efficient_layers=[0, 1], add_grid=False, patch_size=None, **kwargs):
        super().__init__(**kwargs)
        self.add_grid = add_grid
        self.patch_size = patch_size

        if len(dims) == 0 and dim is not None:
            dims = np.array([dim//4, dim, 4*dim, 4*dim])
        else:
            dims = np.array(dims)
        self.n_layers = len(dims)
        if patch_size is None:
            self.input_proj = nn.Linear(input_dim, dims[0])
            self.output_proj = nn.Sequential(nn.Linear(dims[0], dims[0]//2),
                                            nn.GELU(),
                                            nn.Linear(dims[0]//2, output_dim))

        self.enc_layers = nn.ModuleList([])
        self.use_efficient_attention = use_efficient_attention
        for i in range(self.n_layers):
            dim = dims[i]
            efficient_flag = i in efficient_layers and self.use_efficient_attention
            attn_layer = nn.ModuleList([
                nn.LayerNorm(dim),
                WaveletAttentionBlock(wave=wave, dim=dim, use_efficient_attention=efficient_flag),
                nn.LayerNorm(dim),
                FeedForward(dim, dim*4)
                ])

            down_layer = nn.ModuleList([
                nn.Linear(dim, dim//4),
                nn.LayerNorm(dim//4),
                DWT_2D(wave), 
                nn.Conv2d(dim, dims[i+1] if i < self.n_layers - 1 else dim, kernel_size=3, padding=1, stride=1, groups=4),
            ])
                
            self.enc_layers.append(nn.ModuleList([attn_layer, down_layer]))
        # self.norm = nn.LayerNorm(dim)
        self.dec_layers = nn.ModuleList([])
        for i in range(self.n_layers):
            dim = dims[self.n_layers - i - 1]
            new_dim = dims[self.n_layers - i - 2] if self.n_layers - i - 2 >= 0 else dim
            efficient_flag = self.n_layers - i - 1 in efficient_layers and self.use_efficient_attention
            up_layer = nn.ModuleList([
                nn.Linear(dim, dim*4),
                nn.LayerNorm(dim*4),
                IDWT_2D(wave),
                nn.Conv2d(2*dim, new_dim, kernel_size=3, padding=1, stride=1, groups=4),
                ])
            
            self.dec_layers.append(up_layer)

    def get_grid(self, x):
        b, h, w, _ = x.shape
        size_x, size_y = h, w
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).to(x.device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([b, 1, size_y, 1]) 
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float).to(x.device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([b, size_x, 1, 1]) 
        x_grid = torch.cat((gridx, gridy), dim=-1)

        return x_grid

    def attention_block(self, x, layer, h, w):
        ln1, wavelet_block, ln2, ff = layer
        x = wavelet_block(ln1(x), h, w) + x
        x = ln2(ff(x)) + x
        return x
    
    def down_block(self, x, layer, h, w):
        linear, ln, dwt, conv = layer
        x = ln(linear(x))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = dwt(x) # (B, 4xc//4, H/2, W/2)
        x = conv(x)
        h, w= x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, h, w
    

    def up_block(self, x, x_prev, layer, h, w):
        linear, ln, idwt, conv = layer
        x = ln(linear(x)) # (B, (H/2 x W/2), c*4)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = idwt(x) # (B, C, H, W)
        x = torch.cat((x, x_prev), dim=1) # (B, 2C, H, W)
        x = conv(x)
        h, w= x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, h, w

    def forward(self, x):
        if self.add_grid:
            x_grid = self.get_grid(x)
            x = torch.cat((x, x_grid), dim=-1)

        x = self.input_proj(x)
        h, w = x.shape[1], x.shape[2]
        x = rearrange(x, 'b h w c -> b (h w) c')

        x_list = []

        for attn_layer, down_layer in self.enc_layers:
            x_list.append(rearrange(x, 'b (h w) c -> b c h w', h=h, w=w))
            x = self.attention_block(x, attn_layer, h, w)
            x, h, w = self.down_block(x, down_layer, h, w) # h and w would be updated here
            
        
        for up_layer in self.dec_layers:
            x, h, w = self.up_block(x,  x_list.pop(), up_layer, h, w)
        
        # x = self.norm(x)
        x = self.output_proj(x)
        x = rearrange(x, 'b (h w) c-> b h w c', h=h, w=w)
        return x


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



if __name__ == "__main__":
    # x = torch.randn(2, 64, 64, 3)
    x = torch.rand(2, 96, 192, 3)
    # model = MultiscaleWaveletTransformer2D(input_dim=3, output_dim=3, dim=64, use_efficient_attention=True)
    
    model = MultiscaleWaveletTransformer2DDecoderNoAttention(input_dim=3, output_dim=3, dim=96, use_efficient_attention=True)
    
    print("number of parameters:", model.count_parameters())
    with torch.autograd.set_detect_anomaly(True):
        output = model(x)
        output = output[0] if isinstance(output, tuple) else output
        print("output shape:", output.shape)
        loss = output.mean()
        loss.backward()
        print("backward done")
