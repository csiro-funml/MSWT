"""
Periodic variant of MSWT for 2D data.

This mirrors the MSWT structure but uses circular padding and periodic
wavelet transforms to respect periodic boundary conditions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
import sys
sys.path.append(os.path.dirname(__file__))
from wavelet_transform import Attention, FeedForward
# from periodic_ops import (
#     AddPeriodicGrid,
#     CircularConv2d,
#     PeriodicDWT2D,
#     PeriodicIDWT2D,
# )

from periodic_ops_strictly_periodic import (
    AddPeriodicGrid,
    CircularConv2d,
    PeriodicDWT2D,
    PeriodicIDWT2D,
)


class PeriodicWaveletAttentionBlock(nn.Module):
    def __init__(
        self,
        wave="haar",
        dim=64,
        use_efficient_attention=False,
        local_attention_size=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dwt = PeriodicDWT2D(wave)
        self.idwt = PeriodicIDWT2D(wave)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
        )
        self.conv_post = nn.Sequential(
            CircularConv2d(dim, dim, kernel_size=3, padding=1, stride=1),
        )
        self.attention = Attention(dim)
        self.final_proj = nn.Linear(dim // 4, dim)
        self.use_efficient_attention = use_efficient_attention
        self.local_attention_size = local_attention_size

    def local_attention(self, x, h, w):
        b, c, _, _ = x.shape
        patch = self.local_attention_size
        pad_h = (patch - h % patch) % patch
        pad_w = (patch - w % patch) % patch
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")

        new_h, new_w = (h + pad_h) // patch, (w + pad_w) // patch
        x = rearrange(
            x,
            "b c (h_patch new_h) (w_patch new_w) -> (b new_h new_w) (h_patch w_patch) c",
            h_patch=patch,
            w_patch=patch,
            new_h=new_h,
            new_w=new_w,
        )
        x = self.attention(x)
        x = rearrange(
            x,
            "(b new_h new_w) (h_patch w_patch) c -> b c (h_patch new_h) (w_patch new_w)",
            b=b,
            new_h=new_h,
            new_w=new_w,
            h_patch=patch,
            w_patch=patch,
        )
        if pad_h or pad_w:
            x = x[:, :, :h, :w]
        return x

    def global_attention(self, x, h, w):
        b, c, _, _ = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.attention(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x

    def forward(self, x, h, w):
        b, c = x.shape[0], x.shape[-1]
        x = self.mlp_head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.dwt(x)
        new_h, new_w = x.shape[-2], x.shape[-1]
        x = self.conv_post(x)
        if self.use_efficient_attention:
            x = self.local_attention(x, new_h, new_w)
        else:
            x = self.global_attention(x, new_h, new_w)
        x = torch.reshape(x, (b, 4, c // 4, new_h, new_w))
        x = self.idwt(x, target_size=(h, w))
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.final_proj(x)
        return x


class PeriodicMultiscaleWaveletTransformer2D(nn.Module):
    def __init__(
        self,
        wave="haar",
        input_dim=3,
        output_dim=3,
        dim=None,
        dims=None,
        use_efficient_attention=False,
        efficient_layers=None,
        add_grid=False,
        add_periodic_grid=False,
        patch_size=None,
        local_attention_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_grid = add_grid
        self.add_periodic_grid = add_periodic_grid
        self.patch_size = patch_size
        self.local_attention_size = local_attention_size
        self.use_efficient_attention = use_efficient_attention
        self.add_periodic = AddPeriodicGrid() if add_periodic_grid else None

        if dims is None:
            dims = []
        if efficient_layers is None:
            efficient_layers = [0, 1]

        if len(dims) == 0 and dim is not None:
            dims = np.array([dim // 2, 2 * dim, 4 * dim, 4 * dim])
        else:
            dims = np.array(dims)
        self.n_layers = len(dims)

        proj_in_dim = input_dim
        if add_grid:
            proj_in_dim += 2
        if add_periodic_grid:
            proj_in_dim += 4

        if patch_size is None:
            self.input_proj = nn.Linear(proj_in_dim, dims[0])
            self.output_proj = nn.Sequential(
                nn.Linear(dims[0], dims[0] // 2),
                nn.GELU(),
                nn.Linear(dims[0] // 2, output_dim),
            )

        self.enc_layers = nn.ModuleList([])
        for i in range(self.n_layers):
            dim = dims[i]
            efficient_flag = i in efficient_layers and self.use_efficient_attention
            attn_layer = nn.ModuleList(
                [
                    nn.LayerNorm(dim),
                    PeriodicWaveletAttentionBlock(
                        wave=wave,
                        dim=dim,
                        use_efficient_attention=efficient_flag,
                        local_attention_size=local_attention_size,
                    ),
                    nn.LayerNorm(dim),
                    FeedForward(dim, dim * 4),
                ]
            )

            down_layer = nn.ModuleList(
                [
                    nn.Linear(dim, dim // 4),
                    nn.LayerNorm(dim // 4),
                    PeriodicDWT2D(wave),
                    CircularConv2d(
                        dim,
                        dims[i + 1] if i < self.n_layers - 1 else dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    ),
                ]
            )

            self.enc_layers.append(nn.ModuleList([attn_layer, down_layer]))

        self.dec_layers = nn.ModuleList([])
        for i in range(self.n_layers):
            dim = dims[self.n_layers - i - 1]
            new_dim = dims[self.n_layers - i - 2] if self.n_layers - i - 2 >= 0 else dim
            efficient_flag = self.n_layers - i - 1 in efficient_layers and self.use_efficient_attention
            up_layer = nn.ModuleList(
                [
                    nn.Linear(dim, dim * 4),
                    nn.LayerNorm(dim * 4),
                    PeriodicIDWT2D(wave),
                    CircularConv2d(
                        2 * dim,
                        new_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    ),
                ]
            )

            attn_layer = nn.ModuleList(
                [
                    nn.LayerNorm(new_dim),
                    PeriodicWaveletAttentionBlock(
                        wave=wave,
                        dim=new_dim,
                        use_efficient_attention=efficient_flag,
                        local_attention_size=local_attention_size,
                    ),
                    nn.LayerNorm(new_dim),
                    FeedForward(new_dim, new_dim * 4),
                ]
            )

            self.dec_layers.append(nn.ModuleList([up_layer, attn_layer]))

    def get_grid(self, x):
        b, h, w, _ = x.shape
        gridx = torch.linspace(0, 1, h, dtype=x.dtype, device=x.device)
        gridy = torch.linspace(0, 1, w, dtype=x.dtype, device=x.device)
        xx, yy = torch.meshgrid(gridx, gridy, indexing="ij")
        return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

    def attention_block(self, x, layer, h, w):
        ln1, wavelet_block, ln2, ff = layer
        x = wavelet_block(ln1(x), h, w) + x
        x = ln2(ff(x)) + x
        return x

    def down_block(self, x, layer, h, w):
        linear, ln, dwt, conv = layer
        x = ln(linear(x))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = dwt(x)
        x = conv(x)
        h, w = x.shape[-2], x.shape[-1]
        x = rearrange(x, "b c h w -> b (h w) c")
        return x, h, w

    def up_block(self, x, x_prev, layer, h, w):
        linear, ln, idwt, conv = layer
        x = ln(linear(x))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = idwt(x)
        x = torch.cat((x, x_prev), dim=1)
        x = conv(x)
        h, w = x.shape[-2], x.shape[-1]
        x = rearrange(x, "b c h w -> b (h w) c")
        return x, h, w

    def forward(self, x):
        if self.add_grid:
            x = torch.cat((x, self.get_grid(x)), dim=-1)
        if self.add_periodic_grid:
            x = self.add_periodic(x)

        x = self.input_proj(x)
        h, w = x.shape[1], x.shape[2]
        x = rearrange(x, "b h w c -> b (h w) c")

        x_list = []
        for attn_layer, down_layer in self.enc_layers:
            x_list.append(rearrange(x, "b (h w) c -> b c h w", h=h, w=w))
            x = self.attention_block(x, attn_layer, h, w)
            x, h, w = self.down_block(x, down_layer, h, w)

        for up_layer, attn_layer in self.dec_layers:
            x, h, w = self.up_block(x, x_list.pop(), up_layer, h, w)
            x = self.attention_block(x, attn_layer, h, w)
        
        x = self.output_proj(x)
        x = rearrange(x, "b (h w) c -> b h w c", h=h, w=w)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class PeriodicMSWT2D_Patching(nn.Module):
    def __init__(
        self,
        wave="haar",
        input_dim=3,
        output_dim=3,
        dim=None,
        dims=None,
        use_efficient_attention=False,
        efficient_layers=None,
        add_grid=False,
        add_periodic_grid=False,
        patch_size=None,
        local_attention_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_grid = add_grid
        self.add_periodic_grid = add_periodic_grid
        self.patch_size = patch_size
        self.local_attention_size = local_attention_size
        self.use_efficient_attention = use_efficient_attention
        self.add_periodic = AddPeriodicGrid() if add_periodic_grid else None

        if dims is None:
            dims = []
        if efficient_layers is None:
            efficient_layers = [0, 1]

        if len(dims) == 0 and dim is not None:
            dims = np.array([dim // 2, 2 * dim, 4 * dim, 4 * dim])
        else:
            dims = np.array(dims)
        self.n_layers = len(dims)

        proj_in_dim = input_dim
        if add_grid:
            proj_in_dim += 2
        if add_periodic_grid:
            proj_in_dim += 4

        if patch_size is None:
            self.input_proj = nn.Linear(proj_in_dim, dims[0])
            self.output_proj = nn.Sequential(
                nn.Linear(dims[0], dims[0] // 2),
                nn.GELU(),
                nn.Linear(dims[0] // 2, output_dim),
            )
        else:
            self.patch_size = patch_size
            self.input_proj_layer =  nn.Linear(proj_in_dim * patch_size * patch_size, dims[0])
            self.output_proj_layer = nn.Sequential(
                nn.Linear(dims[0], dims[0] // 2),
                nn.GELU(),
                nn.Linear(dims[0] // 2, output_dim * patch_size * patch_size),
            )
        self.enc_layers = nn.ModuleList([])
        for i in range(self.n_layers):
            dim = dims[i]
            efficient_flag = i in efficient_layers and self.use_efficient_attention
            attn_layer = nn.ModuleList(
                [
                    nn.LayerNorm(dim),
                    PeriodicWaveletAttentionBlock(
                        wave=wave,
                        dim=dim,
                        use_efficient_attention=efficient_flag,
                        local_attention_size=local_attention_size,
                    ),
                    nn.LayerNorm(dim),
                    FeedForward(dim, dim * 4),
                ]
            )

            down_layer = nn.ModuleList(
                [
                    nn.Linear(dim, dim // 4),
                    nn.LayerNorm(dim // 4),
                    PeriodicDWT2D(wave),
                    CircularConv2d(
                        dim,
                        dims[i + 1] if i < self.n_layers - 1 else dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    ),
                ]
            )

            self.enc_layers.append(nn.ModuleList([attn_layer, down_layer]))

        self.dec_layers = nn.ModuleList([])
        for i in range(self.n_layers):
            dim = dims[self.n_layers - i - 1]
            new_dim = dims[self.n_layers - i - 2] if self.n_layers - i - 2 >= 0 else dim
            efficient_flag = self.n_layers - i - 1 in efficient_layers and self.use_efficient_attention
            up_layer = nn.ModuleList(
                [
                    nn.Linear(dim, dim * 4),
                    nn.LayerNorm(dim * 4),
                    PeriodicIDWT2D(wave),
                    CircularConv2d(
                        2 * dim,
                        new_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    ),
                ]
            )

            attn_layer = nn.ModuleList(
                [
                    nn.LayerNorm(new_dim),
                    PeriodicWaveletAttentionBlock(
                        wave=wave,
                        dim=new_dim,
                        use_efficient_attention=efficient_flag,
                        local_attention_size=local_attention_size,
                    ),
                    nn.LayerNorm(new_dim),
                    FeedForward(new_dim, new_dim * 4),
                ]
            )

            self.dec_layers.append(nn.ModuleList([up_layer, attn_layer]))

    def get_grid(self, x):
        b, h, w, _ = x.shape
        gridx = torch.linspace(0, 1, h, dtype=x.dtype, device=x.device)
        gridy = torch.linspace(0, 1, w, dtype=x.dtype, device=x.device)
        xx, yy = torch.meshgrid(gridx, gridy, indexing="ij")
        return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

    def attention_block(self, x, layer, h, w):
        ln1, wavelet_block, ln2, ff = layer
        x = wavelet_block(ln1(x), h, w) + x
        x = ln2(ff(x)) + x
        return x

    def down_block(self, x, layer, h, w):
        linear, ln, dwt, conv = layer
        x = ln(linear(x))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = dwt(x)
        x = conv(x)
        h, w = x.shape[-2], x.shape[-1]
        x = rearrange(x, "b c h w -> b (h w) c")
        return x, h, w

    def up_block(self, x, x_prev, layer, h, w):
        linear, ln, idwt, conv = layer
        x = ln(linear(x))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = idwt(x)
        x = torch.cat((x, x_prev), dim=1)
        x = conv(x)
        h, w = x.shape[-2], x.shape[-1]
        x = rearrange(x, "b c h w -> b (h w) c")
        return x, h, w
    
    def input_proj_block(self, x):
        x = rearrange(x, "b (h p1) (w p2) c -> b h w (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        x = self.input_proj_layer(x)
        return x

    def output_proj_block(self, x, h, w):
        x = self.output_proj_layer(x)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b (h p1) (w p2) c", p1=self.patch_size, p2=self.patch_size, h=h, w=w)
        return x

    def forward(self, x):
        if self.add_grid:
            x = torch.cat((x, self.get_grid(x)), dim=-1)
        if self.add_periodic_grid:
            x = self.add_periodic(x)
        
        if self.patch_size is not None: # patchify the image
            x = self.input_proj_block(x)
        else:
            x = self.input_proj(x)
        h, w = x.shape[1], x.shape[2]
        x = rearrange(x, "b h w c -> b (h w) c")

        x_list = []
        for attn_layer, down_layer in self.enc_layers:
            x_list.append(rearrange(x, "b (h w) c -> b c h w", h=h, w=w))
            x = self.attention_block(x, attn_layer, h, w)
            x, h, w = self.down_block(x, down_layer, h, w)

        for up_layer, attn_layer in self.dec_layers:
            x, h, w = self.up_block(x, x_list.pop(), up_layer, h, w)
            x = self.attention_block(x, attn_layer, h, w)
        
        if self.patch_size is not None:
            x = self.output_proj_block(x, h=h, w=w)
        else:
            x = self.output_proj(x)
            x = rearrange(x, "b (h w) c -> b h w c", h=h, w=w)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)






if __name__ == "__main__":
    # x = torch.randn(2, 64, 64, 3)
    x = torch.rand(2, 96, 192, 3)
    # model = MultiscaleWaveletTransformer2D(input_dim=3, output_dim=1, dim=64, use_efficient_attention=True)
    # model = MultiscaleWaveletTransformer2D(input_dim=3, output_dim=1, dims=[64, 128, 256, 512], use_efficient_attention=True,   efficient_layers=[0, 1, 2])
    # model = PeriodicMultiscaleWaveletTransformer2D(input_dim=3, output_dim=1, dims=[32, 64, 128, 256, 512], 
    # use_efficient_attention=True,   efficient_layers=[0, 1, 2, 3], add_periodic_grid=True, local_attention_size=6)
    model = PeriodicMSWT2D_Patching(input_dim=3, output_dim=1, dims=[64, 64, 128, 512], 
    use_efficient_attention=True,   efficient_layers=[0, 1, 2], add_periodic_grid=True, 
    local_attention_size=8, 
    patch_size=3)
    # model = MultiscaleWaveletTransformer2DDecoderNoAttention(input_dim=3, output_dim=1, dim=96, use_efficient_attention=True)
    
    print("number of parameters:", model.count_parameters())
    with torch.autograd.set_detect_anomaly(True):
        output = model(x)
        output = output[0] if isinstance(output, tuple) else output
        print("output shape:", output.shape)
        loss = output.mean()
        loss.backward()
        print("backward done")