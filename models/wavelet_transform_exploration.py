import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import os
import sys
sys.path.append(os.path.dirname(__file__))
from wavelet_transform import Attention, WaveletAttentionBlock, FeedForward, DWT_2D, IDWT_2D


class LiteDecoderAttentionBlock(nn.Module):
    """
    Lightweight decoder attention block used to add depth without a large
    parameter increase. It reuses the efficient Attention defined in
    wavelet_transform and a narrower feedforward.
    """
    def __init__(self, dim, heads=4, dim_head=32, ff_mult=2.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim, int(dim * ff_mult))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class Adapter(nn.Module):
    """Low-rank adapter to add capacity with minimal params."""
    def __init__(self, dim, bottleneck=16, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.up(self.act(self.down(self.norm(x)))))


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


class MultiscaleWaveletTransformer2DDecoderPE(nn.Module):
    """
    Parameter-efficient variant that keeps total params ~15M while increasing
    effective depth via weight tying and lightweight decoder attention.

    Default hyper-params land around ~14M params (dim=128, ff_mult=3.0,
    stage_depths=2 for each scale).

    Extras for memory/perf:
    - Activation checkpointing on attention/adapter blocks (use_checkpoint).
    - Low-rank adapters to add capacity with minimal params (adapter_rank).
    - More attention where it is cheap: extra depth on the lowest-resolution
      decoder stages (lowres_extra_depth/lowres_levels).

    Key ideas:
    - Multiple passes per scale using the same attention + FFN weights
      (stage_depths) to add depth without adding parameters.
    - Narrower feedforward expansion (ff_mult) to control parameter growth.
    - Optional lightweight decoder attention blocks (heads, dim_head) applied
      after upsampling; also repeated without extra weights if desired.
    """
    def __init__(
        self,
        wave='haar',
        input_dim=3,
        output_dim=3,
        dim=128,
        dims=[],
        stage_depths=None,
        decoder_depths=None,
        ff_mult=3.0,
        use_efficient_attention=True,
        efficient_layers=(0, 1, 2),
        add_grid=False,
        patch_size=None,
        decoder_attn=True,
        decoder_heads=4,
        decoder_dim_head=32,
        use_checkpoint=True,
        lowres_extra_depth=0,
        lowres_levels=1,
        adapter_rank=16,
        adapter_dropout=0.0,
    ):
        super().__init__()
        self.add_grid = add_grid
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.lowres_extra_depth = lowres_extra_depth
        self.lowres_levels = lowres_levels
        self.adapter_rank = adapter_rank
        self.adapter_dropout = adapter_dropout

        if len(dims) == 0 and dim is not None:
            # Cap the width so the default stays around ~15M params.
            dims = np.array([dim, dim * 2, dim * 3, dim * 3])
        else:
            dims = np.array(dims)
        self.n_layers = len(dims)

        stage_depths = stage_depths if stage_depths is not None else [2] * self.n_layers
        decoder_depths = decoder_depths if decoder_depths is not None else list(reversed(stage_depths))

        if patch_size is None:
            self.input_proj = nn.Linear(input_dim, dims[0])
            self.output_proj = nn.Sequential(
                nn.Linear(dims[0], dims[0] // 2),
                nn.GELU(),
                nn.Linear(dims[0] // 2, output_dim),
            )

        self.use_efficient_attention = use_efficient_attention
        self.stage_depths = stage_depths
        self.decoder_depths = decoder_depths
        self.decoder_attn_enabled = decoder_attn

        self.enc_adapters = nn.ModuleList([])
        self.enc_layers = nn.ModuleList([])
        for i in range(self.n_layers):
            dim_i = dims[i]
            efficient_flag = i in efficient_layers and self.use_efficient_attention
            attn_layer = nn.ModuleList(
                [
                    nn.LayerNorm(dim_i),
                    WaveletAttentionBlock(wave=wave, dim=dim_i, use_efficient_attention=efficient_flag),
                    nn.LayerNorm(dim_i),
                    FeedForward(dim_i, int(dim_i * ff_mult)),
                ]
            )

            down_layer = nn.ModuleList(
                [
                    nn.Linear(dim_i, dim_i // 4),
                    nn.LayerNorm(dim_i // 4),
                    DWT_2D(wave),
                    nn.Conv2d(
                        dim_i,
                        dims[i + 1] if i < self.n_layers - 1 else dim_i,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        groups=4,
                    ),
                ]
            )

            self.enc_layers.append(nn.ModuleList([attn_layer, down_layer]))
            if adapter_rank > 0:
                self.enc_adapters.append(Adapter(dim_i, bottleneck=adapter_rank, dropout=adapter_dropout))
            else:
                self.enc_adapters.append(None)

        self.dec_layers = nn.ModuleList([])
        self.dec_attn_layers = nn.ModuleList([])
        self.dec_adapters = nn.ModuleList([])
        for i in range(self.n_layers):
            dim_i = dims[self.n_layers - i - 1]
            new_dim = dims[self.n_layers - i - 2] if self.n_layers - i - 2 >= 0 else dim_i
            efficient_flag = self.n_layers - i - 1 in efficient_layers and self.use_efficient_attention
            up_layer = nn.ModuleList(
                [
                    nn.Linear(dim_i, dim_i * 4),
                    nn.LayerNorm(dim_i * 4),
                    IDWT_2D(wave),
                    nn.Conv2d(2 * dim_i, new_dim, kernel_size=3, padding=1, stride=1, groups=4),
                ]
            )
            self.dec_layers.append(up_layer)

            if self.decoder_attn_enabled:
                self.dec_attn_layers.append(
                    LiteDecoderAttentionBlock(
                        new_dim, heads=decoder_heads, dim_head=decoder_dim_head, ff_mult=ff_mult
                    )
                )
            else:
                self.dec_attn_layers.append(None)
            if adapter_rank > 0:
                self.dec_adapters.append(Adapter(new_dim, bottleneck=adapter_rank, dropout=adapter_dropout))
            else:
                self.dec_adapters.append(None)

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
        x = dwt(x)
        x = conv(x)
        h, w = x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, h, w

    def up_block(self, x, x_prev, layer, h, w):
        linear, ln, idwt, conv = layer
        x = ln(linear(x))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = idwt(x)
        x = torch.cat((x, x_prev), dim=1)
        x = conv(x)
        h, w = x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, h, w

    def _maybe_checkpoint(self, fn, x):
        if self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(fn, x)
        return fn(x)

    def _checkpoint_attn_block(self, x, layer, h, w):
        return self._maybe_checkpoint(lambda t: self.attention_block(t, layer, h, w), x)

    def _checkpoint_module(self, module, x):
        if module is None:
            return x
        return self._maybe_checkpoint(lambda t: module(t), x)

    def forward(self, x):
        if self.add_grid:
            x_grid = self.get_grid(x)
            x = torch.cat((x, x_grid), dim=-1)

        x = self.input_proj(x)
        h, w = x.shape[1], x.shape[2]
        x = rearrange(x, 'b h w c -> b (h w) c')

        x_list = []

        for (attn_layer, down_layer), adapter, depth in zip(self.enc_layers, self.enc_adapters, self.stage_depths):
            x_list.append(rearrange(x, 'b (h w) c -> b c h w', h=h, w=w))
            for _ in range(depth):
                x = self._checkpoint_attn_block(x, attn_layer, h, w)
                if adapter is not None:
                    x = x + self._checkpoint_module(adapter, x)
            x, h, w = self.down_block(x, down_layer, h, w)

        for up_layer_idx, up_layer in enumerate(self.dec_layers):
            x, h, w = self.up_block(x, x_list.pop(), up_layer, h, w)
            if self.decoder_attn_enabled:
                attn_block = self.dec_attn_layers[up_layer_idx]
                adapter = self.dec_adapters[up_layer_idx]
                depth = self.decoder_depths[up_layer_idx]
                if up_layer_idx < self.lowres_levels:
                    depth += self.lowres_extra_depth
                for _ in range(depth):
                    x = self._checkpoint_module(attn_block, x)
                    if adapter is not None:
                        x = x + self._checkpoint_module(adapter, x)

        x = self.output_proj(x)
        x = rearrange(x, 'b (h w) c-> b h w c', h=h, w=w)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



if __name__ == "__main__":
    x = torch.randn(2, 64, 64, 3)
    # x = torch.rand(2, 96, 192, 3)
    # model = MultiscaleWaveletTransformer2D(input_dim=3, output_dim=3, dim=64, use_efficient_attention=True)
    # Baseline decoder-only attention-free variant
    # model = MultiscaleWaveletTransformer2DDecoderNoAttention(input_dim=3, output_dim=3, dim=96, use_efficient_attention=True)
    # Parameter-efficient deep variant (~14M params with defaults)
    model = MultiscaleWaveletTransformer2DDecoderPE(input_dim=3, output_dim=3, use_checkpoint=True, lowres_levels=2)
    
    print("number of parameters:", model.count_parameters())
    with torch.autograd.set_detect_anomaly(True):
        output = model(x)
        output = output[0] if isinstance(output, tuple) else output
        print("output shape:", output.shape)
        loss = output.mean()
        loss.backward()
        print("backward done")
