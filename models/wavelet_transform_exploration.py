import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from wavelet_transform import DWT_2D, IDWT_2D
from wavelet_transform import RelativePositionBias, Transformer



class WaveletTransformer(nn.Module):
    def __init__(self, wave='haar',in_chans=3, out_chans=3, in_timesteps = 4,  dim=64, depth=5,patch_size=(4, 4), normalize=False, meanstd=False):
        super(WaveletTransformer, self).__init__()
        self.meanstd = meanstd
        self.patch_size = patch_size
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_timesteps*in_chans, dim, kernel_size=patch_size, stride=patch_size, padding=0),
            nn.BatchNorm2d(dim),
            nn.ELU(inplace=True),
            )  # (B, D, H, W)
    

        # DWT modules
        self.num_dwt_blocks = 4
        self.dwt_project = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 4, kernel_size=1, stride=1, padding=0),
                           nn.BatchNorm2d(dim // 4),
                           nn.ELU(inplace=True)
                          ) for i in range(self.num_dwt_blocks)
        ])

        self.idwt_project = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels=dim//4, out_channels=dim, kernel_size=1, stride=1, padding=0),
                           nn.BatchNorm2d(dim),
                           nn.ELU(inplace=True)
                          ) for i in range(self.num_dwt_blocks)
        ])
        self.dwt = DWT_2D(wave)
        self.idwt = IDWT_2D(wave)

        # position and scale embeddings
        self.scale_embeddings = nn.ParameterList([
            nn.Parameter(torch.rand(1, 1, dim)) for _ in range(self.num_dwt_blocks)
        ])
        self.relative_position_embeddings = RelativePositionBias(dim=dim)

        # transformer for cross scale attention
        self.transformer =Transformer(dim=dim, depth=depth, heads=8, dim_head=64, mlp_dim=dim*4)

        # final output layer
        if self.meanstd:
            self.output_proj =  nn.ConvTranspose2d(dim, out_chans*2, kernel_size=patch_size, stride=patch_size, padding=0)
        else:
            self.output_proj =  nn.ConvTranspose2d(dim, out_chans, kernel_size=patch_size, stride=patch_size, padding=0)
        self.normalize = normalize
        


    def forward(self, x):
        x = x.view(*x.shape[:-2], -1)           #### B, X, Y, T*C
        # Store original spatial dimensions
        orig_h, orig_w = x.shape[1], x.shape[2]
        
        # get grid and concat
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
            x = (x - mu)/ sigma

        x = x.permute(0, 3, 1, 2).contiguous()  # (B, H, W, C)->(B, C, H, W)

        # Calculate padding needed for patch_size
        patch_h, patch_w = self.patch_size
        pad_h = (patch_h - (orig_h % patch_h)) % patch_h
        pad_w = (patch_w - (orig_w % patch_w)) % patch_w
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # input shape: (B, C, H, W)
        # projecting to a higher dimension, 2d convolution with kernel size 1
        x = self.input_proj(x) # (B, D, H, W)
        
        # Store the projected dimensions (after patch embedding)
        proj_h, proj_w = x.shape[-2], x.shape[-1]
        
        # run several blocks of DWT to obtain the wavelet coefficients
        x_scale = []
        img_size = []
        img_dims = []  # Store actual (height, width) dimensions
        break_idx = 0
        for i in range(self.num_dwt_blocks):
            x = self.dwt_project[i](x)  # (B, D//4, H, W)
            if min(x.shape[-1], x.shape[-2]) ==1: # too small for wavelet transform
                break_idx = i
                break
            # Store original dimensions before DWT
            orig_h_dwt, orig_w_dwt = x.shape[-2], x.shape[-1]
            img_dims.append((orig_h_dwt, orig_w_dwt))
            
            # apply DWT
            x = self.dwt(x) # (B, 4*D//4, H//2, W//2)
            img_size.append((x.shape[2]*x.shape[3]))
            x_scale_input = rearrange(x, 'b d h w -> b (h w) d')  # (B, D, H//2 * W//2)
            # generate relative position embeddings for every position in the [H//2, W//2] grid
            pos_embed = self.relative_position_embeddings(x.shape[2], x.shape[3], x.device) # shape (B, D, H//2 * W//2)
            # scale embeddings for each scale
            scale_embed = self.scale_embeddings[i]  # (B, D, 1)
            
            x_scale_input = x_scale_input + pos_embed + scale_embed # (B, D, H//2 * W//2)
            x_scale.append(x_scale_input)

        # concatenate the wavelet coefficients from all scales
        x = torch.cat(x_scale, dim=1) # (B, N, D)
    

        # apply transformer to the wavelet coefficients
        x = self.transformer(x)
        x = rearrange(x, 'b n d -> b d n')

        # recover the original image with IDWT
        # first split x based on image size to get the wavelet coefficients for each scale
        x_splits = torch.split(x, img_size, dim=-1)
        
        x_recov = torch.zeros_like(x_splits[-1])  # initialize the recovered image
        for i, x_split in enumerate(x_splits[::-1]):  # reverse the order to match the DWT order
            x_recov = x_recov + x_split # combine the wavelet from the current scale (x_split) with the previous recovered image (x_recov)
            # get the scale size and dimensions
            scale_idx = len(x_splits) - 1 - i
            scale_size = img_size[scale_idx]
            orig_h_dwt, orig_w_dwt = img_dims[scale_idx]
            
            # Calculate the DWT output dimensions (after stride=2 with padding)
            dwt_h = (orig_h_dwt + 1) // 2  # This accounts for padding in DWT
            dwt_w = (orig_w_dwt + 1) // 2
            
            # start with the last scale, which has the smallest size
            x_recov = rearrange(x_recov, 'b (p d) (h w) -> b p d h w', p = self.num_dwt_blocks, h=dwt_h, w=dwt_w)
            # apply IDWT to the wavelet coefficients with target size
            x_recov = self.idwt(x_recov, target_size=(orig_h_dwt, orig_w_dwt)) # (b d h w)
            x_recov = self.idwt_project[i](x_recov)  # (B, 4*d, H, W))
            x_recov = rearrange(x_recov, 'b d h w -> b d (h w)') # (b d H*W)
        # the final output is the recovered image from the last scale
        # Get the final dimensions from the first scale (which is the original projected size)
        final_h, final_w = img_dims[0] if img_dims else (proj_h, proj_w)
        x = rearrange(x_recov, 'b d (h w) -> b d h w', h=final_h, w=final_w)  # (B, D, H, W)
        x = self.output_proj(x) # (B, 3, H, W)
        
        # Crop back to original dimensions if padding was applied
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :orig_h, :orig_w]
        
        # reshape to (B, C, H, W)
        x = rearrange(x, 'b c h w -> b h w 1 c')
        if self.normalize:
            x = x * sigma  + mu
        return x

    def count_parameters(self):

        # count the parameters  of dwt_project and idwt_project
        dwt_params = sum(p.numel() for p in self.dwt_project.parameters() if p.requires_grad)
        idwt_params = sum(p.numel() for p in self.idwt_project.parameters() if p.requires_grad)
        transformer_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        input_proj_params = sum(p.numel() for p in self.input_proj.parameters() if p.requires_grad)
        print("DWT parameters:", dwt_params)
        print("IDWT parameters:", idwt_params)
        print("Transformer parameters:", transformer_params)
        print("Input projection parameters:", input_proj_params)
        print("Total parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class WaveletTransformerOverlap(WaveletTransformer):
    """
    Variant with overlapping patch embeddings and a light deblocking head to
    mitigate 4x4 blocking artifacts without changing the wavelet/transformer core.
    """
    def __init__(self, patch_stride=2, use_deblock=False, **kwargs):
        super().__init__(**kwargs)
        self.patch_stride = patch_stride
        self.use_deblock = use_deblock
        self.out_chans = kwargs.get('out_chans', 3)
        self.embed_dim = self.scale_embeddings[0].shape[-1]
        patch_h, patch_w = self.patch_size
        padding = (patch_h // 2, patch_w // 2)

        in_ch = kwargs.get('in_timesteps', 4) * kwargs.get('in_chans', 3)
        out_ch = self.out_chans * (2 if self.meanstd else 1)

        # Overlapping convs replace the stride=patch embedding
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, self.embed_dim,
                      kernel_size=self.patch_size, stride=self.patch_stride,
                      padding=padding),
            nn.BatchNorm2d(self.embed_dim),
            nn.ELU(inplace=True),
        )
        self.output_proj = nn.ConvTranspose2d(self.embed_dim, out_ch,
                                              kernel_size=self.patch_size, stride=self.patch_stride,
                                              padding=padding)

        if use_deblock:
            deblock_channels = self.out_chans
            self.deblock = nn.Identity()

    def forward(self, x):
        orig_h, orig_w = x.shape[1], x.shape[2]
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6
            x = (x - mu)/ sigma
        x = x.view(*x.shape[:-2], -1).permute(0, 3, 1, 2).contiguous()

        x = self.input_proj(x)
        proj_h, proj_w = x.shape[-2], x.shape[-1]

        x_scale, img_size, img_dims = [], [], []
        for i in range(self.num_dwt_blocks):
            x = self.dwt_project[i](x)
            if min(x.shape[-1], x.shape[-2]) ==1:
                break
            orig_h_dwt, orig_w_dwt = x.shape[-2], x.shape[-1]
            img_dims.append((orig_h_dwt, orig_w_dwt))
            x = self.dwt(x)
            img_size.append((x.shape[2]*x.shape[3]))
            x_scale_input = rearrange(x, 'b d h w -> b (h w) d')
            pos_embed = self.relative_position_embeddings(x.shape[2], x.shape[3], x.device)
            scale_embed = self.scale_embeddings[i]
            x_scale_input = x_scale_input + pos_embed + scale_embed
            x_scale.append(x_scale_input)

        x = torch.cat(x_scale, dim=1)
        x = self.transformer(x)
        x = rearrange(x, 'b n d -> b d n')

        x_splits = torch.split(x, img_size, dim=-1)
        x_recov = torch.zeros_like(x_splits[-1])
        for i, x_split in enumerate(x_splits[::-1]):
            x_recov = x_recov + x_split
            scale_idx = len(x_splits) - 1 - i
            orig_h_dwt, orig_w_dwt = img_dims[scale_idx]
            dwt_h = (orig_h_dwt + 1) // 2
            dwt_w = (orig_w_dwt + 1) // 2
            x_recov = rearrange(x_recov, 'b (p d) (h w) -> b p d h w',
                                p = self.num_dwt_blocks, h=dwt_h, w=dwt_w)
            x_recov = self.idwt(x_recov, target_size=(orig_h_dwt, orig_w_dwt))
            x_recov = self.idwt_project[i](x_recov)
            x_recov = rearrange(x_recov, 'b d h w -> b d (h w)')

        final_h, final_w = img_dims[0] if img_dims else (proj_h, proj_w)
        x = rearrange(x_recov, 'b d (h w) -> b d h w', h=final_h, w=final_w)
        x = self.output_proj(x)
        x = x[:, :, :orig_h, :orig_w]

        if self.meanstd:
            value, scale = torch.split(x, x.shape[1]//2, dim=1)
            if self.use_deblock:
                value = self.deblock(value)
            x = torch.cat([value, scale], dim=1)
        else:
            if self.use_deblock:
                x = self.deblock(x)
        x = rearrange(x, 'b c h w -> b h w 1 c')
        if self.normalize:
            x = x * sigma  + mu
        return x


class WaveletTransformerHFSKip(nn.Module):
    """
    Band-aware variant: LL goes through the transformer; LH/HL/HH are preserved
    via skip connections and re-injected before each IDWT to keep high-frequency detail.
    """
    def __init__(self, wave='haar', in_chans=3, out_chans=3, in_timesteps=1, dim=64,
                 depth=8, num_levels=4, patch_size=(8,8), patch_stride=4, normalize=False, meanstd=False,
                 use_deblock=False):
        super().__init__()
        self.meanstd = meanstd
        self.normalize = normalize
        self.patch_size = patch_size
        self.dim = dim
        self.num_levels = num_levels
        self.out_chans = out_chans
        self.use_deblock = use_deblock

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_timesteps*in_chans, dim, kernel_size=patch_size, stride=patch_stride, padding=0),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.level_pre = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU()) for _ in range(num_levels)])
        self.hf_proj = nn.ModuleList([nn.Conv2d(3*dim, 3*dim, kernel_size=1) for _ in range(num_levels)])

        self.dwt = DWT_2D(wave)
        self.idwt = IDWT_2D(wave)

        self.scale_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, dim) * 0.02) for _ in range(num_levels)
        ])
        self.relative_position_embeddings = RelativePositionBias(dim=dim)
        self.transformer = Transformer(dim=dim, depth=depth, heads=8, dim_head=64, mlp_dim=dim*4)

        out_ch = out_chans * (2 if meanstd else 1)
        self.output_proj = nn.ConvTranspose2d(dim, out_ch,
                                              kernel_size=patch_size, stride=patch_stride, padding=0)
        if use_deblock:
            self.deblock = nn.Identity()

    def forward(self, x):
        orig_h, orig_w = x.shape[1], x.shape[2]
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6
            x = (x - mu)/ sigma

        x = x.view(*x.shape[:-2], -1).permute(0, 3, 1, 2).contiguous()
        pad_h = (self.patch_size[0] - (orig_h % self.patch_size[0])) % self.patch_size[0]
        pad_w = (self.patch_size[1] - (orig_w % self.patch_size[1])) % self.patch_size[1]
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        x = self.input_proj(x)

        tokens, hf_skips, level_shapes, pre_shapes, lengths = [], [], [], [], []
        feat = x
        for lvl in range(self.num_levels):
            pre_shapes.append(feat.shape[-2:])  # shape before DWT, used for IDWT target size
            feat = self.level_pre[lvl](feat)
            if min(feat.shape[-2:]) <= 1:
                break
            bands = self.dwt(feat)  # (B, 4*dim, H/2, W/2)
            h, w = bands.shape[-2], bands.shape[-1]
            level_shapes.append((h, w))
            LL = bands[:, :self.dim, :, :]
            HF = bands[:, self.dim:, :, :]
            hf_skips.append(self.hf_proj[lvl](HF)) # mix the high frequency together
            tok = rearrange(LL, 'b d h w -> b (h w) d')
            pos = self.relative_position_embeddings(h, w, bands.device)
            tok = tok + pos + self.scale_embeddings[lvl]
            tokens.append(tok)
            lengths.append(tok.shape[1])
            feat = LL
            if min(h, w) <= 2:
                break

        if len(tokens) == 0:
            raise RuntimeError("Input too small for any DWT level.")

        tokens_cat = torch.cat(tokens, dim=1)
        tokens_cat = self.transformer(tokens_cat)
        tokens_split = torch.split(tokens_cat, lengths, dim=1)

        feat = None
        for lvl in reversed(range(len(tokens_split))):
            h, w = level_shapes[lvl]
            LL_hat = rearrange(tokens_split[lvl], 'b (h w) d -> b d h w', h=h, w=w)
            if feat is not None:
                feat_up = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
                LL_hat = LL_hat + feat_up
            bands = torch.cat([LL_hat, hf_skips[lvl]], dim=1)
            bands = rearrange(bands, 'b (p d) h w -> b p d h w', p=4)
            feat = self.idwt(bands, target_size=pre_shapes[lvl])

        x = self.output_proj(feat)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :orig_h, :orig_w]

        if self.meanstd:
            value, scale = torch.split(x, x.shape[1]//2, dim=1)
            if self.use_deblock:
                value = self.deblock(value)
            x = torch.cat([value, scale], dim=1)
        else:
            if self.use_deblock:
                x = self.deblock(x)
        x = rearrange(x, 'b c h w -> b h w 1 c')
        if self.normalize:
            x = x * sigma + mu
        return x


def build_wavelet_overlap_small(in_chans=3, out_chans=3, in_timesteps=1):
    """~16M params (reference): dim=128, depth=4, patch 8/stride 4."""
    return WaveletTransformerOverlap(
        in_chans=in_chans,
        out_chans=out_chans,
        in_timesteps=in_timesteps,
        patch_size=(8, 8),
        patch_stride=4,
        dim=128,
        depth=4,
    )


def build_wavelet_overlap_medium(in_chans=3, out_chans=3, in_timesteps=1):
    """Target ~30–40M params: dim=192, depth=6, patch 8/stride 4."""
    return WaveletTransformerOverlap(
        in_chans=in_chans,
        out_chans=out_chans,
        in_timesteps=in_timesteps,
        patch_size=(8, 8),
        patch_stride=4,
        dim=192,
        depth=5,
    )


def build_wavelet_overlap_big(in_chans=3, out_chans=3, in_timesteps=1):
    """Target ~60M params: dim=256, depth=8, patch 8/stride 4."""
    return WaveletTransformerOverlap(
        in_chans=in_chans,
        out_chans=out_chans,
        in_timesteps=in_timesteps,
        patch_size=(8, 8),
        patch_stride=4,
        dim=256,
        depth=8,
    )

    def count_parameters(self):
    # count the parameters  of dwt_project and idwt_project
        transformer_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        input_proj_params = sum(p.numel() for p in self.input_proj.parameters() if p.requires_grad)
        output_proj_params = sum(p.numel() for p in self.output_proj.parameters() if p.requires_grad)
        print("Transformer parameters:", transformer_params)
        print("Input projection parameters:", input_proj_params)
        print("Output projection parameters:", output_proj_params)
        # count the parameters  of dwt_project and idwt_project
        print("Total parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.randn(2, 256, 256, 1, 3)
    # x = torch.randn(2, 96, 192, 7, 3)
    print("x shape:", x.shape)
    # dwt = DWT_2D('haar')
    # idwt = IDWT_2D('haar')
    # x = dwt(x)
    # print("after wavelet transform", x.shape)
    # x = idwt(x)
    # print("after inverse wavelet transform", x.shape)

    # model = WaveletTransformer(in_chans=x.shape[-1],out_chans=x.shape[-1], in_timesteps=x.shape[-2], dim=128, depth=4)
    model = WaveletTransformerOverlap(in_chans=x.shape[-1],out_chans=x.shape[-1], patch_size=(8, 8), patch_stride=4, in_timesteps=x.shape[-2], dim=192, depth=5)
    # model = WaveletTransformerHFSKip(in_chans=x.shape[-1],out_chans=x.shape[-1], in_timesteps=x.shape[-2], dim=96, depth=4, num_levels=4)
    model.count_parameters()
    with torch.autograd.set_detect_anomaly(True):
        output = model(x)
        print("output shape:", output.shape)
        loss = output.mean()
        loss.backward()
        print("backward done")
    
