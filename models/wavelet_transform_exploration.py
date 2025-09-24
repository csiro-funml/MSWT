import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import pywt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from wavelet_utils import UPerHead

def build_bn(num_features, requires_grad=False):
    bn = nn.BatchNorm2d(num_features)
    if not requires_grad:
        for param in bn.parameters():
            param.requires_grad = False
    return bn

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)

    def forward(self, x):
        """
        x: (B, 4*C, H//2, W//2)
        return: (B, C, H, W)
        """
        return IDWT_Function.apply(x, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, 4*C, H//2, W//2)
        """
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, 
        dim, 
        num_heads, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        sr_ratio=1, 
        block_type = 'wave'
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        self.pre_attn_conv = nn.Conv2d(4*dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.post_attn_conv = nn.Conv2d(dim, 4*dim, 3, 1, 1, bias=True, groups=dim)

        self.attn = Attention(dim, num_heads)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dwt = DWT_2D('haar')
        self.idwt = IDWT_2D('haar')
        self.apply(self._init_weights)
       
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    
    def forward(self, x, H, W, dowmsample=False):
        """
        x: (B, N, C)
        H: int
        W: int
        return: (B, N, C)
        """
        # reshape the input to (B, H, W, C)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        # run the DWT layer: (B, C, H, W) -> (B, 4*C, H//2, W//2)
        x = self.dwt(x)
        
        x = self.pre_attn_conv(x) # (B, 4*C, H//2, W//2) -> (B, C, H//2, W//2)
        nh, nw = x.shape[-2], x.shape[-1]
        # reshape the input to (B, N//4, C)
        x = x.flatten(2).transpose(1, 2)

        # standard transformer block
        x = x + self.drop_path(self.attn(self.norm1(x), nh, nw))
        x = x + self.drop_path(self.mlp(self.norm2(x), nh, nw))

        # reshape the input to (B, C, H//2, W//2)
        x = x.transpose(1, 2).view(B, C, nh, nw)
        x = self.post_attn_conv(x) # (B, C, H//2, W//2) -> (B, 4*C, H//2, W//2)
        
        # if not downsample, run the idwt to get the original shape (B, 4*C, H//2, W//2) -> (B, C, H, W)
        # if downsample, only take the first low frequency coefficients (B, 4*C, H//2, W//2) -> (B,C, H//2, W//2)
        x = self.idwt(x) if not dowmsample else x[:, :C]
        H, W = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            build_bn(hidden_dim, requires_grad=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            build_bn(hidden_dim, requires_grad=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            build_bn(hidden_dim, requires_grad=False),
            nn.ReLU(inplace=False),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x) # subsample onece, convolution with different kernel sizes
        x = self.proj(x) # subsample twice
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class WaveletTransformer(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, in_timesteps=7):
        super(WaveletTransformer, self).__init__()
        self.num_stages = 4
        stem_hidden_dim = 32 
        embed_dims = [64, 128, 320, 448]
        num_heads = [2, 4, 10, 14]
        mlp_ratios = [8, 8, 4, 4] 
        drop_path_rate = 0. 
        depths = [3, 4, 6, 3]
        sr_ratios = [4, 2, 1, 1]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        blocks =[]
        norms = []
        patch_embeds = []
        cur = 0
        for i in range(self.num_stages):
            if i == 0:
                patch_embed = Stem(in_timesteps*in_chans+2, stem_hidden_dim, embed_dims[i]) # plus 3 for time coordinates
            else:
                # patch_embed = nn.Identity()
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i]) # no downsampling, just a projection layer

            block = nn.ModuleList([Block(
                dim=embed_dims[i], 
                num_heads=num_heads[i], 
                mlp_ratio=mlp_ratios[i],
                drop_path=dpr[cur + j], 
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], 
                block_type='wave' if i < 2 else 'std_att')
            for j in range(depths[i])])
            
            norm = norm_layer(embed_dims[i])
            
            cur = cur + depths[i]
            norms.append(norm)
            blocks.append(block)
            patch_embeds.append(patch_embed)
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList(norms)
        self.patch_embeds = nn.ModuleList(patch_embeds)
        # FPN head as a decoder
        self.decoder = UPerHead(
                        in_channels=[64, 64, 128, 320, 448],
                        in_index=[0, 1, 2, 3, 4],
                        pool_scales=(1, 1, 2, 3, 6),
                        channels=256,
                        dropout_ratio=0.1,
                        output_channels=out_chans,
                        output_size=(128, 128),
                            )

    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid

    def forward(self, x):
        B, H, W, T, C = x.shape
        x = x.view(*x.shape[:-2], -1)           #### B, H, W, T*C
        grid = self.get_grid(x)
        x = torch.cat((x, grid), dim=-1)        #### B, H, W, T*C +2
        x = x.permute(0, 3, 1, 2).contiguous() # (B, T*C+2, H, W)

        outs = []

        for i in range(self.num_stages):
            patch_embed = self.patch_embeds[i]
            block = self.blocks[i]
            norm = self.norms[i]
            x, H, W = patch_embed(x) # for the first stage, we need to reduce the input size  by 4
            if i == 0:
                outs.append(x.transpose(1, 2).view(B, -1, H, W))
            for k, blk in enumerate(block): 
                # inside each block, we run x' = IDWT(ViT(DWT(x))), 
                # if it is the last stage, we pass the first componeof x' to the next block without IDWT
                x, H, W = blk(x, H, W, dowmsample= k == len(block) - 1) # we update the H and W after the last block
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        
        output = self.decoder(outs)
        output = output.permute(0, 2, 3, 1).unsqueeze(-2).contiguous()
        return output
     
    def count_parameters(self):
        print("total parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))
        # count parameters for each modules
        print("parameters for blocks:", sum(p.numel() for p in self.blocks.parameters() if p.requires_grad))
        print("parameters for norms:", sum(p.numel() for p in self.norms.parameters() if p.requires_grad))
        print("parameters for patch_embeds:", sum(p.numel() for p in self.patch_embeds.parameters() if p.requires_grad))
        print("parameters for decoder:", sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))
        




if __name__ == "__main__":
    x = torch.randn(2, 128, 128, 7, 3)
    print("x shape:", x.shape)
    # dwt = DWT_2D('haar')
    # idwt = IDWT_2D('haar')
    # x = dwt(x)
    # print("after wavelet transform", x.shape)
    # x = idwt(x)
    # print("after inverse wavelet transform", x.shape)

    model = WaveletTransformer()
    model.count_parameters()
    with torch.autograd.set_detect_anomaly(True):
        output = model(x)
        print("output shape:", output.shape)
        loss = output.mean()
        loss.backward()
        print("backward done")
    


