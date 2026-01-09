"""
Multiscale Wavelet Transformer for 2D data

Codes for backpropagation of wavelet transforms are referenced from https://github.com/YehLi/ImageNetModel/blob/main/classification/torch_wavelets.py 
"""

import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh, format='cat'):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        
        # Store original shape for reconstruction
        B, C, H, W = x.shape
        ctx.shape = x.shape
        
        # Calculate padding needed for odd dimensions
        pad_h = H % 2
        pad_w = W % 2
        
        # Apply padding if needed (pad on right and bottom)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Store padded shape and padding info for backward pass
        ctx.padded_shape = x.shape
        ctx.padding = (pad_w, pad_h)

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        if format == 'cat':
            x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        else:
            x = torch.stack([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            pad_w, pad_h = ctx.padding
            
            dx = dx.view(B, 4, -1, dx.shape[-2], dx.shape[-1])
            dx = dx.transpose(1,2).reshape(B, -1, dx.shape[-2], dx.shape[-1])
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)
            
            # Remove padding if it was added in forward pass
            if pad_h > 0 or pad_w > 0:
                dx = dx[:, :, :H, :W]

        return dx, None, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters, target_size=None):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        ctx.target_size = target_size
        ctx.is_cat_format = (x.dim() == 4)

        B, H, W = x.shape[0], x.shape[-2], x.shape[-1]

        # x = x.transpose(1, 2)  # (B, D//4, 4, H, W)
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        
        # Store the uncropped size for backward pass
        ctx.uncropped_size = (x.shape[-2], x.shape[-1])
        
        # If target size is provided, crop to match it
        if target_size is not None:
            target_h, target_w = target_size
            current_h, current_w = x.shape[-2], x.shape[-1]
            if current_h > target_h or current_w > target_w:
                x = x[:, :, :target_h, :target_w]
        
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            if ctx.is_cat_format:
                # Input was (B, 4*C, H, W) so recover channel count from original shape
                C = ctx.shape[1] // 4
            else:
                # Input was stacked (B, 4, C, H, W)
                C = ctx.shape[2]
            
            # If we cropped in forward pass, we need to pad dx back to uncropped size
            if ctx.target_size is not None:
                uncropped_h, uncropped_w = ctx.uncropped_size
                current_h, current_w = dx.shape[-2], dx.shape[-1]
                
                if uncropped_h > current_h or uncropped_w > current_w:
                    pad_h = uncropped_h - current_h
                    pad_w = uncropped_w - current_w
                    dx = F.pad(dx, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.stack([x_ll, x_lh, x_hl, x_hh], dim=1)
            # Restore original layout to keep autograd shapes consistent with the forward input
            if ctx.is_cat_format:
                dx = dx.reshape(ctx.shape[0], 4 * C, dx.shape[-2], dx.shape[-1])
            else:
                dx = dx.reshape(ctx.shape[0], 4, C, dx.shape[-2], dx.shape[-1])
        return dx, None, None


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
        # self.filters = self.filters.to(dtype=torch.float16)

    def forward(self, x, target_size=None):
        return IDWT_Function.apply(x, self.filters, target_size)

class DWT_2D(nn.Module):
    def __init__(self, wave, format='cat'):
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
        self.format = format
        # self.w_ll = self.w_ll.to(dtype=torch.float16)
        # self.w_lh = self.w_lh.to(dtype=torch.float16)
        # self.w_hl = self.w_hl.to(dtype=torch.float16)
        # self.w_hh = self.w_hh.to(dtype=torch.float16)

    def forward(self, x, format='cat'):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh, self.format)
        

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)



class WaveletAttentionBlock(nn.Module):
    def __init__(self, wave='haar', dim=64, use_efficient_attention=False, local_attention_size=8, **kwargs):
        super().__init__(**kwargs)
        self.dwt = DWT_2D(wave)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim//4),
             nn.LayerNorm(dim//4))
        self.conv_post =  self.filter = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
                # nn.BatchNorm2d(dim),
            )
        self.idwt = IDWT_2D(wave)
        self.attention = Attention(dim)
        self.final_proj = nn.Linear(dim//4, dim)
        self.use_efficient_attention = use_efficient_attention
        if self.use_efficient_attention:
            self.local_attention_size = local_attention_size

    def old_local_attention(self, x, h, w):
        # input: (B, C, H, W) output : (B, C, H, W)
        b, c, h, w = x.shape
        new_h, new_w = h//self.local_attention_size, w//self.local_attention_size
        x = rearrange(x, 'b c (h_patch new_h) (w_patch new_w)-> (b new_h new_w) (h_patch w_patch) c', new_h=new_h, new_w=new_w)
        x = self.attention(x) # -> (B, H/2 x W/2, C)
        x = rearrange(x, '(b new_h new_w) (h_patch w_patch) c -> b c (h_patch new_h) (w_patch new_w)', b=b, new_h=new_h, new_w=new_w, h_patch=self.local_attention_size, w_patch=self.local_attention_size)
        return x
    
    def local_attention(self, x, h, w):
        # input: (B, C, H, W) output : (B, C, H, W)
        b, c, h, w = x.shape
        patch = self.local_attention_size
        pad_h = (patch - h % patch) % patch
        pad_w = (patch - w % patch) % patch
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        new_h, new_w = (h + pad_h)//patch, (w + pad_w)//patch
        x = rearrange(
            x,
            'b c (h_patch new_h) (w_patch new_w)-> (b new_h new_w) (h_patch w_patch) c',
            h_patch=patch, w_patch=patch, new_h=new_h, new_w=new_w,
        )
        x = self.attention(x) # -> (B, H/2 x W/2, C)
        x = rearrange(
            x,
            '(b new_h new_w) (h_patch w_patch) c -> b c (h_patch new_h) (w_patch new_w)',
            b=b, new_h=new_h, new_w=new_w, h_patch=patch, w_patch=patch,
        )
        if pad_h or pad_w:
            x = x[:, :, :h, :w]
        return x
    
    def global_attention(self, x, h, w):
        # input: (B, C, H, W) output : (B, C, H, W)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.attention(x) # -> (B, H/2 x W/2, C)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
    
    def forward(self, x, h, w):
        """
        Input: (B, (H x W), c)
        Output: (B, (H, W), c)
        """
        b, c = x.shape[0], x.shape[-1]
        x = self.mlp_head(x) # (B, (H x W), C) -> (B, (H x W), C//4)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.dwt(x) # -> (B, 4, C//4, H/2, W/2)
        new_h, new_w = x.shape[-2], x.shape[-1]
        x = self.conv_post(x) # -> (B, 4C//4, H/2, W/2)
        if self.use_efficient_attention:
            x = self.local_attention(x, new_h, new_w)
            # x = self.old_local_attention(x, new_h, new_w)
        else:
            x = self.global_attention(x, new_h, new_w)
        x = torch.reshape(x, (b, 4, c//4, new_h, new_w))
        x = self.idwt(x) # -> (B, C, H, W)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.final_proj(x) # -> (B, (H x W), C)
        return x


class MultiscaleWaveletTransformer2D(nn.Module):
    def __init__(self, wave='haar', input_dim=3, output_dim=3, dim=None, dims=[], use_efficient_attention=False,
                       efficient_layers=[0, 1], add_grid=False, patch_size=None, **kwargs):
        super().__init__(**kwargs)
        self.add_grid = add_grid
        self.patch_size = patch_size

        if len(dims) == 0 and dim is not None:
            dims = np.array([dim//2, 2*dim, 4*dim, 4*dim])
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
                nn.Conv2d(dim, dims[i+1] if i < self.n_layers - 1 else dim, kernel_size=3, padding=1, stride=1),
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
                nn.Conv2d(2*dim, new_dim, kernel_size=3, padding=1, stride=1),
                ])
                
            attn_layer = nn.ModuleList([
                nn.LayerNorm(new_dim),
                WaveletAttentionBlock(wave=wave, dim=new_dim, use_efficient_attention=efficient_flag),
                nn.LayerNorm(new_dim),
                FeedForward(new_dim, new_dim*4)
                ])
            
            self.dec_layers.append(nn.ModuleList([up_layer, attn_layer]))

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
            
        
        for up_layer, attn_layer in self.dec_layers:
            x, h, w = self.up_block(x,  x_list.pop(), up_layer, h, w)
            x = self.attention_block(x, attn_layer, h, w)
        
        # x = self.norm(x)
        x = self.output_proj(x)
        x = rearrange(x, 'b (h w) c-> b h w c', h=h, w=w)
        return x


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)







if __name__ == "__main__":
    # x = torch.randn(2, 64, 64, 3)
    x = torch.rand(2, 96, 192, 3)
    # model = MultiscaleWaveletTransformer2D(input_dim=3, output_dim=1, dim=64, use_efficient_attention=True)
    # model = MultiscaleWaveletTransformer2D(input_dim=3, output_dim=1, dims=[64, 128, 256, 512], use_efficient_attention=True,   efficient_layers=[0, 1, 2])
    model = MultiscaleWaveletTransformer2D(input_dim=3, output_dim=1, dims=[32, 64, 128, 256, 512], use_efficient_attention=True,   efficient_layers=[0, 1, 2, 3])
    # model = MultiscaleWaveletTransformer2DDecoderNoAttention(input_dim=3, output_dim=1, dim=96, use_efficient_attention=True)
    
    print("number of parameters:", model.count_parameters())
    with torch.autograd.set_detect_anomaly(True):
        output = model(x)
        output = output[0] if isinstance(output, tuple) else output
        print("output shape:", output.shape)
        loss = output.mean()
        loss.backward()
        print("backward done")
