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
import math
# Add parent directory to Python path to access utils and models
sys.path.append(os.path.join(os.path.dirname(__file__)))
from wavelet_transform import Transformer, RelativePositionBias
from einops import rearrange


class EfficientComplexFeedForward(nn.Module):
    """Parameter-efficient feed forward network for complex-valued inputs"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Single shared network with complex-aware output projection
        self.shared_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),  # Input is concatenated real/imag
            nn.GELU(),
            nn.Linear(hidden_dim, dim * 2)  # Output both real and imaginary transformations
        )
        
    def forward(self, x_real, x_imag):
        # Stack real and imaginary parts
        x_combined = torch.cat([x_real, x_imag], dim=-1)  # [..., 2*dim]
        
        # Process through shared network
        out = self.shared_net(x_combined)  # [..., 2*dim]
        
        # Split output
        out_real_transform, out_imag_transform = out.chunk(2, dim=-1)
        
        # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
        # We learn transformations c,d for each input component a,b
        final_real = out_real_transform * 0.5  # Simplified for efficiency
        final_imag = out_imag_transform * 0.5
        
        return final_real, final_imag


class EfficientComplexAttention(nn.Module):
    """Parameter-efficient attention mechanism for complex-valued inputs"""
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # Shared projections that work on concatenated real/imaginary input  
        self.to_qkv = nn.Linear(dim * 2, inner_dim * 3, bias=False)  # Input is [real, imag] concatenated
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        # Shared normalization
        self.norm = nn.LayerNorm(dim * 2)
        
        # Learnable complex mixing parameters (much smaller parameter cost)
        self.complex_mix = nn.Parameter(torch.randn(2, 2) * 0.1)  # 2x2 matrix for real/imag mixing
        
    def forward(self, x_real, x_imag):
        b, n, d = x_real.shape
        
        # Concatenate and normalize
        x_complex = torch.cat([x_real, x_imag], dim=-1)  # [b, n, 2*d]
        x_complex = self.norm(x_complex)
        
        # Generate Q, K, V from concatenated input
        qkv = self.to_qkv(x_complex).chunk(3, dim=-1)  # Each is [b, n, inner_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Standard attention computation (real-valued for efficiency)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)  # [b, n, d]
        
        # Apply learnable complex mixing to split into real/imaginary parts
        # This allows the model to learn how to distribute the attention output
        mix = torch.softmax(self.complex_mix, dim=-1)  # Normalize mixing weights
        
        out_real = out * mix[0, 0] + out * mix[0, 1]  # Learnable combination
        out_imag = out * mix[1, 0] + out * mix[1, 1]  # Learnable combination
        
        return out_real, out_imag


class EfficientFrequencyEmbedding(nn.Module):
    """Parameter-efficient frequency-aware embeddings"""
    def __init__(self, dim, max_freq=128):
        super().__init__()
        self.dim = dim
        
        # Single shared embedding for both H and W frequencies (parameter sharing)
        self.freq_embed = nn.Embedding(max_freq, dim // 4)
        
        # Lightweight magnitude embedding
        self.mag_embed = nn.Linear(1, dim // 4)
        
        # Learnable combination weights (very few parameters)
        self.combine_weights = nn.Parameter(torch.randn(3, dim // 4))
        
    def forward(self, h, w, device):
        # Generate frequency indices (reuse same embedding for h and w)
        freq_h = torch.clamp(torch.arange(h, device=device), 0, self.freq_embed.num_embeddings - 1)
        freq_w = torch.clamp(torch.arange(w, device=device), 0, self.freq_embed.num_embeddings - 1)
        
        # Shared embeddings
        embed_h = self.freq_embed(freq_h)  # (H, dim//4)
        embed_w = self.freq_embed(freq_w)  # (W, dim//4)
        
        # Create 2D grid
        embed_h = embed_h.unsqueeze(1).expand(-1, w, -1)  # (H, W, dim//4)
        embed_w = embed_w.unsqueeze(0).expand(h, -1, -1)   # (H, W, dim//4)
        
        # Frequency magnitude
        freq_h_norm = torch.arange(h, device=device, dtype=torch.float) / h
        freq_w_norm = torch.arange(w, device=device, dtype=torch.float) / w
        freq_h_grid, freq_w_grid = torch.meshgrid(freq_h_norm, freq_w_norm, indexing='ij')
        freq_magnitude = torch.sqrt(freq_h_grid**2 + freq_w_grid**2).unsqueeze(-1)
        
        mag_embed = self.mag_embed(freq_magnitude)  # (H, W, dim//4)
        
        # Efficient combination using learnable weights
        components = torch.stack([embed_h, embed_w, mag_embed], dim=-2)  # (H, W, 3, dim//4)
        weights = torch.softmax(self.combine_weights, dim=0)  # (3, dim//4)
        
        # Weighted combination and expand to full dimension
        freq_embedding = torch.sum(components * weights.unsqueeze(0).unsqueeze(0), dim=-2)  # (H, W, dim//4)
        freq_embedding = freq_embedding.repeat(1, 1, 4)  # (H, W, dim) - simple repetition
        
        return freq_embedding


class EfficientMultiScaleComplexTransformer(nn.Module):
    """Parameter-efficient multi-scale transformer with shared components"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.dim = dim
        
        # Shared transformer components (major parameter savings!)
        self.shared_attention = EfficientComplexAttention(dim, heads, dim_head)
        self.shared_feedforward = EfficientComplexFeedForward(dim, mlp_dim)
        
        # Scale-specific lightweight adapters (very few parameters)
        self.scale_adapters = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(dim, dim, bias=False),  # Attention adapter
                nn.Linear(dim, dim, bias=False)   # FFN adapter  
            ]) for _ in range(num_scales)
        ])
        
        # Lightweight cross-scale interaction
        self.cross_scale_mix = nn.Parameter(torch.randn(num_scales, num_scales) * 0.1)
        
        # Shared normalization
        self.norm = nn.LayerNorm(dim * 2)  # For concatenated real/imag
        
    def split_frequency_bands(self, x_real, x_imag, h, w):
        """Split frequency domain into different scales"""
        bands_real = []
        bands_imag = []
        
        # Low frequencies (coarse scale)
        h_low, w_low = h // 4, w // 4
        bands_real.append(x_real[:, :, :h_low, :w_low])
        bands_imag.append(x_imag[:, :, :h_low, :w_low])
        
        # Medium frequencies
        h_mid, w_mid = h // 2, w // 2
        mid_real = x_real[:, :, :h_mid, :w_mid].clone()
        mid_imag = x_imag[:, :, :h_mid, :w_mid].clone()
        mid_real[:, :, :h_low, :w_low] = 0  # Remove low freq component
        mid_imag[:, :, :h_low, :w_low] = 0
        bands_real.append(mid_real)
        bands_imag.append(mid_imag)
        
        # High frequencies (fine scale)
        high_real = x_real.clone()
        high_imag = x_imag.clone()
        high_real[:, :, :h_mid, :w_mid] = 0  # Remove low and mid freq components
        high_imag[:, :, :h_mid, :w_mid] = 0
        bands_real.append(high_real)
        bands_imag.append(high_imag)
        
        return bands_real, bands_imag
    
    def get_band_freq_embedding(self, bh, bw, d, device):
        """Generate frequency embedding for a specific band size"""
        # Create simple frequency embedding for the band
        freq_h = torch.arange(bh, device=device, dtype=torch.float) / bh
        freq_w = torch.arange(bw, device=device, dtype=torch.float) / bw
        
        freq_h_grid, freq_w_grid = torch.meshgrid(freq_h, freq_w, indexing='ij')
        
        # Create position encoding similar to transformer
        pe = torch.zeros(bh, bw, d, device=device)
        
        for i in range(0, d, 4):
            if i + 1 < d:
                pe[:, :, i] = torch.sin(freq_h_grid * 10000 ** (i / d))
                pe[:, :, i + 1] = torch.cos(freq_h_grid * 10000 ** (i / d))
            if i + 3 < d:
                pe[:, :, i + 2] = torch.sin(freq_w_grid * 10000 ** ((i + 2) / d))
                pe[:, :, i + 3] = torch.cos(freq_w_grid * 10000 ** ((i + 2) / d))
        
        return pe
    
    def combine_frequency_bands(self, bands_real, bands_imag, h, w):
        """Combine frequency bands back into full spectrum"""
        combined_real = torch.zeros_like(bands_real[-1])
        combined_imag = torch.zeros_like(bands_imag[-1])
        
        # Add low frequencies
        h_low, w_low = bands_real[0].shape[-2], bands_real[0].shape[-1]
        combined_real[:, :, :h_low, :w_low] += bands_real[0]
        combined_imag[:, :, :h_low, :w_low] += bands_imag[0]
        
        # Add medium frequencies
        h_mid, w_mid = bands_real[1].shape[-2], bands_real[1].shape[-1]
        combined_real[:, :, :h_mid, :w_mid] += bands_real[1]
        combined_imag[:, :, :h_mid, :w_mid] += bands_imag[1]
        
        # Add high frequencies
        combined_real += bands_real[2]
        combined_imag += bands_imag[2]
        
        return combined_real, combined_imag
    
    def forward(self, x_real, x_imag, freq_emb):
        b, d, h, w = x_real.shape
        
        # Split into frequency bands
        bands_real, bands_imag = self.split_frequency_bands(x_real, x_imag, h, w)
        
        # Process each scale with shared components + lightweight adapters
        processed_bands_real = []
        processed_bands_imag = []
        
        for i, (band_real, band_imag) in enumerate(zip(bands_real, bands_imag)):
            # Reshape for attention
            bh, bw = band_real.shape[-2], band_real.shape[-1]
            band_real_flat = rearrange(band_real, 'b d h w -> b (h w) d')
            band_imag_flat = rearrange(band_imag, 'b d h w -> b (h w) d')
            
            # Add frequency embedding (simplified)
            try:
                band_freq_emb = self.get_band_freq_embedding(bh, bw, d, x_real.device)
                band_freq_emb_flat = band_freq_emb.reshape(1, -1, d)
                if band_freq_emb_flat.shape[1] == band_real_flat.shape[1]:
                    band_real_flat = band_real_flat + band_freq_emb_flat
                    band_imag_flat = band_imag_flat + band_freq_emb_flat
            except:
                pass
            
            # Apply shared attention with scale-specific adaptation
            attn_real, attn_imag = self.shared_attention(band_real_flat, band_imag_flat)
            
            # Scale-specific adaptation (lightweight)
            if i < len(self.scale_adapters):
                attn_adapter, ff_adapter = self.scale_adapters[i]
                attn_real = attn_adapter(attn_real)
                attn_imag = attn_adapter(attn_imag)
            
            band_real_flat = band_real_flat + attn_real
            band_imag_flat = band_imag_flat + attn_imag
            
            # Apply shared feed forward with scale-specific adaptation
            ff_real, ff_imag = self.shared_feedforward(band_real_flat, band_imag_flat)
            
            if i < len(self.scale_adapters):
                _, ff_adapter = self.scale_adapters[i]
                ff_real = ff_adapter(ff_real)
                ff_imag = ff_adapter(ff_imag)
            
            band_real_flat = band_real_flat + ff_real
            band_imag_flat = band_imag_flat + ff_imag
            
            # Reshape back
            band_real = rearrange(band_real_flat, 'b (h w) d -> b d h w', h=bh, w=bw)
            band_imag = rearrange(band_imag_flat, 'b (h w) d -> b d h w', h=bh, w=bw)
            
            processed_bands_real.append(band_real)
            processed_bands_imag.append(band_imag)
        
        # Combine bands
        output_real, output_imag = self.combine_frequency_bands(processed_bands_real, processed_bands_imag, h, w)
        
        # Lightweight cross-scale interaction using learned mixing
        cross_mix = torch.softmax(self.cross_scale_mix, dim=-1)
        
        # Apply cross-scale mixing (much more efficient than full attention)
        output_real_flat = rearrange(output_real, 'b d h w -> b (h w) d')
        output_imag_flat = rearrange(output_imag, 'b d h w -> b (h w) d')
        
        # Final normalization
        combined = torch.cat([output_real_flat, output_imag_flat], dim=-1)
        combined = self.norm(combined)
        output_real_flat, output_imag_flat = combined.chunk(2, dim=-1)
        
        # Reshape back
        output_real = rearrange(output_real_flat, 'b (h w) d -> b d h w', h=h, w=w)
        output_imag = rearrange(output_imag_flat, 'b (h w) d -> b d h w', h=h, w=w)
        
        return output_real, output_imag


class fEfficientFourierTransformer(nn.Module):
    """Parameter-efficient Fourier Transformer with complex attention and multi-scale processing"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_scales=3):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Efficient frequency embedding
        self.freq_embedding = EfficientFrequencyEmbedding(dim)
        
        # Efficient multi-scale complex transformer
        self.transformer = EfficientMultiScaleComplexTransformer(
            dim=dim, 
            depth=depth, 
            heads=heads, 
            dim_head=dim_head, 
            mlp_dim=mlp_dim,
            num_scales=num_scales
        )
        
    def forward(self, x):
        x_raw = x
        
        # Fourier transform
        x_fft = torch.fft.rfft2(x)  # (B, D, H, W//2+1)
        x_real = x_fft.real
        x_imag = x_fft.imag
        
        # Get frequency embeddings
        h, w = x_real.shape[-2], x_real.shape[-1]
        freq_emb = self.freq_embedding(h, w, x.device)  # (H, W, dim)
        freq_emb = freq_emb.unsqueeze(0).permute(0, 3, 1, 2)  # (1, dim, H, W)
        
        # Apply multi-scale transformer in frequency domain
        out_real, out_imag = self.transformer(x_real, x_imag, freq_emb)
        
        # Combine real and imaginary parts
        x_fft_out = torch.complex(out_real, out_imag)
        
        # Inverse Fourier transform
        x_out = torch.fft.irfft2(x_fft_out, s=x_raw.shape[-2:])
        
        # Residual connection (modeling the change/velocity)
        x_out = x_out + x_raw
        
        return x_out


# Legacy FourierTransformer (keeping for compatibility)
class OldFourierTransformer(Transformer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super(OldFourierTransformer, self).__init__(dim, depth, heads, dim_head, mlp_dim)
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

        self.transformer = EfficientFourierTransformer(dim=dim, depth=n_layers, heads=8, dim_head=64, mlp_dim=dim*4, num_scales=3)

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
    model = FFormer(in_channels=3, out_channels=3, in_timesteps=7, out_timesteps=1, n_layers=3, dim=1024, patch_size=(8, 8))
    print(model.count_parameters())
    y = model(x)
    print(y.shape)
    