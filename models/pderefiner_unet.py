# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
This code is developed with reference to the following GitHub repo: PDERefiner: https://github.com/pdearena/pdearena/blob/main/scripts/pderefiner_train.py
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}

# Largely based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License

import math
from abc import abstractmethod

import torch
from torch import nn


def fourier_embedding(timesteps: torch.Tensor, dim, max_period=10000):
    r"""Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.
    Returns:
        embedding (torch.Tensor): [N $\times$ dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(nn.Module):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.cond_emb = nn.Linear(cond_channels, out_channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        embed_out= self.cond_emb(emb)[:, :, None, None]
        h = h + embed_out
        # Second convolution layer
        h = self.conv2(self.activation(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Attention block This is similar to [transformer multi-head
    attention]

    Args:
        n_channels (int): the number of channels in the input
        n_heads (int): the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups (int): the number of groups for [group normalization][torch.nn.GroupNorm].

    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 1):
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res



class DownBlock(nn.Module):
    """Down block This combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the first half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, cond_channels, activation=activation, norm=norm)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Up block that combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (str): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, cond_channels, activation=activation, norm=norm)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x



class MiddleBlock(nn.Module):
    """Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """

    def __init__(self, n_channels: int, cond_channels: int, has_attn: bool = False, activation: str = "gelu", norm: bool = False):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, cond_channels, activation=activation, norm=norm)
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(n_channels, n_channels, cond_channels, activation=activation, norm=norm)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x


class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)



class SimpleConvNet(nn.Module):
    """Simple ConvNet that processes (B, T, C, H, W) -> (B, D)
    
    First aggregates T and C channels, then applies convolution layers with skip connections,
    finally flattens H and W dimensions and aggregates them to (B, D).
    
    Args:
        input_time_steps (int): Number of time steps (T)
        input_channels (int): Number of channels per time step (C)
        output_dim (int): Output embedding dimension (D)
        hidden_channels (int): Number of hidden channels in conv layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
    """
    
    def __init__(
        self,
        input_time_steps: int = 7,
        input_channels: int = 3,
        output_dim: int = 256,
        hidden_channels: int = 64,
        activation: str = "gelu",
        norm: bool = True,
    ):
        super().__init__()
        
        self.input_time_steps = input_time_steps
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        
        # Aggregate T and C channels
        aggregated_channels = input_time_steps * input_channels
        
        # Initial convolution to process aggregated channels
        self.initial_conv = nn.Conv2d(aggregated_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Encoder part with skip connections (similar to UNet)
        self.encoder_conv1 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1)
        
        # Downsample layers
        self.downsample1 = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, stride=2, padding=1)
        
        # Decoder part with skip connections
        self.upsample1 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv1 = nn.Conv2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1)  # *4 due to concat
        
        self.upsample2 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv2 = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1)  # *2 due to concat
        
        # Final convolution
        self.final_conv = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        
        # Normalization layers
        if norm:
            self.norm1 = nn.GroupNorm(8, hidden_channels)
            self.norm2 = nn.GroupNorm(8, hidden_channels * 2)
            self.norm3 = nn.GroupNorm(8, hidden_channels * 4)
            self.norm4 = nn.GroupNorm(8, hidden_channels * 2)
            self.norm5 = nn.GroupNorm(8, hidden_channels)
            self.norm6 = nn.GroupNorm(8, hidden_channels // 2)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()
            self.norm4 = nn.Identity()
            self.norm5 = nn.Identity()
            self.norm6 = nn.Identity()
        
        # Global average pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.final_projection = nn.Linear(hidden_channels // 2, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
            
        Returns:
            Output tensor of shape (B, D)
        """
        B, T, C, H, W = x.shape
        
        # Aggregate T and C channels: (B, T, C, H, W) -> (B, T*C, H, W)
        x = x.reshape(B, T * C, H, W)
        
        # Initial convolution
        x = self.activation(self.norm1(self.initial_conv(x)))
        skip1 = x  # Store for skip connection
        
        # Encoder with downsampling and skip connections
        x = self.activation(self.norm2(self.encoder_conv1(x)))
        x = self.downsample1(x)
        skip2 = x  # Store for skip connection
        
        x = self.activation(self.norm3(self.encoder_conv2(x)))
        x = self.downsample2(x)
        
        # Decoder with upsampling and skip connections
        x = self.upsample1(x)
        x = torch.cat([x, skip2], dim=1)  # Skip connection
        x = self.activation(self.norm4(self.decoder_conv1(x)))
        
        x = self.upsample2(x)
        x = torch.cat([x, skip1], dim=1)  # Skip connection
        x = self.activation(self.norm5(self.decoder_conv2(x)))
        
        # Final convolution
        x = self.activation(self.norm6(self.final_conv(x)))
        
        # Global average pooling to aggregate H and W dimensions: (B, C, H, W) -> (B, C, 1, 1)
        x = self.global_pool(x)
        
        # Flatten and project to output dimension: (B, C, 1, 1) -> (B, C) -> (B, D, 1, 1)
        x = x.flatten(1)
        x = self.final_projection(x)
        
        return x


class Unet(nn.Module):
    """Modern U-Net architecture

    This is a modern U-Net architecture with wide-residual blocks and spatial attention blocks

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input
        time_future (int): Number of time steps in the output
        hidden_channels (int): Number of channels in the hidden layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
        ch_mults (list): List of channel multipliers for each resolution
        is_attn (list): List of booleans indicating whether to use attention blocks
        mid_attn (bool): Whether to use attention block in the middle block
        n_blocks (int): Number of residual blocks in each resolution
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layers
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        activation: str,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use1x1: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels

        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = time_history * self.input_channels + time_future * self.input_channels
        n_channels = hidden_channels
        # Project image into feature map
        if use1x1:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        time_embed_dim = hidden_channels * 4
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        time_embed_dim,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, time_embed_dim, has_attn=mid_attn, activation=activation, norm=norm)
        
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, time_embed_dim),
            self.activation,
            nn.Linear(time_embed_dim, time_embed_dim),
        )


        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        time_embed_dim,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, time_embed_dim, has_attn=is_attn[i], activation=activation, norm=norm))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = time_future * self.output_channels
        #
        if use1x1:
            self.final = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor,  time: torch.Tensor = None, z: torch.Tensor = None):
        
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
    
        emb = 0
        if time is not None:
            emb = emb + self.time_embed(fourier_embedding(time, self.hidden_channels))
            self.param_use_time = True
        else:
            assert not self.param_use_time, "Cannot pass time=None after using it in a previous forward pass"

        x = self.image_proj(x)
    
        h = [x]
        for m in self.down:
            if isinstance(m, Downsample):
                x = m(x)
            else:
                x = m(x, emb)
            h.append(x)

        x = self.middle(x, emb)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)        #
                x = m(x, emb)

        x = self.final(self.activation(self.norm(x)))
        x = x.reshape(
            orig_shape[0], -1, self.input_channels, *orig_shape[3:]
        )
        return x


if __name__ == "__main__":
    # Test Unet with SimpleConvNet
    print("Testing Unet with SimpleConvNet...")
    model = Unet(
        input_channels=3,
        time_history=7,
        time_future=1,
        hidden_channels=64,
        activation='gelu',
        norm=True,
    )
    print(model)
    
    u_t = torch.randn(4, 1, 3, 64, 64)
    u_prev = torch.randn(4, 7, 3, 64, 64)
    time = torch.randn(4)
    y = model(u_prev, time=time, z=u_t)
    print(f"Unet output shape: {y.shape}")
    print("Unet test passed!")