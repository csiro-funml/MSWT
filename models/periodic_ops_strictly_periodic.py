"""
Periodic-friendly building blocks for 2D operator models.

These are intended for periodic PDE data (e.g., SW2D_PDA) where circular
boundary handling is preferred over zero/reflect padding.
"""

import math
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


def periodic_pad2d(x, pad_h, pad_w):
    """Pad bottom/right with circular wrap to reach even or patch-aligned size."""
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="circular")


class CircularConv2d(nn.Module):
    """Conv2d with explicit circular padding."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if isinstance(padding, tuple):
            self.pad_h, self.pad_w = padding
        else:
            self.pad_h = self.pad_w = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        if self.pad_h or self.pad_w:
            x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode="circular")
        return self.conv(x)


class PeriodicDWT2D(nn.Module):
    """
    Periodic discrete wavelet transform with circular padding.

    Returns (B, 4C, H/2, W/2) if format='cat', or (B, 4, C, H/2, W/2) if stack.
    """

    def __init__(self, wave="haar", format="cat"):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer("w_ll", w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_lh", w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hl", w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hh", w_hh.unsqueeze(0).unsqueeze(0))
        self.format = format

    def forward(self, x, return_shape=False):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = periodic_pad2d(x, pad_h, pad_w)

        w_ll = self.w_ll.to(dtype=x.dtype).expand(c, -1, -1, -1)
        w_lh = self.w_lh.to(dtype=x.dtype).expand(c, -1, -1, -1)
        w_hl = self.w_hl.to(dtype=x.dtype).expand(c, -1, -1, -1)
        w_hh = self.w_hh.to(dtype=x.dtype).expand(c, -1, -1, -1)

        x_ll = F.conv2d(x, w_ll, stride=2, groups=c)
        x_lh = F.conv2d(x, w_lh, stride=2, groups=c)
        x_hl = F.conv2d(x, w_hl, stride=2, groups=c)
        x_hh = F.conv2d(x, w_hh, stride=2, groups=c)

        if self.format == "cat":
            out = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        else:
            out = torch.stack([x_ll, x_lh, x_hl, x_hh], dim=1)

        if return_shape:
            return out, (h, w)
        return out


class PeriodicIDWT2D(nn.Module):
    """
    Periodic inverse DWT. Expects input in cat format (B, 4C, H, W) or
    stacked format (B, 4, C, H, W). Use target_size to crop after odd padding.
    """

    def __init__(self, wave="haar"):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
        rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        filters = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer("filters", filters)

    def forward(self, x, target_size=None):
        if x.dim() == 4:
            b, c4, h, w = x.shape
            c = c4 // 4
            x = x.view(b, 4, c, h, w).transpose(1, 2).reshape(b, -1, h, w)
        elif x.dim() == 5:
            b, four, c, h, w = x.shape
            if four != 4:
                raise ValueError(f"Expected 4 wavelet bands, got {four}")
            x = x.transpose(1, 2).reshape(b, -1, h, w)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        filters = self.filters.to(dtype=x.dtype).repeat(c, 1, 1, 1)
        out = F.conv_transpose2d(x, filters, stride=2, groups=c)

        if target_size is not None:
            target_h, target_w = target_size
            out = out[:, :, :target_h, :target_w]
        return out


def periodic_grid_2d(h, w, device=None, dtype=None):
    """Return sin/cos positional grid: (H, W, 4)."""
    gridx = torch.linspace(0, 1, h, device=device, dtype=dtype)
    gridy = torch.linspace(0, 1, w, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(gridx, gridy, indexing="ij")
    two_pi = 2.0 * math.pi
    feats = [
        torch.sin(two_pi * xx),
        torch.cos(two_pi * xx),
        torch.sin(two_pi * yy),
        torch.cos(two_pi * yy),
    ]
    return torch.stack(feats, dim=-1)


class AddPeriodicGrid(nn.Module):
    """Append periodic (sin/cos) positional grid features to a (B, H, W, C) tensor."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, h, w, _ = x.shape
        grid = periodic_grid_2d(h, w, device=x.device, dtype=x.dtype)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        return torch.cat([x, grid], dim=-1)

