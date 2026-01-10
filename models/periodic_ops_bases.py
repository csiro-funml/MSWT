"""
Wavelet base variants for periodic ops with size-preserving padding.
"""

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


def _even_downsample_pad(filter_len):
    if (filter_len - 2) % 2 != 0:
        raise ValueError(f"Unsupported wavelet filter length {filter_len}")
    return (filter_len - 2) // 2


class PeriodicDWT2DBase(nn.Module):
    def __init__(self, wave, format="cat"):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)

        self.pad = _even_downsample_pad(len(dec_lo))

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
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        if self.pad:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")

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


class PeriodicIDWT2DBase(nn.Module):
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
        rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)

        self.pad = _even_downsample_pad(len(rec_lo))

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
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
            start_h = self.pad
            start_w = self.pad
            out = out[:, :, start_h:start_h + target_h, start_w:start_w + target_w]
        return out


class PeriodicDWT2D_DB2(PeriodicDWT2DBase):
    def __init__(self, format="cat"):
        super().__init__(wave="db2", format=format)


class PeriodicIDWT2D_DB2(PeriodicIDWT2DBase):
    def __init__(self):
        super().__init__(wave="db2")


class PeriodicDWT2D_DB4(PeriodicDWT2DBase):
    def __init__(self, format="cat"):
        super().__init__(wave="db4", format=format)


class PeriodicIDWT2D_DB4(PeriodicIDWT2DBase):
    def __init__(self):
        super().__init__(wave="db4")


class PeriodicDWT2D_SYM4(PeriodicDWT2DBase):
    def __init__(self, format="cat"):
        super().__init__(wave="sym4", format=format)


class PeriodicIDWT2D_SYM4(PeriodicIDWT2DBase):
    def __init__(self):
        super().__init__(wave="sym4")
