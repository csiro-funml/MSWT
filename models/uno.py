### Mostly from hhttps://github.com/ashiq24/UNO/blob/main/navier_stokes_uno2d.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


import operator
from functools import reduce
from functools import partial

from timeit import default_timer



class SpectralConv2d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2, modes1=None, modes2=None):
        super(SpectralConv2d_Uno, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        dim2 = Default output grid size along y ( or 2nd dimension of output domain)
        Ratio of grid size of the input and the output implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2 = Number of fourier modes to consider for the ontegral operator
                        Number of modes must be compatibale with the input grid size 
                        and desired output grid size.
                        i.e., modes1 <= min( dim1/2, input_dim1/2). 
                        Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
                        Other modes also the have same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension
        """

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1
            self.modes2 = modes2
        else:
            self.modes1 = dim1 // 2 - 1
            self.modes2 = dim2 // 2
        self.scale = (1 / (2 * in_codim)) ** (1.0 / 2.0)
        self.weights1 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
                )
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
                )
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1=None, dim2=None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm="forward")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            self.dim1,
            self.dim2 // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2), norm="forward")
        return x

class pointwise_op_2D(nn.Module):
    """
    dim1 = Default output grid size along x (or 1st dimension)
    dim2 = Default output grid size along y ( or 2nd dimension)
    in_codim = Input co-domian dimension
    out_codim = output co-domain dimension
    """

    def __init__(self, in_codim, out_codim, dim1, dim2):
        super(pointwise_op_2D, self).__init__()
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self, x, dim1=None, dim2=None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)

        # ft = torch.fft.rfft2(x_out)
        # ft_u = torch.zeros_like(ft)
        # ft_u[:dim1//2-1,:dim2//2-1] = ft[:dim1//2-1,:dim2//2-1]
        # ft_u[-(dim1//2-1):,:dim2//2-1] = ft[-(dim1//2-1):,:dim2//2-1]
        # x_out = torch.fft.irfft2(ft_u)

        x_out = torch.nn.functional.interpolate(
            x_out, size=(dim1, dim2), mode="bicubic", align_corners=True, antialias=True
        )
        return x_out


class OperatorBlock_2D(nn.Module):
    """
    Normalize = if true performs InstanceNorm2d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv2d_Uno class.
    """

    def __init__(
        self,
        in_codim,
        out_codim,
        dim1,
        dim2,
        modes1,
        modes2,
        Normalize=False,
        Non_Lin=True,
    ):
        super(OperatorBlock_2D, self).__init__()
        self.conv = SpectralConv2d_Uno(in_codim, out_codim, dim1, dim2, modes1, modes2)
        self.w = pointwise_op_2D(in_codim, out_codim, dim1, dim2)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim), affine=True)

    def forward(self, x, dim1=None, dim2=None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        x1_out = self.conv(x, dim1, dim2)
        x2_out = self.w(x, dim1, dim2)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


#####
# UNO model
# it has less aggressive scaling factors for domains and co-domains.
# ####
class UNO(nn.Module):
    def __init__(self, width, n_channels=1, in_timesteps =10, out_timesteps=1, pad=0, factor=3 / 4):
        super(UNO, self).__init__()
       
        self.in_width = 4 + in_timesteps * n_channels # input channel
        self.width = width
        self.factor = factor
        self.padding = pad
        self.in_timesteps = in_timesteps
        self.fc = nn.Linear(self.in_width, self.width // 2)

        self.fc0 = nn.Linear(
            self.width // 2, self.width
        )  # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2 * factor * self.width, 48, 48, 22, 22)

        self.L1 = OperatorBlock_2D(
            2 * factor * self.width, 4 * factor * self.width, 32, 32, 14, 14
        )

        self.L2 = OperatorBlock_2D(
            4 * factor * self.width, 8 * factor * self.width, 16, 16, 6, 6
        )

        self.L3 = OperatorBlock_2D(
            8 * factor * self.width, 8 * factor * self.width, 16, 16, 6, 6
        )

        self.L4 = OperatorBlock_2D(
            8 * factor * self.width, 4 * factor * self.width, 32, 32, 6, 6
        )

        self.L5 = OperatorBlock_2D(
            8 * factor * self.width, 2 * factor * self.width, 48, 48, 14, 14
        )

        self.L6 = OperatorBlock_2D(
            4 * factor * self.width, self.width, 64, 64, 22, 22
        )  # will be reshaped

        self.fc1 = nn.Linear(2 * self.width, 4 * self.width)
        self.fc2 = nn.Linear(4 * self.width, n_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = x.view(*x.shape[:-2], -1)           #### B, X, Y, T*C
        x = torch.cat((x, grid), dim=-1)

        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)

        x_fc0 = x_fc0.permute(0, 3, 1, 2)

        x_fc0 = F.pad(x_fc0, [self.padding, self.padding, self.padding, self.padding])

        D1, D2 = x_fc0.shape[-2], x_fc0.shape[-1]

        x_c0 = self.L0(x_fc0, int(D1 * self.factor), int(D2 * self.factor))
        x_c1 = self.L1(x_c0, D1 // 2, D2 // 2)

        x_c2 = self.L2(x_c1, D1 // 4, D2 // 4)
        x_c3 = self.L3(x_c2, D1 // 4, D2 // 4)
        x_c4 = self.L4(x_c3, D1 // 2, D2 // 2)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)
        x_c5 = self.L5(x_c4, int(D1 * self.factor), int(D2 * self.factor))
        x_c5 = torch.cat([x_c5, x_c0], dim=1)
        x_c6 = self.L6(x_c5, D1, D2)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding != 0:
            x_c6 = x_c6[..., : -self.padding, : -self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)

        x_out = self.fc2(x_fc1)

        return x_out.unsqueeze(-2) # add time dimension

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2 * np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 2 * np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat(
            (torch.sin(gridx), torch.sin(gridy), torch.cos(gridx), torch.cos(gridy)),
            dim=-1,
        ).to(device)  # (bs, H, W, 4)
        return grid


if __name__== '__main__':
    x = torch.rand(4, 128, 128, 10, 3)
    # x = torch.rand(4, 96, 192, 10, 3)
    model = UNO(32, x.shape[-1])
    y = model(x)
    print(y.shape)
