#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
from einops import rearrange
import math as mt
import numpy as np
import scipy.stats as stats
import torch.nn as nn
from typing import Tuple

def get_loss_func(name, component, normalizer):
    if name == 'rel2':
        return RelLpLoss(p=2,component=component, normalizer=normalizer)
    elif name == "rel1":
        return RelLpLoss(p=1,component=component, normalizer=normalizer)
    elif name == 'l2':
        return LpLoss(p=2, component=component, normalizer=normalizer)
    elif name == "l1":
        return LpLoss(p=1, component=component, normalizer=normalizer)
    else:
        raise NotImplementedError




class RelL2Norm(_WeightedLoss):
    def __init__(self, d=2, p=2, size_average=True, reduction=True,return_comps = False):
        super(RelL2Norm, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.return_comps = return_comps



    def forward(self, pred, y):
        # x: shape (B, H, W, T, C), y: shape (B, H, W, T, C)
        B, H, W, T, C = y.shape
        # reshape the x and y to (B*T, HW, C)
        pred = rearrange(pred, 'b h w t c -> (b t) (h w) c')
        y = rearrange(y, 'b h w t c -> (b t) (h w) c')

        diff_norms = torch.sqrt(torch.sum((pred - y)**2, dim=1)) # (B*T, C)
        y_norms = torch.sqrt(torch.sum(y**2, dim=1)) # (B*T, C)
        loss = torch.mean(diff_norms/(y_norms + 1e-8)) # average over the batch size, time steps, and channels
        return loss


class RMSE(_WeightedLoss):
    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, y):
        # x: shape (B, H, W, T, C), y: shape (B, H, W, T, C)
        B, H, W, T, C = y.shape
        # reshape the x and y to (B*T, HW, C)
        rmse = torch.sqrt(self.mse(pred, y))
        return rmse



class BoundaryRMSE(_WeightedLoss):
    def __init__(self):
        super(BoundaryRMSE, self).__init__()
        self.mse = nn.MSELoss()

    def extract_boundary(self, x):
        # x (N, H, W, C),
        # return (N, 2*H+2*W, C)
        x_bound = torch.cat([x[..., 0, :, :], x[..., -1, :, :], x[..., :, 0, :], x[..., :, -1, :]], dim=-2)
        return x_bound

    def forward(self, pred, y):
        B, H, W, T, C = y.shape
        # reshape the x and y to (B*T, HW, C)
        pred = rearrange(pred, 'b h w t c -> (b t) h w c')
        y = rearrange(y, 'b h w t c -> (b t) h w c')
        
        # extrat
        pred_bound = self.extract_boundary(pred)
        y_bound = self.extract_boundary(y)
        rmse = torch.sqrt(self.mse(pred_bound, y_bound))
        return rmse


class MaxAbsError(_WeightedLoss):
    """
    Compute the max absolute error sample average
    """
    def __init__(self):
        super(MaxAbsError, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, y):
        B, H, W, T, C = y.shape
        # reshape the x and y to (B*T, HW, C)
        pred = rearrange(pred, 'b h w t c -> (b t) (h w) c')
        y = rearrange(y, 'b h w t c -> (b t) (h w) c')
        
        # compute the max error across the (H, W) then compute the sample average
        max_error = torch.max(torch.abs(pred - y), dim=1).values # (B, C)
        max_error = torch.mean(max_error) # average over the batch size and channels
        return max_error

class GlobalMaxAbsError(_WeightedLoss):
    """
    Compute the max absolute error global average
    """
    def __init__(self):
        super(GlobalMaxAbsError, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, y):
        B, H, W, T, C = y.shape
        # reshape the x and y to (B*T, HW, C)
        pred = rearrange(pred, 'b h w t c -> (b t) (h w) c')
        y = rearrange(y, 'b h w t c -> (b t) (h w) c')
        
        # compute the global max error
        max_error = torch.max(torch.abs(pred - y)) # just global max error
        return max_error

def compute_frequency_spectrum(y_pred, y):
    # y_pred: (B, H, W, T, C), y: (B, H, W, T, C)
    B, H, W, T, C = y.shape

    # Absolute error averaged over batch, time, channels -> (H, W)
    abs_error = torch.abs(y - y_pred)  # (B, H, W, T, C)
    abs_error = torch.mean(abs_error, dim=(0, -1, -2))  # (H, W) # average over batch, time, channels

    # Use full 2D FFT so the spectrum shape matches (H, W)
    abs_error_fft = torch.fft.fft2(abs_error)
    # Take magnitude and move to numpy for binning
    fourier_amplitudes = torch.abs(abs_error_fft).detach().cpu().numpy()

    # Create the k-frequency grid for rectangular image
    kfreq_x = np.fft.fftfreq(W) * W
    kfreq_y = np.fft.fftfreq(H) * H
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    # Flatten the arrays to use in binning (1D arrays of equal length)
    knrm = knrm.ravel()
    fourier_amplitudes = fourier_amplitudes.ravel()

    # Define the bins for the wavenumber - use the minimum dimension for binning
    min_dim = min(H, W)
    kbins = np.arange(0.5, min_dim // 2 + 1, 1.0)

    # Bin the data (radial mean)
    Abins, _, _ = stats.binned_statistic(
        knrm, fourier_amplitudes, statistic="mean", bins=kbins
    )

    return Abins


def compute_error_fft(model, test_loader, num_bins, device, args):
    # compute the frequency spectrum of the error on the validation set
    with torch.no_grad():
        model.eval()
        error_fft_epoch = torch.zeros(num_bins)
        for xx, yy in test_loader:
            xx = xx.to(device)  ## B, n, n, T_in, C
            yy = yy.to(device)  ## B, n, n, T_ar, C
            xx = test_loader.dataset.normalize_x(xx)
            yy_norm = test_loader.dataset.normalize_x(yy)
            for t in range(0, yy_norm.shape[-2], args.T_bundle):
                # print("t", t)
                y = yy_norm[..., t:t + args.T_bundle, :]
                pred_step = model(xx)
                break
            err_spec = compute_frequency_spectrum(pred_step, y)
            y_spec = compute_frequency_spectrum(torch.zeros_like(y), y)
            rel_error_spec = np.log(np.abs(err_spec / (y_spec + 1e-8)))
            error_fft_epoch += rel_error_spec * xx.shape[0]
        error_fft_epoch /= len(test_loader)
        return error_fft_epoch


class FourierLoss(_WeightedLoss):
    def __init__(self,  d=2, p=2, beta=1):
        super(FourierLoss, self).__init__()
        self.lp_loss = SimpleLpLoss(d=d, p=p)
        self.beta = beta
        self.mseloss = nn.MSELoss()
        
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        fft_loss = self.mseloss(pred_fft.real, target_fft.real) + self.mseloss(pred_fft.imag, target_fft.imag)
        pred_loss = self.lp_loss(pred, target)
        loss =  pred_loss + self.beta * fft_loss
        # print("fft_loss", fft_loss.item())
        # print("pred_loss", pred_loss.item())
        # print("loss", loss.item())
        return loss


class NLLLoss(_WeightedLoss):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, pred, target):
        # predict: (B, H, W, T, 2*C) with mean and std 
        # target: (B, H, W, T, C)
        pred = pred.view(pred.shape[0], -1, target.shape[-1]*2)
        target = target.view(target.shape[0], -1, target.shape[-1])
        pred_mean, pred_std = pred[:, :, :pred.shape[-1]//2], pred[:, :, pred.shape[-1]//2:]
        
        # check if pred_std is non-negative
        if pred_std.min() < 0:
            print("pred_std is negative", pred_std.min())
            pred_std = torch.abs(pred_std)
        # pred_std = torch.exp(pred_pre_std)
        # pred_std = pred_std.clamp(1e-6, 10)
        # print("pred_mean shape", pred_mean.shape, pred_mean.min(), pred_mean.max())
        # print("pred_std shape", pred_std.shape, pred_std.min(), pred_std.max())
        # create a gaussian distribution with pred_mean and pred_std
        dist = torch.distributions.Normal(pred_mean, pred_std)
        # compute the nll loss
        nll_loss = -dist.log_prob(target)
        # print("nll_loss shape", nll_loss.shape)
        nll_loss = nll_loss.sum(dim=1)/target.shape[1] # (B, C)
        # average over the batch size, height, width, and time
        return nll_loss.mean()
       

class LpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0, regularizer=False, normalizer=None):
        super(LpLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component in ['all' , 'all-reduce'] else int(component)

        self.regularizer = regularizer
        self.normalizer = normalizer

    def _lp_losses(self, pred, target):
        if self.component == 'all':
            losses = ((pred - target).view(pred.shape[0],-1,pred.shape[-1]).abs() ** self.p).mean(dim=1) ** (1 / self.p)
            metrics = losses.mean(dim=0).clone().detach().cpu().numpy()

        else:
            assert self.component <= target.shape[1]
            losses = ((pred - target).view(pred.shape[0],-1,pred.shape[-1]).abs() ** self.p).mean(dim=1) ** (1 / self.p)
            metrics = losses.mean().clone().detach().cpu().numpy()

        loss = losses.mean()

        return loss, metrics

    def forward(self, pred, target):

        #### only for computing metrics

        loss, metrics = self._lp_losses(pred, target)

        if self.normalizer is not None:
            ori_pred, ori_target = self.normalizer.transform(pred,component=self.component, inverse=True), self.normalizer.transform(target, inverse=True)
            _, metrics = self._lp_losses(ori_pred, ori_target)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)

        return loss, reg, metrics

class RelLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0, regularizer=False, normalizer=None):
        super(RelLpLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component in ['all' , 'all-reduce'] else int(component)
        self.regularizer = regularizer
        self.normalizer = normalizer

    ### all reduce is used in temporal cases, use only one metric for all components
    def _lp_losses(self, pred, target):
        if (self.component == 'all') or (self.component == 'all-reduce'):
            err_pool = ((pred - target).view(pred.shape[0], -1, pred.shape[-1]).abs()**self.p).sum(dim=1,keepdim=False)
            target_pool = (target.view(target.shape[0], -1, target.shape[-1]).abs()**self.p).sum(dim=1,keepdim=False)
            losses = (err_pool / target_pool)**(1/ self.p)
            if self.component == 'all':
                # metrics = losses.mean(dim=0).clone().detach().cpu().numpy()
                metrics = losses.mean(dim=0).unsqueeze(0).clone().cpu().detach().numpy()  # 1, n

            else:
                # metrics = losses.mean().clone().detach().cpu().numpy()
                metrics = losses.mean().unsqueeze(0).clone().cpu().detach().numpy()   # 1, 1

        else:
            assert self.component <= target.shape[1]
            err_pool = ((pred - target[...,self.component]).view(pred.shape[0], -1, pred.shape[-1]).abs() ** self.p).sum(dim=1,keepdim=False)
            target_pool = (target.view(target.shape[0], -1, target.shape[-1])[...,self.component].abs() ** self.p).sum(dim=1, keepdim=False)
            losses = (err_pool / target_pool)**(1/ self.p)
            # metrics = losses.mean().clone().detach().cpu().numpy()
            metrics = losses.mean().unsqueeze(0).clone().cpu().detach().numpy()


        loss = losses.mean()

        return loss, metrics


    ### pred, target: B, N1, N2..., Nm, C-> B, C
    def forward(self, pred, target):
        loss, metrics = self._lp_losses(pred, target)

        ### only for computing metrics
        if self.normalizer is not None:
            ori_pred, ori_target = self.normalizer.transform(pred,component=self.component,inverse=True), self.normalizer.transform(target, inverse=True)
            _ , metrics = self._lp_losses(ori_pred, ori_target)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)


        return loss, reg, metrics


class RFNELoss(_WeightedLoss):
    '''
    RFNE(y, y_hat) = Frobenius_norm(y-y_hat) / Frobenius_norm(y)
    y: target, (batch, nx^i..., timesteps, nc)
    y_hat: prediction, (batch, nx^i..., timesteps, nc)
    '''
    def forward(self, pred, target):
        dims = target.size()
        error_norm = torch.norm(pred - target, dim=dims[1:-2])
        target_norm = torch.norm(target, dim=dims[1:-2])
        return torch.mean(error_norm / target_norm)




class Evaluator(_WeightedLoss):
    def __init__(self, temporal=False, griddata=False, component=0,  normalizer=None, ilow=4, ihigh=12):
        super(Evaluator, self).__init__()


        self.component = component if component in ['all', 'all-reduce'] else int(component)
        self.normalizer = normalizer
        self.temporal = temporal
        self.griddata = griddata
        self.ilow = ilow
        self.ihigh = ihigh





    ### pred, target: B, N1, N2..., Nm, C-> B, C
    def forward(self, pred, target):


        with torch.no_grad():
            ### only for computing metrics
            if self.normalizer is not None:
                pred, target = self.normalizer.transform(pred, component=self.component,inverse=True), self.normalizer.transform(target, inverse=True)

            if self.component not in ['all', 'all-reduce']:
                target = target[..., self.component]
                pred, target = pred.unsqueeze(-1), target.unsqueeze(-1)

            metrics = {}
            ## 1, C
            _pred, _target = pred.view(pred.shape[0], -1, pred.shape[-1]), target.view(target.shape[0], -1, target.shape[-1])
            nmae = ((_pred - _target).abs().sum(dim=1, keepdim=False) / (_target.abs().sum(dim=1, keepdim=False))).mean(dim=0, keepdim=True)
            nmse = torch.sqrt(((_pred - _target) ** 2).sum(dim=1, keepdim=False) / ((_target) ** 2).sum(dim=1, keepdim=False)).mean(dim=0, keepdim=True)
            nmxe = (torch.amax((_pred - _target).abs(), dim=1, keepdim=False) / torch.amax(_target.abs(), dim=1, keepdim=False)).mean(dim=0, keepdim=True)

            metrics.update({'nmae': nmae, 'nmse': nmse, 'nmxe': nmxe})
            if self.temporal:
                _pred, _target = pred.view(pred.shape[0], -1, pred.shape[-2], pred.shape[-1]), target.view(target.shape[0], -1, target.shape[-2], target.shape[-1])

                nmae_t = ((_pred - _target).abs().sum(dim=1, keepdim=False) / (_target.abs().sum(dim=1, keepdim=False))).mean(dim=0, keepdim=True)
                nmse_t = torch.sqrt(((_pred - _target) ** 2).sum(dim=1, keepdim=False) / (_target ** 2).sum(dim=1, keepdim=False)).mean(dim=0, keepdim=True)
                nmxe_t = (torch.amax((_pred - _target).abs(), dim=1, keepdim=False) / torch.amax(_target.abs(), dim=1, keepdim=False)).mean(dim=0, keepdim=True)

                metrics.update({'nmae_t': nmae_t, 'nmse_t': nmse_t, 'nmxe_t': nmxe_t})
            if self.griddata:
                bdmse, fmse_low, fmse_mid, fmse_high = compute_fourier_error(pred, target, self.ilow, self.ihigh)
                metrics.update({'bdmse': bdmse, 'fmse_low': fmse_low, 'fmse_mid': fmse_mid, 'fmse_high': fmse_high})

            metrics = {key: value.cpu().numpy() for key, value in metrics.items()}
        return metrics



# adapted from Galerkin Transformer
def central_diff(x: torch.Tensor):
    # assuming PBC
    # x: (batch, seq_len, n), h is the step size, assuming n = h*w
    B, HW, T, C = x.shape
    res = int(mt.sqrt(HW))
    H = W= res
    x = rearrange(x, 'b (h w) t c -> b t c h w', h=res, w=res)
    # print("x shape", x.shape)
    h = 1./res
    x = x.view(B * T, C, H, W)     # reshape to [16, 3, 128, 128]
    x = F.pad(x, (1, 1, 1, 1), mode='circular')  #  [b t*c h+2 w+2] pad height and width by 1,  pad for the last two dimensions
    x = x.view(B, T, C, H + 2, W + 2)  # reshape back to [16, 1, 3, 130, 130]
    
    grad_x = (x[..., 1:-1, 2:] - x[..., 1:-1, :-2]) / (2*h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[..., 2:, 1:-1] - x[..., :-2, 1:-1]) / (2*h)  # f(x+h) - f(x-h) / 2h
    
    grad_x = rearrange(grad_x, 'b t c h w -> b t h w c')
    grad_y = rearrange(grad_y, 'b t c h w -> b t h w c')
    return grad_x, grad_y


def compute_fourier_error(pred, target, iLow=4, iHigh=12, if_mean=False):
    # (batch, nx^i..., timesteps, nc)
    idxs = target.size()
    if len(idxs) == 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    if len(idxs) == 5:
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)
    idxs = target.size()
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]

    # RMSE
    err_mean = torch.sqrt(torch.mean((pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])) ** 2, dim=2))
    err_RMSE = torch.mean(err_mean, axis=0)
    # print("err RMSE", err_RMSE.squeeze().detach().cpu().numpy())
    nrm = torch.sqrt(torch.mean(target.view([nb, nc, -1, nt]) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)
    # print("err nRMSE", err_nRMSE.squeeze().detach().cpu().numpy())

    err_CSV = torch.sqrt(torch.mean(
        (torch.sum(pred.view([nb, nc, -1, nt]), dim=2) - torch.sum(target.view([nb, nc, -1, nt]), dim=2)) ** 2,
        dim=0))
    if len(idxs) == 4:
        nx = idxs[2]
        err_CSV /= nx
    elif len(idxs) == 5:
        nx, ny = idxs[2:4]
        err_CSV /= nx * ny
    elif len(idxs) == 6:
        nx, ny, nz = idxs[2:5]
        err_CSV /= nx * ny * nz
    # worst case in all the data
    err_Max = torch.max(torch.max(
        torch.abs(pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])), dim=2)[0], dim=0)[0]
    # print("error max", err_Max.squeeze().detach().cpu().numpy())
    
    if len(idxs) == 4:  # 1D
        err_BD = (pred[:, :, 0, :] - target[:, :, 0, :]) ** 2
        err_BD += (pred[:, :, -1, :] - target[:, :, -1, :]) ** 2
        err_BD = torch.mean(torch.sqrt(err_BD / 2.), dim=0)
    elif len(idxs) == 5:  # 2D
        nx, ny = idxs[2:4]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD = (torch.sum(err_BD_x, dim=-2) + torch.sum(err_BD_y, dim=-2)) / (2 * nx + 2 * ny)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)
    elif len(idxs) == 6:  # 3D
        nx, ny, nz = idxs[2:5]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD_z = (pred[:, :, :, :, 0] - target[:, :, :, :, 0]) ** 2
        err_BD_z += (pred[:, :, :, :, -1] - target[:, :, :, :, -1]) ** 2
        err_BD = torch.sum(err_BD_x.view([nb, -1, nt]), dim=-2) \
                 + torch.sum(err_BD_y.view([nb, -1, nt]), dim=-2) \
                 + torch.sum(err_BD_z.view([nb, -1, nt]), dim=-2)
        err_BD = err_BD / (2 * nx * ny + 2 * ny * nz + 2 * nz * nx)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)

    if len(idxs) == 4:  # 1D
        nx = idxs[2]
        pred_F = torch.fft.rfft(pred, dim=2)
        target_F = torch.fft.rfft(target, dim=2)
        _err_F = torch.sqrt(torch.mean(torch.abs(pred_F - target_F) ** 2, axis=0)) / nx   # Lx, Ly, Lz=1
    if len(idxs) == 5:  # 2D
        pred_F = torch.fft.fftn(pred, dim=[2, 3])
        target_F = torch.fft.fftn(target, dim=[2, 3])
        nx, ny = idxs[2:4]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt]).to(pred.device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                it = mt.floor(mt.sqrt(i ** 2 + j ** 2))
                if it > min(nx // 2, ny // 2) - 1:
                    continue
                err_F[:, :, it] += _err_F[:, :, i, j]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny)
    elif len(idxs) == 6:  # 3D
        pred_F = torch.fft.fftn(pred, dim=[2, 3, 4])
        target_F = torch.fft.fftn(target, dim=[2, 3, 4])
        nx, ny, nz = idxs[2:5]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2, nz // 2), nt]).to(pred.device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                for k in range(nz // 2):
                    it = mt.floor(mt.sqrt(i ** 2 + j ** 2 + k ** 2))
                    if it > min(nx // 2, ny // 2, nz // 2) - 1:
                        continue
                    err_F[:, :, it] += _err_F[:, :, i, j, k]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny * nz)

    fmse_low = torch.mean(_err_F[:, :iLow], dim=1).T  # low freq
    fmse_mid = torch.mean(_err_F[:, iLow:iHigh], dim=1).T  # middle freq
    fmse_high = torch.mean(_err_F[:, iHigh:], dim=1).T

    # err_F = torch.zeros([nc, 3, nt]).to(pred.device)
    # err_F[:, 0] += torch.mean(_err_F[:, :iLow], dim=1)  # low freq
    # err_F[:, 1] += torch.mean(_err_F[:, iLow:iHigh], dim=1)  # middle freq
    # err_F[:, 2] += torch.mean(_err_F[:, iHigh:], dim=1)  # high freq

    # if if_mean:
    #     return torch.mean(err_RMSE, dim=[0, -1]), \
    #            torch.mean(err_nRMSE, dim=[0, -1]), \
    #            torch.mean(err_CSV, dim=[0, -1]), \
    #            torch.mean(err_Max, dim=[0, -1]), \
    #            torch.mean(err_BD, dim=[0, -1]), \
    #            torch.mean(err_F, dim=[0, -1])
    # else:
    #     return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F
    return err_BD, fmse_low, fmse_mid, fmse_high    ## T, C, ### T, C


# USED FOR testing evaluation
def get_frequency_bands_from_cumulative_energy(
    y: torch.Tensor,
    low_percentile: float = 0.33,
    high_percentile: float = 0.67,
    max_freq: int = None,
    eps: float = 1e-12,
) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
    """
    Determine frequency band boundaries using cumulative energy distribution.
    Returns discrete frequency bin indices for low/mid/high frequency aggregation.

    Args:
        y: Tensor of shape (N, H, W, C) - input field data.
        low_percentile: Cumulative energy fraction for low/mid boundary (default 0.33).
        high_percentile: Cumulative energy fraction for mid/high boundary (default 0.67).
        max_freq: Maximum frequency to consider (default = min(H, W)//2).
        eps: Small number to avoid divide-by-zero.

    Returns:
        k_low: int, frequency bin where cumulative energy >= low_percentile.
        k_high: int, frequency bin where cumulative energy >= high_percentile.
        freq_bins: tensor of frequency bin centers [0, 1, 2, ..., max_freq].
        cumulative_energy: tensor of cumulative energy fractions.
    """
    assert y.ndim == 4, "y must have shape (N,H,W,C)"
    N, H, W, C = y.shape
    if max_freq is None:
        max_freq = min(H, W) // 2

    device = y.device
    dtype = y.dtype

    # Reorder to (N, C, H, W) and remove spatial mean
    y_nc_hw = y.permute(0, 3, 1, 2).contiguous()
    y_demean = y_nc_hw - y_nc_hw.mean(dim=(-2, -1), keepdim=True)

    # 2D FFT over spatial dims
    y_hat = torch.fft.fftn(y_demean, dim=(-2, -1))
    # Sum power over batch and channels -> (H, W)
    power = (y_hat.real**2 + y_hat.imag**2).sum(dim=(0, 1))

    # Build discrete frequency grid using integer bins
    # FFT frequencies are organized as [0, 1, 2, ..., N//2, -N//2+1, ..., -1]
    # We want the magnitude |k| = sqrt(kx^2 + ky^2)
    
    freq_x = torch.arange(H, device=device, dtype=dtype)
    freq_y = torch.arange(W, device=device, dtype=dtype)
    
    # Convert to centered frequencies: [0, 1, ..., N//2, -(N//2-1), ..., -1]
    freq_x = torch.where(freq_x <= H//2, freq_x, freq_x - H)
    freq_y = torch.where(freq_y <= W//2, freq_y, freq_y - W)
    
    Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
    Fmag = torch.sqrt(Fx**2 + Fy**2)  # Radial frequency magnitude

    # Create energy bins for each integer frequency from 0 to max_freq
    freq_bins = torch.arange(0, max_freq + 1, device=device, dtype=dtype)
    E_bins = torch.zeros(max_freq + 1, device=device, dtype=dtype)
    
    # Aggregate power into radial frequency bins
    for k in range(max_freq + 1):
        if k == max_freq:
            # Last bin: include all frequencies >= max_freq
            mask = Fmag >= k
        else:
            # Regular bin: [k, k+1)
            mask = (Fmag >= k) & (Fmag < k + 1)
        E_bins[k] = power[mask].sum()

    # Compute cumulative energy fraction
    total_energy = E_bins.sum() + eps
    cumulative_energy = torch.cumsum(E_bins, dim=0) / total_energy

    # Find frequency bins at specified percentiles
    def find_freq_at_percentile(percentile: float) -> int:
        # Find first bin where cumulative energy >= percentile
        mask = cumulative_energy >= percentile
        if mask.any():
            return int(torch.argmax(mask.float()))
        else:
            return max_freq  # fallback

    k_low = find_freq_at_percentile(low_percentile)
    k_high = find_freq_at_percentile(high_percentile)

    return k_low, k_high, freq_bins.detach(), cumulative_energy.detach()


def aggregate_spectral_energy_by_bands(
    pred: torch.Tensor,
    target: torch.Tensor,
    k_low: int,
    k_high: int,
    max_freq: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregate spectral energy into low/mid/high frequency bands.
    
    Args:
        pred: Predicted field (N, H, W, C).
        target: Target field (N, H, W, C).
        k_low: Frequency bin boundary between low and mid bands.
        k_high: Frequency bin boundary between mid and high bands.
        max_freq: Maximum frequency to consider (default = min(H, W)//2).
    
    Returns:
        low_band_error: MSE in low frequency band [0, k_low).
        mid_band_error: MSE in mid frequency band [k_low, k_high).
        high_band_error: MSE in high frequency band [k_high, max_freq].
    """
    assert pred.shape == target.shape, "pred and target must have same shape"
    assert pred.ndim == 4, "Input must have shape (N,H,W,C)"
    
    N, H, W, C = pred.shape
    if max_freq is None:
        max_freq = min(H, W) // 2
        
    device = pred.device
    
    # Reorder to (N, C, H, W)
    pred_nc_hw = pred.permute(0, 3, 1, 2).contiguous()
    target_nc_hw = target.permute(0, 3, 1, 2).contiguous()
    
    # Remove spatial mean
    pred_demean = pred_nc_hw - pred_nc_hw.mean(dim=(-2, -1), keepdim=True)
    target_demean = target_nc_hw - target_nc_hw.mean(dim=(-2, -1), keepdim=True)
    
    # 2D FFT
    pred_fft = torch.fft.fftn(pred_demean, dim=(-2, -1))
    target_fft = torch.fft.fftn(target_demean, dim=(-2, -1))
    
    # Compute error in frequency domain
    error_fft = pred_fft - target_fft
    error_power = error_fft.real**2 + error_fft.imag**2  # (N, C, H, W)
    
    # Build frequency magnitude grid
    freq_x = torch.arange(H, device=device, dtype=torch.float32)
    freq_y = torch.arange(W, device=device, dtype=torch.float32)
    freq_x = torch.where(freq_x <= H//2, freq_x, freq_x - H)
    freq_y = torch.where(freq_y <= W//2, freq_y, freq_y - W)
    Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
    Fmag = torch.sqrt(Fx**2 + Fy**2)
    
    # Create masks for frequency bands
    low_mask = Fmag < k_low
    mid_mask = (Fmag >= k_low) & (Fmag < k_high) 
    high_mask = Fmag >= k_high
    
    # Aggregate errors by frequency bands
    low_band_error = error_power[:, :, low_mask].mean() if low_mask.any() else torch.tensor(0.0, device=device)
    mid_band_error = error_power[:, :, mid_mask].mean() if mid_mask.any() else torch.tensor(0.0, device=device)
    high_band_error = error_power[:, :, high_mask].mean() if high_mask.any() else torch.tensor(0.0, device=device)
    
    return low_band_error, mid_band_error, high_band_error


if __name__ == "__main__":
    x = torch.randn([2, 128, 128, 1, 3])
    y = torch.randn([2, 128, 128, 1, 3])

    # evaluator = Evaluator(temporal=True, griddata=True, component='all', normalizer=None)
    # metrics = evaluator(x, y)
    # print(metrics)
    # myloss = FourierLoss(beta=1)
    # Test the new cumulative energy-based frequency band selection
    print("Testing cumulative energy-based frequency aggregation...")
    
    # Create test data (N, H, W, C)
    torch.manual_seed(42)
    N, H, W, C = 2, 128, 128, 3
    target = torch.randn([N, H, W, C])
    pred = target + 0.1 * torch.randn_like(target)  # Add some noise
    
    print(f"Data shape: {target.shape}")
    print(f"Max frequency: {min(H, W) // 2}")
    
    # Get frequency bands using cumulative energy
    k_low, k_high, freq_bins, cumulative_energy = get_frequency_bands_from_cumulative_energy(
        target, low_percentile=0.33, high_percentile=0.67
    )
    
    print(f"\nFrequency band boundaries:")
    print(f"k_low (33rd percentile): {k_low}")
    print(f"k_high (67th percentile): {k_high}")
    print(f"Frequency bins range: [0, {len(freq_bins)-1}]")
    
    # Show cumulative energy distribution
    print(f"\nCumulative energy at key points:")
    print(f"At k={k_low}: {cumulative_energy[k_low]:.3f}")
    print(f"At k={k_high}: {cumulative_energy[k_high]:.3f}")
    
    # Aggregate spectral errors by frequency bands
    low_err, mid_err, high_err = aggregate_spectral_energy_by_bands(
        pred, target, k_low, k_high
    )
    
    print(f"\nSpectral error by frequency bands:")
    print(f"Low frequency error (0 to {k_low}): {low_err:.6f}")
    print(f"Mid frequency error ({k_low} to {k_high}): {mid_err:.6f}")
    print(f"High frequency error ({k_high}+): {high_err:.6f}")
    
    print("Test completed successfully!")
    # for key, value in metrics.items():
    #     print(key, value.shape)
