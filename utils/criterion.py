#!/usr/bin/env python  
#-*- coding:utf-8 _*-
from errno import EEXIST
import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
from einops import rearrange
import math as mt
import numpy as np
import scipy.stats as stats
import torch.nn as nn
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
import os
from utils.compute_physical_statistics import compute_spectra
from utils.compute_diagnostics import streamfunction_to_velocity

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
        B, C = y.shape[0], y.shape[-1]
        # reshape the x and y to (B*T, HW, C)
        # pred = rearrange(pred, 'b h w t c -> (b t) (h w) c')
        # y = rearrange(y, 'b h w t c -> (b t) (h w) c')
        pred = pred.reshape(B, -1, C)
        y = y.reshape(B, -1, C)

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



class SpectralError(_WeightedLoss):
    def __init__(self, model_name, save_path, low_percentile=0.80, high_percentile=0.99, method='radial'):
        super(SpectralError, self).__init__()
        self.mse = nn.MSELoss()
        self.k_low=None
        self.k_high=None
        self.model_name = model_name
        self.save_path = save_path
        self.low_percentile=low_percentile
        self.high_percentile=high_percentile
        self.method = method
        if method  == 'radial':
            self.method_f = get_frequency_bands_from_cumulative_energy
        elif method == 'square approximation':
            self.method_f = spectrum_2d
        elif method == 'cfd':
            self.method_f = spectral_cfd
        self.method = method


    def forward(self, pred, y, channel=None, time_step=None, save_plot=False):
        B, H, W, T, C = y.shape
        # find the spectral band edges from the truth using OLD method
        k_low_new, k_high_new, k_freq, E_bins_target = self.method_f(
            y, low_percentile=self.low_percentile, high_percentile=self.high_percentile
        )
        _, _, _, E_bins_pred = self.method_f( # new method
            pred, low_percentile=self.low_percentile, high_percentile=self.high_percentile
        )
        if self.k_low is None:
            self.k_low = k_low_new
        if self.k_high is None:
            self.k_high = k_high_new
        # Aggregate spectral errors by frequency bands
        low_err, mid_err, high_err = aggregate_spectral_energy_by_bands(
            self.k_low, self.k_high, np.abs(np.log(E_bins_pred) - np.log(E_bins_target))
        )
        
        # print(f"\nSpectral error by frequency bands:")
        # print(f"Low frequency error (0 to {self.k_low}): {low_err:.6f}")
        # print(f"Mid frequency error ({self.k_low} to {self.k_high}): {mid_err:.6f}")
        # print(f"High frequency error ({self.k_high}+): {high_err:.6f}")

        if save_plot:
            # increase the font size
            plt.loglog(k_freq, E_bins_target, 'X-',markersize=2, label='target', linewidth=2)
            plt.loglog(k_freq, E_bins_pred, 'o-',markersize=2, label=f'{self.model_name} pred', linewidth=2)
            font_size = 16
            plt.legend(fontsize=font_size)
            # draw the dotted line at two points line 1: (self.k_low,0) to (self.k_low, k_freq[self.k_low]), line 2: (self.k_high,0) to (self.k_high, k_freq[self.k_high])
            plt.axvline(x=self.k_low, color='black', linestyle='--', linewidth=1, ymin=0, ymax=k_freq[self.k_low])
            plt.axvline(x=self.k_high, color='black', linestyle='--', linewidth=1, ymin=0, ymax=k_freq[self.k_high])
            # set the font size for x tick and y tick labels
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            if not os.path.exists(f'{self.save_path}/spectral_error'):
                os.makedirs(f'{self.save_path}/spectral_error')
            plt.savefig(f'{self.save_path}/spectral_error/spectral_error_{self.model_name}_{self.method}_c{channel}_t{time_step}.png')
            plt.clf()
        # plt.show()

        return {'spec_low': low_err, 'spec_mid': mid_err, 'spec_high': high_err, 'k_low': self.k_low, 'k_high': self.k_high}


class Energy_Enstropy_SpectrumError(_WeightedLoss):
    def __init__(self, model_name, save_path):
        super(Energy_Enstropy_SpectrumError, self).__init__()
        self.model_name = model_name
        self.save_path = save_path

    def forward(self, pred, y, Lx=2*np.pi, Ly=2*np.pi, channel=None, time_step=None, save_plot=False):
        B, H, W, C = y.shape
        y_dict = {'pred': pred, 'y': y}
        # Get fields at this time
        for y_key in ['pred', 'y']: # pred and y
            psi_grid = y_dict[y_key][0,..., 1].detach().cpu().numpy()  # streamfunction
            omega_grid = y_dict[y_key][0,..., 0].detach().cpu().numpy() # vorticity

            # Compute velocity from streamfunction
            ux_grid, uy_grid = streamfunction_to_velocity(psi_grid, Lx, Ly)
            
            # Spectra (every time step)
            k_bins, Ek, Zk = compute_spectra(ux_grid, uy_grid, Lx, Ly)

            y_dict[y_key+'_Ek'] = Ek
            y_dict[y_key+'_Zk'] = Zk
            y_dict['k_bins'] = k_bins
        if save_plot:
            # plot the energy and enstropy spectra
            font_size = 16
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            cutoff_idx= 65

            k_nyquist = 64
            # increase the font size
            axs[0].loglog(y_dict['k_bins'][:k_nyquist], y_dict['y_Ek'][:k_nyquist], 'X-',markersize=2, label='target', linewidth=2)
            axs[0].loglog(y_dict['k_bins'][:k_nyquist], y_dict['pred_Ek'][:k_nyquist], 'o-',markersize=2, label=f'{self.model_name} pred', linewidth=2)
            # axs[0].set_xlabel('Wavenumber', fontsize=font_size)
            # axs[0].set_ylim(1e-10, 1e-2) # TODO: remove this later
            axs[0].set_ylabel('Energy', fontsize=font_size)
            axs[0].set_title('Energy Spectrum', fontsize=font_size)

            axs[1].loglog(y_dict['k_bins'][:k_nyquist], y_dict['y_Zk'][:k_nyquist], 'X-',markersize=2, label='target', linewidth=2)
            axs[1].loglog(y_dict['k_bins'][:k_nyquist], y_dict['pred_Zk'][:k_nyquist], 'o-',markersize=2, label=f'{self.model_name} pred', linewidth=2)
            axs[1].set_xlabel('Wavenumber', fontsize=font_size)
            axs[1].set_ylabel('Enstropy', fontsize=font_size)
            axs[1].set_title('Enstropy Spectrum', fontsize=font_size)
           
            plt.legend(fontsize=font_size)
            # set the font size for x tick and y tick labels
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            if not os.path.exists(f'{self.save_path}/spectral_error'):
                os.makedirs(f'{self.save_path}/spectral_error')
            plt.savefig(f'{self.save_path}/spectral_error/energy_enstropy_spectra_{self.model_name}_t{time_step}.png')
            plt.clf()
            

def compute_frequency_spectrum(y_pred, y):
    # y_pred: (B, H, W, T, C), y: (B, H, W, T, C)
    B, H, W, T, C = y.shape

    # Absolute error averaged over batch, time, channels -> (H, W)
    abs_error = torch.abs(y - y_pred)  # (B, H, W, T, C)
    abs_error = rearrange(abs_error, 'b h w t c -> b t c h w')

    # Use full 2D FFT so the spectrum shape matches (B T C H, W)
    abs_error_fft = torch.abs(torch.fft.fft2(abs_error))

    # average over batch, time, channels
    abs_error_fft = abs_error_fft.mean(dim=(0, 1, 2)) # (H, W)
    # Take magnitude and move to numpy for binning
    fourier_amplitudes = abs_error_fft.detach().cpu().numpy()

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

def find_freq_from_linear_fit(freq, energy, slope=-5.0/3):
    # find the intercept that best fit the data (freq, energy), slope is given
    # f(freq)) = slope * freq + intersept 
    # error = (energy - f(freq)) ** 2
    # minimize the error by finding the intercept using convex optimization
    def objective(intercept):
        return np.sum((energy - slope * freq - intercept) ** 2)
    # use scipy.optimize.minimize to find the intercept
    result = minimize_scalar(objective)
    intercept = result.x

    # also need to get the x values of the intercept between two lines: 
    # f(freq) = slope * freq + intercept and energy, use energy - f(freq) and find the roots
    
    # Define function to find roots: energy - (slope * freq + intercept) = 0
    def difference_func(f):
        # Interpolate energy at frequency f
        energy_interp = np.interp(f, freq, energy)
        fitted_value = slope * f + intercept
        return energy_interp - fitted_value
    
    # Find multiple intersection points by trying different initial guesses
    freq_min, freq_max = np.min(freq), np.max(freq)
    initial_guesses = np.linspace(freq_min, freq_max, 10)
    
    roots = []
    for guess in initial_guesses:
        try:
            root = fsolve(difference_func, guess)[0]
            # Check if root is valid and within bounds
            if freq_min <= root <= freq_max and abs(difference_func(root)) < 1e-6:
                # Avoid duplicate roots
                if not any(abs(root - existing_root) < 1e-3 for existing_root in roots):
                    roots.append(int(np.floor(np.exp(root))))
        except:
            continue
    
    roots = sorted(roots)
    print(f"Intersection frequencies: {roots}")
    
    return roots[0], roots[1], intercept


def find_freq_from_percentile(E_freq_cumsum, low_percentile, high_percentile):
    ## return the first index that is greater than the percentile
    low_res = E_freq_cumsum > low_percentile
    # find the first index where low_res is True
    low_idx = np.argmax(low_res)
    high_res = E_freq_cumsum > high_percentile
    # find the first index where high_res is True
    high_idx = np.argmax(high_res)
    return low_idx, high_idx


# USED FOR testing evaluation
def get_frequency_bands_from_cumulative_energy_old(
    y: torch.Tensor,
    low_percentile: float = 0.67,
    high_percentile: float = 0.99,
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
    assert y.ndim == 5, "y must have shape (N,H,W,T,C)"
    N, H, W, T, C = y.shape
    if max_freq is None:
        max_freq = min(H, W) // 2

    device = y.device
    dtype = y.dtype

    y = rearrange(y, 'b h w t c -> b t c h w')

    # Use full 2D FFT so the spectrum shape matches (B T C H, W)
    y_fft = torch.abs(torch.fft.fft2(y))

    # average over batch, time, channels
    y_fft = y_fft.mean(dim=(0, 1, 2)) # (H, W)
    # Take magnitude and move to numpy for binning
    fourier_amplitudes = y_fft.detach().cpu().numpy()

    # Create the k-frequency grid for rectangular image
    kfreq_x = np.fft.fftfreq(W) * W
    kfreq_y = np.fft.fftfreq(H) * H
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    # Flatten the arrays to use in binning (1D arrays of equal length,  (H*W,) ) 
    knrm = knrm.ravel() # ALL the frequences in the image
    fourier_amplitudes = fourier_amplitudes.ravel() # ALL the fourier amplitudes in the image

    # Define the bins for the wavenumber - use the minimum dimension for binning
    min_dim = min(H, W)
    kbins = np.arange(1, min_dim // 2 + 1, 1.0)

    # Bin the data (radial mean), turn the 2D array into 1D array
    E_freq, _, _ = stats.binned_statistic(
        knrm, fourier_amplitudes, statistic="mean", bins=kbins
    )


    log_E_freq = np.log(E_freq)
    log_freq = np.log(kbins[:len(log_E_freq)])
    k_freq = kbins[:len(E_freq)]
        
    # k_low, k_high, intercept = find_freq_from_linear_fit(log_freq, log_E_freq)
    # plot it temporarily
    # plt.loglog(k_freq, E_freq, 'X-',markersize=1, label='data')
    # plt.loglog(k_freq, np.exp(-5.0/3*log_freq + intercept), 'r--', label='linear fit') # linear fit
    #     # Mark intersection points
    # plt.legend()
    # plt.show()
    # k_low = 12
    # k_high = 40

    # compute the cumulative sum of the energy
    E_freq_cumsum = np.cumsum(E_freq)
    E_freq_cumsum = E_freq_cumsum / E_freq_cumsum[-1]
    k_low, k_high = find_freq_from_percentile(E_freq_cumsum, low_percentile, high_percentile)

    return k_low, k_high, k_freq, E_freq


def get_frequency_bands_from_cumulative_energy(
    y: torch.Tensor,
    low_percentile: float = 0.67,
    high_percentile: float = 0.99,
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
    assert y.ndim == 5, "y must have shape (N,H,W,T,C)"
    N, H, W, T, C = y.shape
    if max_freq is None:
        max_freq = min(H, W) // 2

    device = y.device
    dtype = y.dtype

    y = rearrange(y, 'b h w t c -> (b t c) h w')

    # Use full 2D FFT so the spectrum shape matches (B T C H, W)
    print("y shape", y.shape)
    y_fft = torch.fft.fft2(y)

    # Take magnitude and move to numpy for binning
    # fourier_amplitudes = (torch.abs(y_fft)**2).detach().cpu().numpy()
    fourier_amplitudes = (torch.abs(y_fft)**2).detach().cpu().numpy()
    # Create the k-frequency grid for rectangular image
    kfreq_x = np.fft.fftfreq(W) * W
    kfreq_y = np.fft.fftfreq(H) * H
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    # Flatten the arrays to use in binning (1D arrays of equal length,  (H*W,) ) 
    knrm = knrm.ravel() # ALL the frequences in the image

    # Define the bins for the wavenumber - use the minimum dimension for binning
    min_dim = min(H, W)
    kbins = np.arange(1, min_dim // 2 + 1, 1.0)

    amplitudes = []
    for idx in range(fourier_amplitudes.shape[0]):
        fourier_idx = fourier_amplitudes[idx].ravel()
        # Bin the data (radial mean), turn the 2D array into 1D array
        E_freq, _, _ = stats.binned_statistic(
            knrm, fourier_idx, statistic="mean", bins=kbins
       )
        amplitudes.append(E_freq)
    amplitudes = np.array(amplitudes)
    E_freq = amplitudes.mean(axis=0)

    log_E_freq = np.log(E_freq)
    log_freq = np.log(kbins[:len(log_E_freq)])
    # k_freq = 0.5 * (kbins[1:] + kbins[:-1])
    k_freq = kbins[:len(E_freq)]
        
    # k_low, k_high, intercept = find_freq_from_linear_fit(log_freq, log_E_freq)
    # plot it temporarily
    # plt.loglog(k_freq, E_freq, 'X-',markersize=1, label='data')
    # plt.loglog(k_freq, np.exp(-5.0/3*log_freq + intercept), 'r--', label='linear fit') # linear fit
    #     # Mark intersection points
    # plt.legend()
    # plt.show()
    # k_low = 12
    # k_high = 40

    # compute the cumulative sum of the energy
    E_freq_cumsum = np.cumsum(E_freq)
    E_freq_cumsum = E_freq_cumsum / E_freq_cumsum[-1]
    k_low, k_high = find_freq_from_percentile(E_freq_cumsum, low_percentile, high_percentile)

    return k_low, k_high, k_freq, E_freq



def aggregate_spectral_energy_by_bands(
    k_low: int,
    k_high: int,
    E_diff: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # E_diff: (H, W)
    # k_low: int
    # k_high: int
        
    # Aggregate errors by frequency bands
    low_band_error = E_diff[:k_low].mean()
    mid_band_error = E_diff[k_low:k_high].mean()
    high_band_error = E_diff[k_high:].mean()
    
    return low_band_error, mid_band_error, high_band_error


def spectrum_2d(y: torch.Tensor,
    low_percentile: float = 0.67,
    high_percentile: float = 0.99,
    max_freq: int = None,
    eps: float = 1e-12,
    normalize=True,
) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Paramaters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.
    normalize: bool
        whether to apply normalization to the output of the 2D FFT. 
        If True, normalizes the outputs by ``1/n_observations``
        (actually ``1/sqrt(n_observations * n_observations)``). 
    Returns
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
        The spectrum is computed using a square approximation to radial
        binning, meaning that the wavenumber 'bin' into which a particular 
        coefficient is the coefficient's location along the diagonal, indexed 
        from the top-left corner of the 2d FFT output. 
    """
    T = y.shape[0]
    y = rearrange(y, 'b h w t c -> (b t c) h w')
    n_observations = y.shape[-1]
    signal = y.view(T, n_observations, n_observations)

    if normalize:
        signal = torch.fft.fft2(signal, norm="ortho")
    else:
        signal = torch.fft.rfft2(
            signal, s=(n_observations, n_observations), norm="backward"
        )

    # 2d wavenumbers following PyTorch fft convention
    k_max = n_observations // 2
    wavenumers = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1),
            torch.arange(start=-k_max, end=0, step=1),
        ),
        0,
    ).repeat(n_observations, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers

    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k

    # Remove symmetric components from wavenumbers
    index = -1.0 * torch.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index[0:k_max1, 0:k_max1] = sum_k[0:k_max1, 0:k_max1]
    print(index)
    spectrum = torch.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = torch.where(index == j)
        spectrum[:, j - 1] = (signal[:, ind[0], ind[1]].sum(dim=1)).abs() ** 2

    E_freq = spectrum.mean(dim=0)
    E_freq = E_freq[:n_observations//2] # keep only the positive frequencies
    print("E_freq shape", E_freq.shape)
    
    min_dim = n_observations
    kbins = np.arange(1, min_dim // 2 + 1, 1.0)
    k_freq = kbins[:len(E_freq)]
    
    E_freq_cumsum = np.cumsum(E_freq)
    E_freq_cumsum = E_freq_cumsum / E_freq_cumsum[-1]
    k_low, k_high = find_freq_from_percentile(E_freq_cumsum, low_percentile, high_percentile)
    return k_low, k_high, k_freq, E_freq


def spectral_cfd(y: torch.Tensor,
    low_percentile: float = 0.67,
    high_percentile: float = 0.99,
    max_freq: int = None,
    eps: float = 1e-12,
) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
    y = rearrange(y, 'b h w t c -> (b t c) h w')
    vorticity = y
    if isinstance(vorticity, np.ndarray):
        vorticity = torch.from_numpy(vorticity)
    n = vorticity.shape[-1]
    h = 2 * np.pi / n
    kx = torch.fft.fftfreq(n, d=h)
    ky = torch.fft.fftfreq(n, d=h)
    kx, ky = torch.meshgrid([kx, ky], indexing="ij")
    kmax = n // 2
    kx = kx[..., : kmax + 1]
    ky = ky[..., : kmax + 1]
    k2 = (4 * torch.pi**2) * (kx**2 + ky**2)
    print("k2 shape", k2.shape)
    k2[0, 0] = 1.0

    wh = torch.fft.rfft2(vorticity)

    tke = (torch.abs(wh)**2).cpu()
    kmod = torch.sqrt(k2)
    k = torch.arange(1, kmax+1, dtype=torch.float64)  # Nyquist limit for this grid
    print("k shape", k.shape)
    dk = (torch.max(k) - torch.min(k)) / (2 * n)
    
    n_samples = tke.shape[0]
    E_freq = []
    for s in range(n_samples):
        Ens = torch.zeros_like(k)
        for i in range(len(k)):
            Ens[i] += (tke[s, (kmod < k[i] + dk) & (kmod >= k[i] - dk)]).sum()
        E_freq.append(Ens)
    E_freq = torch.stack(E_freq)
    E_freq = E_freq.mean(dim=0)
    
    n_observations = n
    E_freq = Ens
    print("E_freq shape", E_freq.shape)
    
    min_dim = n_observations
    kbins = np.arange(1, min_dim // 2 + 1, 1.0)
    k_freq = kbins[:len(E_freq)]
    
    E_freq_cumsum = np.cumsum(E_freq)
    E_freq_cumsum = E_freq_cumsum / E_freq_cumsum[-1]
    k_low, k_high = find_freq_from_percentile(E_freq_cumsum, low_percentile, high_percentile)
    return k_low, k_high, k_freq, E_freq

if __name__ == "__main__":
    # x = torch.randn([2, 128, 128, 1, 3])
    # y = torch.randn([2, 128, 128, 1, 3])
    x = torch.randn([2, 6, 6, 1, 3])
    y = torch.randn([2, 6, 6, 1, 1])
    spectrum = spectrum_2d(y)
    # evaluator = Evaluator(temporal=True, griddata=True, component='all', normalizer=None)
    # metrics = evaluator(x, y)
    # print(metrics)
    # myloss = FourierLoss(beta=1)
    # Test the new cumulative energy-based frequency band selection
    print("Testing cumulative energy-based frequency aggregation...")
    
    # Create test data (N, H, W, C)
    torch.manual_seed(42)
    N, H, W, C = 2, 128, 128, 1
    target = torch.randn([N, H, W, 1, C])
    pred = target + 0.1 * torch.randn_like(target)  # Add some noise
    
    print(f"Data shape: {target.shape}")
    print(f"Max frequency: {min(H, W) // 2}")
    
    # Get frequency bands using cumulative energy
    k_low, k_mid, k_high,  E_bins_target = get_frequency_bands_from_cumulative_energy(
        target, low_percentile=0.70, high_percentile=0.99
    )
    _, _, _,E_bins_pred = get_frequency_bands_from_cumulative_energy(
        pred, low_percentile=0.80, high_percentile=0.99
    )

    # Aggregate spectral errors by frequency bands
    low_err, mid_err, high_err = aggregate_spectral_energy_by_bands(
         k_low, k_high, np.abs(E_bins_pred - E_bins_target)
    )
    
    print(f"\nSpectral error by frequency bands:")
    print(f"Low frequency error (0 to {k_low}): {low_err:.6f}")
    print(f"Mid frequency error ({k_low} to {k_high}): {mid_err:.6f}")
    print(f"High frequency error ({k_high}+): {high_err:.6f}")
    
    print("Test completed successfully!")
    # for key, value in metrics.items():
    #     print(key, value.shape)
