from utils.criterion import Energy_Enstropy_SpectrumError
import torch

loss = Energy_Enstropy_SpectrumError(model_name='FNO', save_path='logs/FNO_ns2d_dedalus_ntrain4968_standardnorm')
pred = torch.randn(1, 128, 128, 2)
target = torch.randn(1, 128, 128, 2)
loss_metric = loss(pred, target, save_plot=False, time_step=0)