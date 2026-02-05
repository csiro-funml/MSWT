import torch
import numpy as np
import os

def integrate_grid(ugrid, dimensionless=False, polar_opt=0, nlon=96, radius=6.37122E6, quad_weights=None):

    dlon = 2 * torch.pi / nlon
    radius = 1 if dimensionless else radius
    if polar_opt > 0:
        out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * quad_weights[polar_opt:-polar_opt] * dlon * radius**2, dim=(-2, -1))
    else:
        out = torch.sum(ugrid * quad_weights * dlon * radius**2, dim=(-2, -1))
    return out

def l2loss_sphere(prd, tar, relative=False, squared=True, quad_weights=None):
    loss = integrate_grid((prd - tar)**2, dimensionless=True, quad_weights=quad_weights).sum(dim=-1)
    if relative:
        loss = loss / integrate_grid(tar**2, dimensionless=True, quad_weights=quad_weights).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


def legendre_gauss_weights(n, a=-1.0, b=1.0):
    r"""
    Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b]
    """

    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)





def save_checkpoint(path, name, model, epoch, optimizer=None, scheduler=None):
    ckpt_dir = path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    optim_dict = optimizer.state_dict() if optimizer is not None else None
    sched_dict = scheduler.state_dict() if scheduler is not None else None

    torch.save({
        'model': model_state_dict,
        'optim': optim_dict,
        'scheduler': sched_dict,
        'epoch': epoch
    }, ckpt_dir + name)
    print('Checkpoint is saved at %s' % ckpt_dir + name)