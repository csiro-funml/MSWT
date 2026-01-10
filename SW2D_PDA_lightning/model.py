"""LightningModule for SW2D_PDA training."""

import os
import sys
import torch
import pytorch_lightning as pl
from einops import rearrange

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.criterion import LpLoss
from models.fno import FNO2d
from models.high_frequency_scaling import ResUNet
from models.wno import WNO2d
from models.saot import SAOTModel
from models.wavelet_transform import MultiscaleWaveletTransformer2D
from models.wavelet_transform_exploration import (
    MultiscaleWaveletTransformer2DDecoderNoAttention,
    MultiscaleWaveletTransformer2DEfficient,
    MultiscaleWaveletDoubleAttention,
    MSWT_DeNoAttn_StackLayers,
)
from models.periodic_mswt import PeriodicMultiscaleWaveletTransformer2D
from models.periodic_mswt_bases import (
    PeriodicMultiscaleWaveletTransformer2D_DB2,
    PeriodicMultiscaleWaveletTransformer2D_DB4,
    PeriodicMultiscaleWaveletTransformer2D_SYM4,
)
from models.pderefiner import PDERefiner
from models.pderefiner_unet import UNetRefiner

def torch2dgrid(num_x, num_y, bot=(0,0), top=(1,1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y)
    xx, yy = torch.meshgrid(x_arr, y_arr, indexing='ij')
    mesh = torch.stack([xx, yy], dim=2)
    return mesh


def build_model(model_cfg, s_data, device):
    model_name = model_cfg.get('name', 'fno2d').lower()

    if model_name == 'fno2d':
        model = FNO2d(
            modes1=model_cfg['modes1'],
            modes2=model_cfg['modes2'],
            fc_dim=model_cfg['fc_dim'],
            layers=model_cfg['layers'],
            act=model_cfg['act'],
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
        )
    elif model_name == 'hfs':
        model = ResUNet(
            in_c=model_cfg.get('in_c', 3),
            out_c=model_cfg.get('out_c', 1),
            target_params=model_cfg.get('target_params', 'medium'),
            device=device,
        )
    elif model_name == 'pderefiner':
        model = PDERefiner(
            name=model_cfg.get('basemodel_name', 'Unetmod-64'),
            time_history=model_cfg.get('time_history', 1),
            time_future=model_cfg.get('time_future', 1),
            time_gap=0,
            max_num_steps=model_cfg.get('max_num_steps', 1),
            n_spatial_dim=model_cfg.get('n_spatial_dim', 2),
            in_channels=model_cfg.get('in_channels', 3),
            out_channels=model_cfg.get('out_channels', 1),
            trajlen=model_cfg.get('trajlen', 64),
            activation=model_cfg.get('activation', 'gelu'),
            criterion=model_cfg.get('criterion', 'mse'),
            hidden_channels=model_cfg.get('hidden_channels', 16),
            n_blocks=model_cfg.get('n_blocks', 3),
        )
    elif model_name in ['wno', 'wno2d']:
        dummy = torch.zeros(1, 1, s_data[0], s_data[1], device=device)
        model = WNO2d(
            in_channels=model_cfg.get('in_chans', 3),
            out_channels=model_cfg.get('out_chans', 1),
            width=model_cfg.get('width', 64),
            level=model_cfg.get('level', 3),
            dummy_data=dummy,
        )
    elif model_name in ['saot', 'saot2d']:
        model = SAOTModel(
            space_dim=model_cfg.get('space_dim', 2),
            n_layers=model_cfg.get('n_layers', 3),
            n_hidden=model_cfg.get('n_hidden', 64),
            dropout=model_cfg.get('dropout', 0.0),
            n_head=model_cfg.get('n_head', 4),
            Time_Input=model_cfg.get('Time_Input', False),
            mlp_ratio=model_cfg.get('mlp_ratio', 1),
            fun_dim=model_cfg.get('fun_dim', 1),
            out_dim=model_cfg.get('out_dim', 1),
            H=s_data[0],
            W=s_data[1],
            slice_num=model_cfg.get('slice_num', 32),
            ref=model_cfg.get('ref', 8),
            unified_pos=model_cfg.get('unified_pos', 0),
            is_filter=model_cfg.get('is_filter', True),
        )
    elif model_name in ['refiner_unet']:
        model = UNetRefiner(
            input_channels=model_cfg.get('in_channels', 3),
            output_channels=model_cfg.get('out_channels', 1),
            time_history=model_cfg.get('time_history', 0),
            time_future=model_cfg.get('time_future', 0),
            hidden_channels=model_cfg.get('hidden_channels', 16),
            activation=model_cfg.get('activation', 'gelu'),
            n_blocks=model_cfg.get('n_blocks', 3),
        )
    elif model_name in ['multiscale_wavelet', 'multiscale_wavelet2d', 'multiscale_wavelet_transformer2d']:
        model = MultiscaleWaveletTransformer2D(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            patch_size=model_cfg.get('patch_size', None),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
        )
    elif model_name in ['multiscale_wavelet2d_nodecoderattn']:
        model = MultiscaleWaveletTransformer2DDecoderNoAttention(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            patch_size=model_cfg.get('patch_size', None),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
        )
    elif model_name in ['multiscale_wavelet2d_attn05124_group4']:
        model = MultiscaleWaveletTransformer2DEfficient(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            patch_size=model_cfg.get('patch_size', None),
        )
    elif model_name in ['multiscale_wavelet2d_double_attn']:
        model = MultiscaleWaveletDoubleAttention(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
        )
    elif model_name in ['multiscale_wavelet2d_periodic', 'mswt_periodic', 'periodic_mswt']:
        model = PeriodicMultiscaleWaveletTransformer2D(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
            local_attention_size=model_cfg.get('local_attention_size', None),
            add_grid=model_cfg.get('add_grid', False),
            add_periodic_grid=model_cfg.get('add_periodic_grid', False),
            patch_size=model_cfg.get('patch_size', None),
        )
    elif model_name in ['multiscale_wavelet2d_periodic_db2', 'mswt_periodic_db2', 'periodic_mswt_db2']:
        model = PeriodicMultiscaleWaveletTransformer2D_DB2(
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
            add_grid=model_cfg.get('add_grid', False),
            add_periodic_grid=model_cfg.get('add_periodic_grid', False),
            patch_size=model_cfg.get('patch_size', None),
            local_attention_size=model_cfg.get('local_attention_size', 8),
        )
    elif model_name in ['multiscale_wavelet2d_periodic_db4', 'mswt_periodic_db4', 'periodic_mswt_db4']:
        model = PeriodicMultiscaleWaveletTransformer2D_DB4(
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
            add_grid=model_cfg.get('add_grid', False),
            add_periodic_grid=model_cfg.get('add_periodic_grid', False),
            patch_size=model_cfg.get('patch_size', None),
            local_attention_size=model_cfg.get('local_attention_size', 8),
        )
    elif model_name in ['multiscale_wavelet2d_periodic_sym4', 'mswt_periodic_sym4', 'periodic_mswt_sym4']:
        model = PeriodicMultiscaleWaveletTransformer2D_SYM4(
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', None),
            dims=model_cfg.get('dims', []),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
            add_grid=model_cfg.get('add_grid', False),
            add_periodic_grid=model_cfg.get('add_periodic_grid', False),
            patch_size=model_cfg.get('patch_size', None),
            local_attention_size=model_cfg.get('local_attention_size', 8),
        )
    elif model_name in ['multiscale_wavelet2d_denoattn_stacklayers3']:
        model = MSWT_DeNoAttn_StackLayers(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dims=model_cfg.get('dims', []),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
        )
    else:
        raise ValueError(f'Model {model_name} not supported')

    return model


class SW2DPDALightningModule(pl.LightningModule):
    def __init__(self, config, s_data):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model_cfg = config['model']
        self.train_cfg = config['train']
        self.external_grid = self.model_cfg.get('external_grid', True)
        self._grid_cache = {}

        self.model = build_model(self.model_cfg, s_data, device=self.device)
        self.loss_fn = LpLoss(size_average=True)

    def _get_grid(self, batch, device, h, w):
        if not self.external_grid:
            return None
        key = (device, h, w)
        if key not in self._grid_cache:
            grid = torch2dgrid(h, w).to(device).unsqueeze(0)
            self._grid_cache[key] = grid
        return self._grid_cache[key].expand(batch, -1, -1, -1)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        if self.external_grid:
            grid = self._get_grid(x.shape[0], x.device, x.shape[1], x.shape[2])
            x_in = torch.cat((x, grid), dim=-1)
        else:
            x_in = x

        if isinstance(self.model, PDERefiner):
            loss = self.model.training_step((x, y))
            pred = None
        else:
            pred = self.model(x_in)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = self.loss_fn(pred, y)
        return loss, pred

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch)
        self.log('train/l2', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch)
        self.log('val/l2', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            betas=(0.9, 0.999),
            lr=self.train_cfg['base_lr'],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_cfg['milestones'],
            gamma=self.train_cfg['scheduler_gamma'],
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }
