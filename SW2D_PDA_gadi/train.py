"""Entry point for SW2D_PDA Lightning training."""

import argparse
import os
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import sys

sys.path.append(os.path.dirname(__file__))
from data import SW2DPDALightningDataModule
from model import SW2DPDALightningModule


def parse_args():
    parser = argparse.ArgumentParser(description='SW2D_PDA Lightning training')
    parser.add_argument('--config_path', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--test_ratio', type=float, default=0.0, help='Hold out fraction for validation split')
    parser.add_argument('--test_seed', type=int, default=42, help='Seed for validation split')
    parser.add_argument('--synthetic_samples', type=int, default=0, help='Use synthetic data for quick runs')
    parser.add_argument('--devices', type=str, default='1', help='Lightning devices arg, e.g. 1 or 0,1')
    parser.add_argument('--accelerator', type=str, default='auto', help='cpu, gpu, or auto')
    parser.add_argument('--strategy', type=str, default='auto', help='ddp, ddp_find_unused_parameters_false, or auto')
    parser.add_argument('--precision', type=str, default='32', help='16-mixed, bf16-mixed, or 32')
    parser.add_argument('--max_epochs', type=int, default=None, help='Override train.epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Enable DataLoader pin_memory')
    return parser.parse_args()


def parse_devices(devices_str):
    if devices_str.lower() == 'auto':
        return 'auto'
    if ',' in devices_str:
        return [int(x) for x in devices_str.split(',')]
    return int(devices_str)


def main():
    args = parse_args()
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    if args.max_epochs is not None:
        if args.max_epochs <= 0:
            raise ValueError('max_epochs must be positive when provided.')
        config['train']['epochs'] = args.max_epochs

    data_cfg = config['data']
    train_cfg = config['train']

    data_module = SW2DPDALightningDataModule(
        data_cfg,
        batch_size=train_cfg['batchsize'],
        test_ratio=args.test_ratio,
        test_seed=args.test_seed,
        synthetic_samples=args.synthetic_samples,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    data_module.setup()

    lightning_module = SW2DPDALightningModule(config, s_data=data_module.S_data)

    save_dir = train_cfg['save_dir'] if torch.cuda.is_available() else 'saved_models'
    tb_dir = train_cfg.get('tensorboard_dir') or os.path.join(save_dir, 'tensorboard')
    logger = TensorBoardLogger(save_dir=tb_dir, name='sw2d_pda_lightning')

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=train_cfg.get('save_name', 'model') + '-{epoch:03d}',
        save_last=True,
        save_top_k=1,
        monitor='val/l2' if args.test_ratio > 0 else None,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=train_cfg['epochs'],
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        strategy=args.strategy,
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    trainer.fit(lightning_module, datamodule=data_module)


if __name__ == '__main__':
    main()
