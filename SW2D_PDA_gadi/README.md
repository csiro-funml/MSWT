# SW2D_PDA Lightning

Minimal PyTorch Lightning training pipeline for SW2D_PDA with multi-GPU support.

## Setup

Ensure `pytorch_lightning` is installed and your environment can import the
existing project modules.

## Example usage

Single GPU (or CPU):

```
python3 SW2D_PDA_lightning/train.py \
  --config_path SW2D_PDA/configs/MSWT_periodic.yaml \
  --test_ratio 0.1 \
  --synthetic_samples 0 \
  --devices 1 \
  --accelerator gpu
```

Multi-GPU (DDP):

```
python3 SW2D_PDA_lightning/train.py \
  --config_path SW2D_PDA/configs/MSWT_periodic.yaml \
  --test_ratio 0.1 \
  --devices 0,1 \
  --accelerator gpu \
  --strategy ddp
```

Quick sanity run:

```
python3 SW2D_PDA_lightning/train.py \
  --config_path SW2D_PDA/configs/MSWT_periodic.yaml \
  --synthetic_samples 100 \
  --max_epochs 2
```

## Notes

- The model builder mirrors `SW2D_PDA/train_operator_AR_rell2_2d.py` and supports
  the periodic MSWT variant.
- External grid concatenation is controlled by `model.external_grid` in YAML.
- Periodic grid features are controlled by `model.add_periodic_grid`.
