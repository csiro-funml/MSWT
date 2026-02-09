"""
Sanity check: build a model from the NS2D MSWT config and run one forward pass with random input.
Run from the project root: python test_sanity_check.py
"""
import os
import sys
import yaml
import torch

# Project root = directory containing this script
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.mswt import PeriodicMSWT2D_Patching


def main():
    config_path = os.path.join(
        ROOT, "NS2D_ChaoticKolmogorovFlow", "configs", "linear_used_in_paper", "MSWT.yaml"
    )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)

    model_cfg = config["model"]
    # Data shape from config (S2, T2 used in NS2D; we only need spatial size for one step)
    S = config["data"].get("S2", 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model_cfg.get("name", "").lower()
    if "multiscale_wavelet" not in model_name and "mswt" not in model_name and "periodic" not in model_name:
        raise ValueError(f"This script expects an MSWT-style config; got model name: {model_cfg.get('name')}")

    model = PeriodicMSWT2D_Patching(
        wave=model_cfg.get("wave", "haar"),
        input_dim=model_cfg.get("in_chans", 3),
        output_dim=model_cfg.get("out_chans", 1),
        dim=model_cfg.get("dim", None),
        dims=model_cfg.get("dims", []),
        use_efficient_attention=model_cfg.get("use_efficient_attention", False),
        efficient_layers=model_cfg.get("efficient_layers", [0, 1, 2]),
        add_grid=model_cfg.get("add_grid", False),
        add_periodic_grid=model_cfg.get("add_periodic_grid", False),
        patch_size=model_cfg.get("patch_size", None),
        local_attention_size=model_cfg.get("local_attention_size", None),
    ).to(device)

    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_cfg.get('name')}  |  Parameters: {nparams:,}")

    # Random input: (batch, H, W, in_chans) as in NS2D test
    batch_size = 2
    in_chans = model_cfg.get("in_chans", 3)
    x = torch.randn(batch_size, S, S, in_chans, device=device, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        out = model(x)

    # Output is (batch, H, W, out_chans) or (batch, H*W, out_chans) depending on implementation
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")

    assert out.shape[0] == batch_size and out.shape[-1] == model_cfg.get("out_chans", 1), (
        f"Unexpected output shape: {out.shape}"
    )
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
