#!/usr/bin/env python3
"""
Parse a training log and write scalars to TensorBoard for **every epoch**:
- train l2 norm
- train l2 denorm
- test rel l2 loss (if present on that epoch's line)

Each scalar is logged with global_step == epoch.
"""

import argparse
import os
import re
from pathlib import Path

# Prefer torch SummaryWriter (common), fall back to tensorboardX if torch isn't installed.
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    from tensorboardX import SummaryWriter  # type: ignore


# Patterns
EPOCH_LINE = re.compile(r"""^epoch\s+(\d+),\s*best epoch:\s*(\d+)\b.*$""", re.IGNORECASE)
FLOAT = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
TRAIN_PAIR = re.compile(
    rf"""train l2 norm\s+({FLOAT})\s+train l2 denorm\s+({FLOAT})""",
    re.IGNORECASE,
)
TEST_LOSS = re.compile(
    rf"""test rel l2 loss\s+({FLOAT})""",
    re.IGNORECASE,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Write per-epoch scalars to TensorBoard from a log.txt.")
    ap.add_argument("--log", required=True, help="Path to log.txt")
    ap.add_argument("--out", required=True, help="Output directory for TensorBoard event files")
    ap.add_argument("--run-name", type=str, default="from_log", help="TensorBoard run name (subdir)")
    return ap.parse_args()


def main():
    # args = parse_args()
    log_path = Path('logs/FNO_ns2d_dedalus_ntrain4968/old_log.txt')
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    # Make writer
    logdir = Path('logs/FNO_ns2d_dedalus_ntrain4968_minmax') 
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))

    n_epochs_logged = 0
    missing_train = []

    for line in text.splitlines():
        s = line.strip()
        m_epoch = EPOCH_LINE.match(s)
        if not m_epoch:
            continue
        epoch = int(m_epoch.group(1))

        m_train = TRAIN_PAIR.search(s)
        m_test = TEST_LOSS.search(s)

        if not m_train:
            # Keep track of epochs where train metrics aren't present (rare)
            missing_train.append(epoch)
            continue

        train_l2_norm = float(m_train.group(1))
        train_l2_denorm = float(m_train.group(2))
        test_rel_l2_loss = float(m_test.group(1)) if m_test else None

        # Write scalars for this epoch
        writer.add_scalar("train_loss_norm", train_l2_norm, global_step=epoch)
        writer.add_scalar("train_loss_denorm", train_l2_denorm, global_step=epoch)
        if test_rel_l2_loss is not None:
            writer.add_scalar("test_rel_l2_loss", test_rel_l2_loss, global_step=epoch)

        n_epochs_logged += 1

    writer.flush()
    writer.close()

    print(f"[epochs_logged] {n_epochs_logged}")
    if missing_train:
        missing_train.sort()
        print(f"[epochs_missing_train] {missing_train[:10]}{'...' if len(missing_train)>10 else ''}")
    print(f"TensorBoard logs written to: {logdir}")
    print("Open with: tensorboard --logdir", logdir.parent.resolve())


if __name__ == "__main__":
    main()
