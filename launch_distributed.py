#!/usr/bin/env python3
"""
Multi-GPU training launcher for the diffusion neural operator
This script automatically detects available GPUs and launches distributed training
"""

import os
import sys
import subprocess
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description='Launch distributed training')
    parser.add_argument('--model', type=str, default='diffusion', help='Model type')
    parser.add_argument('--dataset', type=str, default='ns2d_pda', help='Dataset')
    parser.add_argument('--use_writer', action='store_true', default=False, help='Use tensorboard writer')
    parser.add_argument('--comment', type=str, default="", help='Comment for logging')
    parser.add_argument('--log_path', type=str, default='/scratch3/wan410/operator_learning_model/', help='Log path')
    parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs (auto-detect if not specified)')
    parser.add_argument('--master_port', type=str, default='12355', help='Master port for distributed training')
    
    args = parser.parse_args()
    
    # Auto-detect number of GPUs if not specified
    if args.world_size is None:
        args.world_size = torch.cuda.device_count()
    
    if args.world_size < 2:
        print("Warning: Less than 2 GPUs detected. Running single GPU training...")
        # Run single GPU training
        cmd = [
            sys.executable, 'train_diffusion_NO.py',
            '--model', args.model,
            '--dataset', args.dataset,
            '--comment', args.comment,
            '--log_path', args.log_path
        ]
        if args.use_writer:
            cmd.append('--use_writer')
        
        subprocess.run(cmd)
    else:
        print(f"Launching distributed training on {args.world_size} GPUs...")
        
        # Set environment variables
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.master_port
        
        # Build the command for distributed training
        cmd = [
            sys.executable, 'train_diffusion_NO.py',
            '--distributed',
            '--world_size', str(args.world_size),
            '--model', args.model,
            '--dataset', args.dataset,
            '--comment', args.comment,
            '--log_path', args.log_path
        ]
        if args.use_writer:
            cmd.append('--use_writer')
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)

if __name__ == '__main__':
    main()
