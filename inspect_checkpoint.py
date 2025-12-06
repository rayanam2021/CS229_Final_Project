#!/usr/bin/env python3
"""
Checkpoint Inspection Utility

This script helps you inspect checkpoint files to understand their contents
and verify they can be loaded correctly.

Usage:
    python inspect_checkpoint.py --checkpoint <path_to_checkpoint>
    python inspect_checkpoint.py --run_dir <path_to_run_directory>
"""

import torch
import argparse
import os
import sys
import glob
from pathlib import Path


def format_size(num_bytes):
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def count_parameters(state_dict):
    """Count total parameters in a state dict."""
    total = 0
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel()
    return total


def inspect_checkpoint(checkpoint_path):
    """Inspect a single checkpoint file."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return False

    print(f"\n{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}")

    # File size
    file_size = os.path.getsize(checkpoint_path)
    print(f"\nFile size: {format_size(file_size)}")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Basic info
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

        # Epoch
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")

        # Network state
        if 'network_state' in checkpoint:
            network_state = checkpoint['network_state']
            num_params = count_parameters(network_state)
            print(f"\nNetwork State:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Layers: {len(network_state)}")
            print(f"  Layer names:")
            for key in list(network_state.keys())[:10]:  # Show first 10
                shape = network_state[key].shape
                print(f"    - {key}: {shape}")
            if len(network_state) > 10:
                print(f"    ... and {len(network_state) - 10} more layers")

        # Optimizer state
        if 'optimizer_state' in checkpoint:
            opt_state = checkpoint['optimizer_state']
            print(f"\nOptimizer State:")
            print(f"  Keys: {list(opt_state.keys())}")
            if 'param_groups' in opt_state:
                for i, pg in enumerate(opt_state['param_groups']):
                    print(f"  Param group {i}:")
                    print(f"    lr: {pg.get('lr', 'N/A')}")
                    print(f"    weight_decay: {pg.get('weight_decay', 'N/A')}")

        # Training history
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            print(f"\nTraining History:")
            for key, values in history.items():
                if isinstance(values, list) and len(values) > 0:
                    print(f"  {key}: {len(values)} values")
                    if len(values) >= 3:
                        print(f"    First 3: {values[:3]}")
                        print(f"    Last 3: {values[-3:]}")
                    else:
                        print(f"    Values: {values}")

        print(f"\n✓ Checkpoint is valid and can be loaded")
        return True

    except Exception as e:
        print(f"\n✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def inspect_run_directory(run_dir):
    """Inspect all checkpoints in a run directory."""
    if not os.path.exists(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        return False

    print(f"\n{'='*80}")
    print(f"Run Directory: {run_dir}")
    print(f"{'='*80}")

    # Check for config
    config_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(config_path):
        print(f"✓ run_config.json found")
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"  Target episodes: {config.get('monte_carlo', {}).get('num_episodes', 'N/A')}")
            print(f"  Batch size: {config.get('training', {}).get('batch_size', 'N/A')}")
            print(f"  Learning rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
    else:
        print(f"✗ run_config.json not found")

    # Check for checkpoints
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        print(f"✗ checkpoints directory not found")
        return False

    checkpoint_files = glob.glob(os.path.join(ckpt_dir, "checkpoint_ep_*.pt"))
    checkpoint_files.extend(glob.glob(os.path.join(ckpt_dir, "best.pt")))

    if not checkpoint_files:
        print(f"✗ No checkpoint files found")
        return False

    print(f"\n✓ Found {len(checkpoint_files)} checkpoint(s)")

    # Parse episode numbers
    episode_data = []
    for ckpt_file in checkpoint_files:
        basename = os.path.basename(ckpt_file)
        try:
            if basename.startswith("checkpoint_ep_"):
                ep_num = int(basename.split('_')[-1].split('.')[0])
                episode_data.append((ep_num, ckpt_file))
            elif basename == "best.pt":
                episode_data.append((-1, ckpt_file))  # Special marker for best
        except:
            continue

    episode_data.sort(key=lambda x: x[0])

    print("\nCheckpoints:")
    for ep_num, ckpt_file in episode_data:
        file_size = os.path.getsize(ckpt_file)
        if ep_num == -1:
            print(f"  - best.pt ({format_size(file_size)})")
        else:
            print(f"  - Episode {ep_num} ({format_size(file_size)})")

    # Count episodes
    episode_dirs = glob.glob(os.path.join(run_dir, "episode_*"))
    print(f"\n✓ Found {len(episode_dirs)} episode directories")

    # Inspect latest checkpoint
    if episode_data:
        latest_ep, latest_ckpt = episode_data[-1]
        print(f"\nInspecting latest checkpoint (Episode {latest_ep}):")
        inspect_checkpoint(latest_ckpt)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Inspect checkpoint files for AlphaZero training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--checkpoint',
        type=str,
        help='Path to a specific checkpoint file'
    )
    group.add_argument(
        '--run_dir',
        type=str,
        help='Path to a run directory (inspects all checkpoints)'
    )

    args = parser.parse_args()

    if args.checkpoint:
        success = inspect_checkpoint(args.checkpoint)
    else:
        success = inspect_run_directory(args.run_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
