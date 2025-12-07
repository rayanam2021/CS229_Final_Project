#!/usr/bin/env python
"""
Resume Pure MCTS from a checkpoint.

Usage:
    python resume_pure_mcts.py --checkpoint outputs/mcts/2025-12-07_01-48-42/checkpoint_step_10.pkl
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime
import multiprocessing

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

from simulation.scenario_full_mcts import run_orbital_camera_sim_full_mcts

def main():
    parser = argparse.ArgumentParser(description='Resume Pure MCTS from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config_pure_mcts.json', help='Config file')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N steps')
    args = parser.parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Get output folder from checkpoint path
    checkpoint_dir = os.path.dirname(args.checkpoint)

    # Convert initial ROE from meters to dimensionless
    mu_earth = config['orbit']['mu_earth']
    a_chief = config['orbit']['a_chief_km']
    n_chief = np.sqrt(mu_earth / (a_chief ** 3))

    initial_roe_m = np.array([
        config['initial_roe_meters']['da'],
        config['initial_roe_meters']['dl'],
        config['initial_roe_meters']['dex'],
        config['initial_roe_meters']['dey'],
        config['initial_roe_meters']['dix'],
        config['initial_roe_meters']['diy']
    ])

    initial_state_roe = initial_roe_m / np.array([a_chief*1000, a_chief*1000, 1, 1, 1, 1])

    print(f"{'='*70}")
    print(f"Resuming Pure MCTS from Checkpoint")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output folder: {checkpoint_dir}")
    print(f"{'='*70}")

    # Save config to output folder (if not already there)
    config_path_out = os.path.join(checkpoint_dir, "run_config.json")
    if not os.path.exists(config_path_out):
        with open(config_path_out, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"📝 Saved config to {config_path_out}")

    # Run simulation
    run_orbital_camera_sim_full_mcts(
        sim_config=config['simulation'],
        orbit_params=config['orbit'],
        camera_params=config['camera'],
        control_params=config['control'],
        initial_state_roe=initial_state_roe,
        out_folder=checkpoint_dir,
        mcts_params=config.get('mcts', {}),
        resume_from=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval
    )

if __name__ == "__main__":
    main()
