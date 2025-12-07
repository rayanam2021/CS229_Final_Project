#!/usr/bin/env python3
"""
Resume Training Script for AlphaZero Orbital MCTS (GPU-ENABLED)

This script allows resuming training from a previous run that was interrupted.
It loads the run configuration, latest checkpoint, and continues training.

Usage:
    python resume_training.py --run_dir <path_to_run_directory> [--additional_episodes N]

Examples:
    # Resume and complete the original training plan (if interrupted)
    python resume_training.py --run_dir outputs/training/run_2025-12-04_11-08-29

    # Continue training for 65 MORE episodes (beyond what was originally planned)
    python resume_training.py --run_dir outputs/training/run_2025-12-04_11-08-29 --additional_episodes 65
"""

import numpy as np
import torch
import os
import json
import logging
import sys
import time
import argparse
import glob
from datetime import datetime
from pathlib import Path

# Import local modules
from learning.training import SelfPlayTrainer
from learning.training_loop import run_episode_worker, AlphaZeroTrainer
from learning.policy_value_network import PolicyValueNetwork
from concurrent.futures import ProcessPoolExecutor, as_completed


class ResumeAlphaZeroTrainer(AlphaZeroTrainer):
    """
    Extended AlphaZeroTrainer that can resume from a checkpoint.
    GPU-ENABLED: Supports GPU-accelerated ray tracing.
    Inherits improved loss plotting from base class.
    """

    def __init__(self, run_dir: str, additional_episodes: int = None):
        """
        Initialize the trainer by loading from an existing run directory.

        Args:
            run_dir: Path to the previous run directory
            additional_episodes: Number of additional episodes to run beyond current checkpoint
                               (if None, completes remaining episodes from original config)
        """
        self.run_dir = run_dir

        # Load the run configuration
        config_path = os.path.join(run_dir, "run_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Setup logging to append to existing log file
        self._setup_logging_resume()

        # GPU Configuration
        self.use_gpu = self.config.get('gpu', {}).get('enable_ray_tracing', False)
        self.device = 'cuda' if (self.use_gpu and torch.cuda.is_available()) else 'cpu'
        self.log(f"GPU Ray Tracing: {'ENABLED' if self.use_gpu and self.device == 'cuda' else 'DISABLED'} (device: {self.device})")

        # Explicitly define grid dimensions
        self.grid_dims = (20, 20, 20)

        # Initialize network
        self.network = PolicyValueNetwork(grid_dims=self.grid_dims, num_actions=13, hidden_dim=128)

        # Setup trainer
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.trainer = SelfPlayTrainer(
            network=self.network,
            learning_rate=self.config['training'].get('learning_rate', 0.001),
            weight_decay=1e-4,
            device='cpu',
            checkpoint_dir=ckpt_dir,
            max_buffer_size=self.config['training'].get('buffer_size', 10000)
        )

        # Load the latest checkpoint
        self.starting_episode = self._load_latest_checkpoint()

        # Determine how many episodes to run
        total_episodes_in_config = self.config['monte_carlo']['num_episodes']

        if additional_episodes is not None:
            # User specified additional episodes - continue beyond current checkpoint
            self.target_episodes = self.starting_episode + additional_episodes
            self.log(f"Will run {additional_episodes} ADDITIONAL episodes")
            self.log(f"Starting from episode {self.starting_episode} -> Target episode {self.target_episodes}")
        else:
            # No additional episodes specified - complete the original training plan
            self.target_episodes = total_episodes_in_config
            remaining = total_episodes_in_config - self.starting_episode

            if remaining > 0:
                self.log(f"Will run REMAINING {remaining} episodes from original plan")
                self.log(f"Starting from episode {self.starting_episode} -> Target episode {self.target_episodes}")
            else:
                self.log(f"Original training plan already complete ({self.starting_episode}/{total_episodes_in_config} episodes)")
                self.log(f"Use --additional_episodes N to continue training further")

    def _setup_logging_resume(self):
        """Setup logging to append to existing log file with UTF-8 encoding."""
        self.logger = logging.getLogger("AlphaZero_Resume")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Append to existing log file with UTF-8 encoding (fixes Windows Unicode errors)
        log_path = os.path.join(self.run_dir, "training.log")
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

        # Log resume marker
        self.log("\n" + "="*80)
        self.log(f"RESUMING TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("="*80 + "\n")

    def _load_latest_checkpoint(self) -> int:
        """
        Load the latest checkpoint from the checkpoints directory.

        Returns:
            The episode number of the loaded checkpoint
        """
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")

        # Find all checkpoint files
        checkpoint_files = glob.glob(os.path.join(ckpt_dir, "checkpoint_ep_*.pt"))

        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

        # Extract episode numbers and find the latest
        episode_numbers = []
        for ckpt_file in checkpoint_files:
            basename = os.path.basename(ckpt_file)
            # Extract number from "checkpoint_ep_39.pt"
            try:
                ep_num = int(basename.split('_')[-1].split('.')[0])
                episode_numbers.append((ep_num, ckpt_file))
            except:
                continue

        if not episode_numbers:
            raise ValueError("Could not parse episode numbers from checkpoint files")

        # Sort by episode number and get the latest
        episode_numbers.sort(key=lambda x: x[0])
        latest_episode, latest_checkpoint = episode_numbers[-1]

        self.log(f"Loading checkpoint from episode {latest_episode}: {latest_checkpoint}")

        # Load the checkpoint
        self.trainer.load_checkpoint(latest_checkpoint)

        # Log training history
        history = self.trainer.training_history
        if history.get('total_loss'):
            n_epochs = len(history['total_loss'])
            recent_losses = history['total_loss'][-3:] if n_epochs >= 3 else history['total_loss']
            self.log(f"Restored training history: {n_epochs} epochs")
            self.log(f"Recent total losses: {[f'{l:.4f}' for l in recent_losses]}")

        return latest_episode

    def run_resumed_training(self):
        """
        Continue training from where it left off.
        """
        # Check if there's any work to do
        if self.starting_episode >= self.target_episodes:
            self.log("\nNo episodes to run! Training already at or beyond target.")
            self.log(f"Current: {self.starting_episode} episodes | Target: {self.target_episodes} episodes")
            self.log("\nTo continue training, use: --additional_episodes N")
            self.log("Example: python resume_training.py --run_dir ... --additional_episodes 65")

            # Still plot the current training history
            self.log("\nGenerating loss plots from current training history...")
            self.plot_history()
            return

        cfg = self.config
        tr_cfg = cfg['training']

        num_workers = os.cpu_count() - 1
        if num_workers < 1:
            num_workers = 1

        episodes_to_run = self.target_episodes - self.starting_episode
        self.log(f"\nCONTINUING TRAINING:")
        self.log(f"  Starting from: Episode {self.starting_episode}")
        self.log(f"  Target:        Episode {self.target_episodes}")
        self.log(f"  Episodes to run: {episodes_to_run}")
        self.log(f"  Workers: {num_workers} parallel")
        self.log(f"  Network restored from checkpoint")
        self.log(f"  Starting with fresh replay buffer (will populate from new episodes)\n")

        episodes_completed = self.starting_episode
        parallel_batch_size = num_workers

        while episodes_completed < self.target_episodes:
            current_batch_size = min(parallel_batch_size, self.target_episodes - episodes_completed)
            self.log(f"\n--- Spawning Batch of {current_batch_size} Episodes (Progress: {episodes_completed}/{self.target_episodes}) ---")

            current_weights = self.network.state_dict()

            futures = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for i in range(current_batch_size):
                    global_idx = episodes_completed + i
                    futures.append(
                        executor.submit(run_episode_worker, global_idx, cfg, current_weights, self.run_dir)
                    )

                for future in as_completed(futures):
                    try:
                        res = future.result()
                        ep_idx = res['episode_idx']
                        traj = res['trajectory']

                        roe_str = np.array2string(res['initial_roe'] * self.config['orbit']['a_chief_km'] * 1000.0, precision=1, separator=', ')
                        self.log(f"Ep {ep_idx+1} Finished | Init Ent: {res['initial_entropy']:.2f} -> Final: {res['final_entropy']:.2f} | Init ROE: {roe_str}")

                        # Episode directory and data saving
                        ep_dir = os.path.join(self.run_dir, f"episode_{ep_idx+1:02d}")
                        os.makedirs(ep_dir, exist_ok=True)

                        # Save episode data
                        import pandas as pd
                        data_rows = []
                        for i, t in enumerate(traj):
                            row = {
                                'time': t['time'],
                                'step': i + 1,
                                'action': t['action'].tolist(),
                                'reward': t['reward'],
                                'entropy': res['entropy_history'][i+1],
                                'state': t['roe'].tolist(),
                                'next_state': t['next_roe'].tolist()
                            }
                            data_rows.append(row)
                        pd.DataFrame(data_rows).to_csv(os.path.join(ep_dir, "episode_data.csv"), index=False)

                        # Save entropy plot
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        plt.figure()
                        plt.plot(res['entropy_history'], marker='o')
                        plt.xlabel('Step')
                        plt.ylabel('Entropy')
                        plt.title(f'Episode {ep_idx+1} Entropy Progression')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(ep_dir, "entropy.png"))
                        plt.close()

                        # Save visualization if enabled (GPU-ENABLED)
                        if cfg['simulation'].get('visualize', True):
                            from camera.camera_observations import VoxelGrid, GroundTruthRSO
                            from learning.training_loop import create_visualization_frames
                            import imageio

                            frames = create_visualization_frames(
                                ep_dir,
                                VoxelGrid(self.grid_dims, use_torch=self.use_gpu, device=self.device),
                                GroundTruthRSO(VoxelGrid(self.grid_dims, use_torch=self.use_gpu, device=self.device)),
                                self.config['camera'],
                                np.array(res['camera_positions']),
                                np.array(res['view_directions']),
                                res['burn_indices'],
                                use_torch=self.use_gpu,
                                device=self.device
                            )
                            if frames:
                                imageio.mimsave(os.path.join(ep_dir, "video.mp4"), frames, fps=5, macro_block_size=1)
                                imageio.imwrite(os.path.join(ep_dir, "final_frame.png"), frames[-1])

                        # Add to replay buffer with discounted returns
                        R = 0
                        for i in reversed(range(len(traj))):
                            t = traj[i]
                            R = t['reward'] + tr_cfg['gamma'] * R
                            self.trainer.add_to_replay_buffer(
                                t['roe'], t['belief'], t['pi'], float(R),
                                t['action'], t['reward'], t['next_roe'], t['time']
                            )

                    except Exception as e:
                        self.log(f"Worker failed: {e}")
                        import traceback
                        traceback.print_exc()

            # Train the network
            if len(self.trainer.replay_buffer) >= tr_cfg['batch_size']:
                self.log("Training Network on updated buffer...")
                for _ in range(tr_cfg['epochs_per_cycle']):
                    l = self.trainer.train_epoch(5, tr_cfg['batch_size'])
                self.log(f"Loss: P={l['policy_loss']:.4f} V={l['value_loss']:.4f} T={l['total_loss']:.4f}")

            # Save checkpoint
            self.trainer.save_checkpoint(episodes_completed + current_batch_size)
            episodes_completed += current_batch_size

        # Plot final training history (uses improved base class method)
        self.plot_history()
        self.log("\nRESUMED TRAINING COMPLETE!")
        self.log(f"Total episodes completed: {episodes_completed}")
        self.log(f"Checkpoints saved in: {os.path.join(self.run_dir, 'checkpoints')}")


def main():
    parser = argparse.ArgumentParser(
        description='Resume AlphaZero training from a previous run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Resume and complete the original training plan (if interrupted early)
    python resume_training.py --run_dir output_training/run_2025-12-04_11-08-29

    # Continue training for 65 MORE episodes (beyond original plan)
    python resume_training.py --run_dir output_training/run_2025-12-04_11-08-29 --additional_episodes 65

    # Continue training for 130 MORE episodes
    python resume_training.py --run_dir output_training/run_2025-12-04_11-08-29 --additional_episodes 130

Note:
    - Without --additional_episodes, script completes remaining episodes from original config
    - With --additional_episodes N, script runs N MORE episodes beyond current checkpoint
        """
    )

    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='Path to the run directory containing checkpoints and run_config.json'
    )

    parser.add_argument(
        '--additional_episodes',
        type=int,
        default=None,
        help='Number of ADDITIONAL episodes to run beyond current checkpoint (e.g., 65 for another 65 episodes)'
    )

    args = parser.parse_args()

    # Validate run directory
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory does not exist: {args.run_dir}")
        sys.exit(1)

    if not os.path.exists(os.path.join(args.run_dir, "run_config.json")):
        print(f"Error: run_config.json not found in {args.run_dir}")
        sys.exit(1)

    if not os.path.exists(os.path.join(args.run_dir, "checkpoints")):
        print(f"Error: checkpoints directory not found in {args.run_dir}")
        sys.exit(1)

    # Initialize and run
    print(f"Initializing resume training from: {args.run_dir}")
    if args.additional_episodes:
        print(f"Will train for {args.additional_episodes} ADDITIONAL episodes")
    trainer = ResumeAlphaZeroTrainer(args.run_dir, args.additional_episodes)
    trainer.run_resumed_training()


if __name__ == "__main__":
    main()
