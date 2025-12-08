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

# OpenMP thread management
# With 3 workers, auto thread count works fine (no OMP warnings observed)
# Only limit threads if using 4+ workers to prevent memory contention
import os
import sys
import json

# Try to load thread count from run config, fallback to safe value
# For old runs without omp_threads_per_worker field, use '2' to prevent OMP warnings
# Auto threading can cause "fork while parallel region active" warnings with dynamic queue
omp_threads = '2'  # Safe default: 2 threads per worker (good performance, no warnings)
try:
    # Check if run_dir argument provided
    if '--run_dir' in sys.argv:
        run_dir_idx = sys.argv.index('--run_dir') + 1
        if run_dir_idx < len(sys.argv):
            run_dir = sys.argv[run_dir_idx]
            config_path = os.path.join(run_dir, 'run_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                omp_threads = config.get('gpu', {}).get('omp_threads_per_worker', 'auto')
except:
    pass  # Use default

if omp_threads != 'auto':
    omp_threads = str(omp_threads)
    os.environ['OMP_NUM_THREADS'] = omp_threads
    os.environ['MKL_NUM_THREADS'] = omp_threads
    os.environ['OPENBLAS_NUM_THREADS'] = omp_threads
    os.environ['NUMEXPR_NUM_THREADS'] = omp_threads

import numpy as np
import torch
import json
import logging
import sys
import time
import argparse
import glob
import multiprocessing
from datetime import datetime
from pathlib import Path

# CRITICAL: Set multiprocessing to 'spawn' mode for CUDA compatibility
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

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

    def __init__(self, run_dir: str, additional_episodes: int = None, fill_gaps: bool = True):
        """
        Initialize the trainer by loading from an existing run directory.

        Args:
            run_dir: Path to the previous run directory
            additional_episodes: Number of additional episodes to run beyond current checkpoint
                               (if None, completes remaining episodes from original config)
            fill_gaps: If True, automatically detect and fill missing episode gaps (default: True)
        """
        self.run_dir = run_dir
        self.fill_gaps = fill_gaps

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

    def _find_missing_episodes(self, up_to_episode: int):
        """
        Find missing episode directories (gaps in the sequence).

        Args:
            up_to_episode: Check for missing episodes from 1 to this episode number

        Returns:
            List of missing episode numbers
        """
        existing_episodes = set()

        # Check which episode directories exist
        for i in range(1, up_to_episode + 1):
            ep_dir = os.path.join(self.run_dir, f"episode_{i:02d}")
            if os.path.exists(ep_dir):
                # Verify it has actual data (not just empty directory)
                if os.path.exists(os.path.join(ep_dir, "episode_data.csv")):
                    existing_episodes.add(i)

        # Find gaps
        missing = []
        for i in range(1, up_to_episode + 1):
            if i not in existing_episodes:
                missing.append(i)

        return missing

    def run_resumed_training(self):
        """
        Continue training from where it left off.
        """
        # Detect missing episodes if gap filling is enabled
        missing_episodes = []
        if self.fill_gaps:
            missing_episodes = self._find_missing_episodes(self.target_episodes)
            if missing_episodes:
                self.log(f"\n⚠️  DETECTED MISSING EPISODES: {missing_episodes}")
                self.log(f"   Total missing: {len(missing_episodes)} episodes")
                self.log(f"   Will prioritize filling these gaps before continuing\n")

        # Check if there's any work to do
        if self.starting_episode >= self.target_episodes and not missing_episodes:
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

        # GPU Memory Management: Limit workers if using GPU ray tracing
        # Auto-detects GPU memory and calculates safe worker count
        if self.use_gpu and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Memory allocation (based on profiling):
            # - Network training: 500 MB (conservative buffer)
            # - Per worker: 1800 MB (network + MCTS tree + ray tracing buffers + CUDA overhead + auto thread overhead)
            #   Note: Workers use 960-1020 MB observed (CUDA fragmentation grows during MCTS)
            #   With OMP_NUM_THREADS=auto (~12 threads on 12-core CPU), higher thread overhead
            #   1800 MB accounts for auto thread overhead, worst-case fragmentation, and PyTorch memory pooling
            # These values are tuned for RTX 2060 6GB (yields 3 workers); adjust if needed
            reserve_mb = 500  # MB to reserve for main process (network training)
            worker_mb = 1800  # MB per parallel episode worker (targets 3 workers on 6GB GPU with auto threads)

            max_gpu_workers = int((gpu_memory_gb * 1024 - reserve_mb) / worker_mb)
            if num_workers > max_gpu_workers:
                self.log(f"⚠️  GPU Memory Limit: Reducing workers from {num_workers} to {max_gpu_workers}")
                self.log(f"   GPU: {gpu_memory_gb:.1f} GB | ~1300 MB per worker (conservative allocation)")
                num_workers = max(1, max_gpu_workers)

        # Determine which episodes to run
        if missing_episodes:
            # Fill gaps first
            episodes_to_run_list = missing_episodes
            self.log(f"\nFILLING MISSING EPISODES:")
            self.log(f"  Episodes to fill: {episodes_to_run_list}")
            self.log(f"  Total count: {len(episodes_to_run_list)}")
        else:
            # Normal sequential execution
            episodes_to_run_list = list(range(self.starting_episode, self.target_episodes))
            self.log(f"\nCONTINUING TRAINING:")
            self.log(f"  Starting from: Episode {self.starting_episode}")
            self.log(f"  Target:        Episode {self.target_episodes}")
            self.log(f"  Episodes to run: {len(episodes_to_run_list)}")

        self.log(f"  Workers: {num_workers} parallel")
        self.log(f"  Network restored from checkpoint")
        self.log(f"  Starting with fresh replay buffer (will populate from new episodes)\n")

        parallel_batch_size = num_workers
        episodes_run_index = 0  # Index in episodes_to_run_list
        current_workers = num_workers  # Track current worker count (can be reduced on OOM)
        min_workers = 1  # Minimum workers before giving up

        # Start overall training timer
        training_start_time = time.time()
        episode_times = []
        episode_cpu_memories = []
        episode_gpu_memories = []

        while episodes_run_index < len(episodes_to_run_list):
            current_batch_size = min(parallel_batch_size, len(episodes_to_run_list) - episodes_run_index)
            batch_episodes = episodes_to_run_list[episodes_run_index:episodes_run_index + current_batch_size]
            self.log(f"\n--- Spawning Batch: Episodes {batch_episodes} ({episodes_run_index + current_batch_size}/{len(episodes_to_run_list)} total) ---")

            current_weights = self.network.state_dict()
            successful_episodes = 0  # Track successful episodes in this batch
            oom_errors = 0  # Track OOM errors in this batch

            # Static batch submission: submit all episodes at once
            with ProcessPoolExecutor(max_workers=current_workers) as executor:
                futures = [
                    executor.submit(run_episode_worker, ep_num - 1, cfg, current_weights, self.run_dir)
                    for ep_num in batch_episodes
                ]

                for future in as_completed(futures):
                    try:
                        res = future.result()
                        ep_idx = res['episode_idx']
                        traj = res['trajectory']
                        successful_episodes += 1  # Count successful episode

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

                        # Save performance metrics
                        metrics = {
                            'episode': ep_idx + 1,
                            'duration_seconds': res['duration_seconds'],
                            'duration_minutes': res['duration_seconds'] / 60,
                            'cpu_memory_used_mb': res['cpu_memory_used_mb'],
                            'cpu_memory_peak_mb': res['cpu_memory_peak_mb'],
                            'gpu_memory_peak_mb': res['gpu_memory_peak_mb'] if res['gpu_memory_peak_mb'] else 'N/A',
                            'initial_entropy': res['initial_entropy'],
                            'final_entropy': res['final_entropy'],
                            'entropy_reduction': res['initial_entropy'] - res['final_entropy'],
                            'omp_threads_env': res.get('omp_threads', 'auto'),
                            'omp_threads_config': res.get('omp_threads_config', 'auto')
                        }
                        with open(os.path.join(ep_dir, "metrics.json"), 'w') as f:
                            json.dump(metrics, f, indent=4)

                        # Collect for overall summary
                        episode_times.append(res['duration_seconds'])
                        episode_cpu_memories.append(res['cpu_memory_peak_mb'])
                        if res['gpu_memory_peak_mb']:
                            episode_gpu_memories.append(res['gpu_memory_peak_mb'])

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
                        # Check if it's an OOM error
                        error_str = str(e)
                        if "CUDA out of memory" in error_str or "OutOfMemoryError" in error_str:
                            oom_errors += 1
                            self.log(f"Worker failed with OOM: {e}")
                        else:
                            self.log(f"Worker failed: {e}")
                            import traceback
                            traceback.print_exc()

            # Train the network
            if len(self.trainer.replay_buffer) >= tr_cfg['batch_size']:
                self.log("Training Network on updated buffer...")
                for _ in range(tr_cfg['epochs_per_cycle']):
                    l = self.trainer.train_epoch(5, tr_cfg['batch_size'])
                self.log(f"Loss: P={l['policy_loss']:.4f} V={l['value_loss']:.4f} T={l['total_loss']:.4f}")

            # Only save checkpoint if at least one episode succeeded (avoid duplicate checkpoints)
            if successful_episodes > 0:
                self.log(f"Batch completed: {successful_episodes}/{current_batch_size} episodes successful")
                # Save checkpoint with the highest episode number in this batch
                max_ep_in_batch = max(batch_episodes)
                self.trainer.save_checkpoint(max_ep_in_batch)
                # Batch succeeded - advance to next batch
                episodes_run_index += current_batch_size
            else:
                self.log(f"⚠️  Batch failed: 0/{current_batch_size} episodes successful - skipping checkpoint (no new data)")

                # Check if failure was due to OOM and if we can reduce workers
                if oom_errors > 0 and current_workers > 1:
                    # Reduce worker count and retry
                    old_workers = current_workers
                    current_workers = max(1, current_workers - 2)  # Reduce by 2, minimum 1
                    self.log(f"")
                    self.log(f"🔄 RETRYING BATCH WITH REDUCED WORKERS:")
                    self.log(f"   OOM errors detected: {oom_errors}/{current_batch_size} workers")
                    self.log(f"   Reducing workers: {old_workers} → {current_workers}")
                    self.log(f"   Retrying episodes: {batch_episodes}")
                    self.log(f"")
                    # Do NOT advance episodes_run_index - retry same batch
                elif oom_errors > 0 and current_workers == 1:
                    # Already at minimum workers - cannot reduce further
                    self.log(f"")
                    self.log(f"❌ FATAL: OOM with single worker - cannot reduce further")
                    self.log(f"   Consider:")
                    self.log(f"   1. Reducing MCTS iterations (currently {tr_cfg['mcts_iters']})")
                    self.log(f"   2. Reducing grid resolution")
                    self.log(f"   3. Using smaller network architecture")
                    self.log(f"")
                    # Skip this batch and move on
                    episodes_run_index += current_batch_size
                else:
                    # Non-OOM failure - skip batch
                    self.log(f"⚠️  Batch failed for non-OOM reasons - skipping")
                    episodes_run_index += current_batch_size

        # Plot final training history (uses improved base class method)
        self.plot_history()

        # Save overall training summary
        training_duration = time.time() - training_start_time
        summary = {
            'resumed_episodes': len(episodes_to_run_list),
            'total_duration_seconds': training_duration,
            'total_duration_minutes': training_duration / 60,
            'total_duration_hours': training_duration / 3600,
            'average_episode_duration_seconds': np.mean(episode_times) if episode_times else 0,
            'min_episode_duration_seconds': np.min(episode_times) if episode_times else 0,
            'max_episode_duration_seconds': np.max(episode_times) if episode_times else 0,
            'average_cpu_memory_mb': np.mean(episode_cpu_memories) if episode_cpu_memories else 0,
            'peak_cpu_memory_mb': np.max(episode_cpu_memories) if episode_cpu_memories else 0,
            'average_gpu_memory_mb': np.mean(episode_gpu_memories) if episode_gpu_memories else 'N/A',
            'peak_gpu_memory_mb': np.max(episode_gpu_memories) if episode_gpu_memories else 'N/A',
            'final_workers': current_workers,
            'initial_workers': num_workers,
            'omp_threads_per_worker': cfg.get('gpu', {}).get('omp_threads_per_worker', 'auto')
        }

        with open(os.path.join(self.run_dir, "resume_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)

        self.log("\n=== RESUMED TRAINING SUMMARY ===")
        self.log(f"Episodes run: {len(episodes_to_run_list)}")
        self.log(f"Total duration: {training_duration/3600:.2f} hours ({training_duration/60:.1f} minutes)")
        if episode_times:
            self.log(f"Average episode: {np.mean(episode_times):.1f} seconds")
        if episode_cpu_memories:
            self.log(f"Peak CPU memory: {np.max(episode_cpu_memories):.1f} MB")
        if episode_gpu_memories:
            self.log(f"Peak GPU memory: {np.max(episode_gpu_memories):.1f} MB")
        self.log(f"Workers used: {num_workers} → {current_workers}")

        self.log("\nRESUMED TRAINING COMPLETE!")

        # Report what was accomplished
        if missing_episodes:
            final_missing = self._find_missing_episodes(self.target_episodes)
            if final_missing:
                self.log(f"Filled: {len(missing_episodes) - len(final_missing)}/{len(missing_episodes)} missing episodes")
                self.log(f"Still missing: {final_missing}")
            else:
                self.log(f"Successfully filled all {len(missing_episodes)} missing episodes!")
        else:
            self.log(f"Completed episodes: {self.starting_episode} → {self.target_episodes}")

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

    parser.add_argument(
        '--no-fill-gaps',
        action='store_true',
        help='Disable automatic detection and filling of missing episode gaps (default: gaps are filled)'
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
    if args.no_fill_gaps:
        print(f"Gap filling disabled - will only run new episodes")
    else:
        print(f"Gap filling enabled - will detect and fill missing episodes first")

    trainer = ResumeAlphaZeroTrainer(args.run_dir, args.additional_episodes, fill_gaps=not args.no_fill_gaps)
    trainer.run_resumed_training()


if __name__ == "__main__":
    main()
