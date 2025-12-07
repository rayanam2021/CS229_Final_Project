# OpenMP thread management
# With 3 workers, auto thread count works fine (no OMP warnings observed)
# OMP threads will be set by the main script (run_alphazero.py or resume_training.py)
# No need to override here - let the parent script control it
import os

import numpy as np
import torch
import json
import logging
import sys
import time
import pandas as pd
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# CRITICAL FIX: Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be done before any CUDA operations
if __name__ != "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

from learning.training import SelfPlayTrainer
from learning.policy_value_network import PolicyValueNetwork
from mcts.mcts_alphazero_controller import MCTSAlphaZeroCPU
from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from camera.camera_observations import VoxelGrid, GroundTruthRSO, simulate_observation, plot_scenario, update_plot

def create_visualization_frames(out_folder, grid_initial, rso, camera_fn, camera_positions, view_directions, burn_indices, use_torch=False, device='cpu'):
    # GPU-ENABLED: Visualization grid uses same GPU settings as training
    vis_grid = VoxelGrid(
        grid_dims=grid_initial.dims,
        voxel_size=grid_initial.voxel_size,
        origin=grid_initial.origin,
        use_torch=use_torch,
        device=device
    )
    all_pos = np.vstack([camera_positions, [vis_grid.origin], [vis_grid.max_bound]])
    mid = np.mean(all_pos, axis=0)
    max_range = np.max(np.ptp(all_pos, axis=0)) / 2.0
    extent = max_range * 1.2 + 10.0

    fig, ax, artists = plot_scenario(vis_grid, rso, camera_positions[0], view_directions[0],
                                     camera_fn['fov_degrees'], camera_fn['sensor_res'])
    ax.set_xlim(mid[0]-extent, mid[0]+extent)
    ax.set_ylim(mid[1]-extent, mid[1]+extent)
    ax.set_zlim(mid[2]-extent, mid[2]+extent)
    ax.set_box_aspect([1,1,1])

    frames = []
    for frame in range(len(camera_positions)):
        simulate_observation(vis_grid, rso, camera_fn, camera_positions[frame])
        update_plot(frame, vis_grid, rso, camera_positions, view_directions,
                    camera_fn['fov_degrees'], camera_fn['sensor_res'],
                    camera_fn['noise_params'], ax, artists, np.array([0.0, 0.0, 0.0]),
                    burn_indices)
        fig.canvas.draw()
        try: buf = fig.canvas.buffer_rgba()
        except: buf = fig.canvas.renderer.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).copy().reshape((h, w, 4))[:, :, :3]
        frames.append(img)
    plt.close(fig)
    return frames

def run_episode_worker(episode_idx, config, model_state_dict, run_dir):
    """
    Worker function to run a single episode in a separate process.
    GPU-ENABLED: Now supports GPU-accelerated ray tracing.
    """
    import psutil

    # Start timing and memory tracking
    episode_start_time = time.time()
    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / 1024 / 1024

    # Robust Seeding
    seed = (int(time.time() * 1000) + episode_idx + os.getpid()) % 2**32
    np.random.seed(seed)
    torch.manual_seed(seed)

    # GPU Configuration (from config)
    use_gpu = config.get('gpu', {}).get('enable_ray_tracing', False)
    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'

    # Track GPU memory if available
    gpu_memory_start_mb = None
    gpu_memory_peak_mb = None
    if use_gpu and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gpu_memory_start_mb = torch.cuda.memory_allocated() / 1024 / 1024

    # Get thread configuration for metrics
    omp_threads = os.environ.get('OMP_NUM_THREADS', 'auto')
    omp_threads_config = config.get('gpu', {}).get('omp_threads_per_worker', 'auto')

    # Re-initialize environment for this process
    op, cp, rm = config['orbit'], config['camera'], config['initial_roe_meters']
    ctrl = config.get('control', {'lambda_dv': 0.01})
    am = op['a_chief_km'] * 1000.0

    # Base ROE & Perturbation
    base_roe = np.array([rm['da'], rm['dl'], rm['dex'], rm['dey'], rm['dix'], rm['diy']], dtype=float) / am
    b = config['monte_carlo']['perturbation_bounds']
    p = np.array([np.random.uniform(-b[k], b[k]) for k in ['da','dl','dex','dey','dix','diy']])
    initial_roe = base_roe + p/am

    # GPU-ENABLED: VoxelGrid now supports GPU acceleration
    grid = VoxelGrid(grid_dims=(20, 20, 20), use_torch=use_gpu, device=device)
    rso = GroundTruthRSO(grid)

    mdp = OrbitalMCTSModel(
        op['a_chief_km'], op['e_chief'], np.deg2rad(op['i_chief_deg']),
        np.deg2rad(op['omega_chief_deg']), np.sqrt(op['mu_earth']/op['a_chief_km']**3),
        rso, cp, grid.dims, ctrl['lambda_dv'],
        config['simulation']['time_step'], config['simulation']['max_horizon'],
        use_torch=use_gpu, device=device  # Pass GPU config to MDP
    )

    # Initial Observation (Open the eyes at t=0)
    from roe.propagation import map_roe_to_rtn
    r_init, _ = map_roe_to_rtn(initial_roe, mdp.a_chief, mdp.n_chief, f=0.0, omega=mdp.omega_chief)
    pos_init = r_init * 1000.0
    simulate_observation(grid, rso, cp, pos_init)

    # Set Initial Entropy
    initial_ent = grid.get_entropy()
    mdp.initial_entropy = initial_ent

    state = OrbitalState(initial_roe, grid, 0.0)

    # Setup Network & MCTS
    network = PolicyValueNetwork(grid_dims=grid.dims, num_actions=13, hidden_dim=128)
    network.load_state_dict(model_state_dict)
    network.to(device)  # Move network to same device as grid (CPU or CUDA)

    # OPTIMIZATION: torch.compile for 1.5-2x faster inference (PyTorch 2.0+)
    try:
        network = torch.compile(network, mode='reduce-overhead')
    except Exception:
        pass  # Fall back to eager mode if torch.compile not available

    network.eval()

    tr_cfg = config['training']
    mcts = MCTSAlphaZeroCPU(mdp, network, n_iters=tr_cfg['mcts_iters'], c_puct=tr_cfg['c_puct'], gamma=tr_cfg['gamma'], device=device)

    trajectory = []
    camera_positions = []
    view_directions = []
    burn_indices = []
    entropy_history = [initial_ent]

    camera_positions.append(pos_init)
    view_directions.append(-pos_init/np.linalg.norm(pos_init))

    sim_time = 0.0
    steps = config['simulation']['num_steps']

    # --- EPISODE LOOP ---
    for step in range(steps):
        # Progress logging (Minimal)
        if step % 10 == 0:
            print(f"[Worker Ep {episode_idx+1}] Step {step}/{steps} | Ent: {state.grid.get_entropy():.2f} | Device: {device}")

        pi, _, root_node = mcts.search(state)

        # Viz tree for first episode only
        if episode_idx == 0 and (step == 0 or step % 5 == 0):
            ep_dir = os.path.join(run_dir, f"episode_{episode_idx+1:02d}")
            os.makedirs(ep_dir, exist_ok=True)
            try:
                mcts.export_tree_to_dot(root_node, episode_idx+1, step+1, os.path.join(ep_dir, "trees"))
            except Exception: pass

        action_idx = np.random.choice(len(pi), p=pi)

        # CRITICAL: Free MCTS tree to prevent GPU memory leak
        del root_node
        if use_gpu and step % 5 == 0:  # Clear GPU cache every 5 steps
            torch.cuda.empty_cache()
        action = mdp.get_all_actions()[action_idx]

        if np.linalg.norm(action) > 1e-6:
            burn_indices.append(len(camera_positions)-1)

        next_state, reward = mdp.step(state, action)

        # Viz Data Propagation
        t_burn = np.array([state.time])
        from roe.dynamics import apply_impulsive_dv
        imp_state = apply_impulsive_dv(state.roe, action, mdp.a_chief, mdp.n_chief, t_burn, mdp.e_chief, mdp.i_chief, mdp.omega_chief)
        t_next = np.array([state.time + mdp.time_step])
        from roe.propagation import propagateGeomROE
        rho_n, _ = propagateGeomROE(imp_state, mdp.a_chief, mdp.e_chief, mdp.i_chief, mdp.omega_chief, mdp.n_chief, t_next, t0=state.time)
        pos_next = rho_n[:,0]*1000

        camera_positions.append(pos_next)
        view_directions.append(-pos_next/np.linalg.norm(pos_next))

        ent = next_state.grid.get_entropy()
        entropy_history.append(ent)

        # Convert belief to numpy for storage (if on GPU)
        belief_to_store = next_state.grid.belief
        if use_gpu:
            belief_to_store = belief_to_store.cpu().numpy()

        trajectory.append({
            'roe': state.roe,
            'belief': belief_to_store.copy(),
            'pi': pi,
            'reward': reward,
            'action': action,
            'time': sim_time,
            'next_roe': next_state.roe
        })

        state = next_state
        sim_time += mdp.time_step

    # CRITICAL: Clean up GPU memory at end of episode
    if use_gpu:
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Collect final metrics
    episode_duration = time.time() - episode_start_time
    final_memory_mb = process.memory_info().rss / 1024 / 1024
    memory_used_mb = final_memory_mb - initial_memory_mb
    peak_memory_mb = process.memory_info().rss / 1024 / 1024  # Peak RSS

    if use_gpu and torch.cuda.is_available():
        gpu_memory_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        'episode_idx': episode_idx,
        'trajectory': trajectory,
        'initial_entropy': initial_ent,
        'final_entropy': entropy_history[-1],
        'entropy_history': entropy_history,
        'camera_positions': camera_positions,
        'view_directions': view_directions,
        'burn_indices': burn_indices,
        'initial_roe': initial_roe,
        # Performance metrics
        'duration_seconds': episode_duration,
        'cpu_memory_used_mb': memory_used_mb,
        'cpu_memory_peak_mb': peak_memory_mb,
        'gpu_memory_peak_mb': gpu_memory_peak_mb,
        'omp_threads': omp_threads,
        'omp_threads_config': omp_threads_config
    }

class AlphaZeroTrainer:
    def __init__(self, config_path="config.json"):
        if not os.path.exists(config_path): raise FileNotFoundError
        with open(config_path, 'r') as f: self.config = json.load(f)

        self.base_out_dir = self.config['simulation'].get('output_dir', 'output_training')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(self.base_out_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self._setup_logging()

        # GPU Configuration
        self.use_gpu = self.config.get('gpu', {}).get('enable_ray_tracing', False)
        self.device = 'cuda' if (self.use_gpu and torch.cuda.is_available()) else 'cpu'
        self.log(f"GPU Ray Tracing: {'ENABLED' if self.use_gpu and self.device == 'cuda' else 'DISABLED'} (device: {self.device})")

        # Explicitly define grid dimensions
        self.grid_dims = (20, 20, 20)

        self.network = PolicyValueNetwork(grid_dims=self.grid_dims, num_actions=13, hidden_dim=128)

        # CRITICAL FIX: Use GPU for training when available (was hardcoded to CPU!)
        self.training_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log(f"Training Device: {self.training_device.upper()}")

        # Adjust batch size based on device (GPU can handle larger batches)
        base_batch_size = self.config['training'].get('batch_size', 64)
        if self.training_device == 'cuda':
            self.batch_size = base_batch_size * 2  # 2x larger batches on GPU
            self.log(f"GPU detected: Increasing batch size from {base_batch_size} to {self.batch_size}")
        else:
            self.batch_size = base_batch_size

        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.trainer = SelfPlayTrainer(
            network=self.network,
            learning_rate=self.config['training'].get('learning_rate', 0.001),
            weight_decay=1e-4,
            device=self.training_device,  # Use GPU when available!
            checkpoint_dir=ckpt_dir,
            max_buffer_size=self.config['training'].get('buffer_size', 10000),
            use_amp=self.training_device == 'cuda'  # Mixed precision on GPU
        )

    def _setup_logging(self):
        self.logger = logging.getLogger("AlphaZero")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(self.run_dir, "training.log"), encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

    def log(self, msg): self.logger.info(msg)

    def plot_history(self):
        """
        Create separate high-quality plots for policy, value, and total losses.
        Each loss type gets its own figure with detailed styling.
        """
        history = self.trainer.training_history
        if not history.get('total_loss'):
            self.log("No loss history to plot.")
            return

        epochs = np.arange(1, len(history['total_loss']) + 1)
        policy_loss = np.array(history['policy_loss'])
        value_loss = np.array(history['value_loss'])
        total_loss = np.array(history['total_loss'])

        # Calculate moving averages if we have enough data
        window_size = min(10, len(epochs) // 5) if len(epochs) > 10 else 0

        # ========== FIGURE 1: POLICY LOSS ==========
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, policy_loss, color='#D32F2F', linewidth=2, label='Policy Loss', alpha=0.7)

        if window_size > 0:
            moving_avg = np.convolve(policy_loss, np.ones(window_size)/window_size, mode='valid')
            ax.plot(epochs[window_size-1:], moving_avg, color='#1976D2', linewidth=2.5,
                    linestyle='--', label=f'{window_size}-Epoch Moving Average', alpha=0.9)

        ax.set_xlabel('Training Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cross-Entropy Loss', fontsize=13, fontweight='bold')
        ax.set_title('Policy Network Training Loss', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

        # Add statistics text box
        stats_text = f'Initial: {policy_loss[0]:.4f}\nFinal: {policy_loss[-1]:.4f}\nMin: {policy_loss.min():.4f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        policy_path = os.path.join(self.run_dir, "loss_policy.png")
        plt.savefig(policy_path, dpi=300, bbox_inches='tight')
        self.log(f"Policy loss plot saved to {policy_path}")
        plt.close()

        # ========== FIGURE 2: VALUE LOSS ==========
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, value_loss, color='#388E3C', linewidth=2, label='Value Loss', alpha=0.7)

        if window_size > 0:
            moving_avg = np.convolve(value_loss, np.ones(window_size)/window_size, mode='valid')
            ax.plot(epochs[window_size-1:], moving_avg, color='#1976D2', linewidth=2.5,
                    linestyle='--', label=f'{window_size}-Epoch Moving Average', alpha=0.9)

        ax.set_xlabel('Training Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Squared Error', fontsize=13, fontweight='bold')
        ax.set_title('Value Network Training Loss', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

        # Add statistics text box
        stats_text = f'Initial: {value_loss[0]:.4f}\nFinal: {value_loss[-1]:.4f}\nMin: {value_loss.min():.4f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        value_path = os.path.join(self.run_dir, "loss_value.png")
        plt.savefig(value_path, dpi=300, bbox_inches='tight')
        self.log(f"Value loss plot saved to {value_path}")
        plt.close()

        # ========== FIGURE 3: TOTAL LOSS ==========
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, total_loss, color='#1976D2', linewidth=2, label='Total Loss', alpha=0.7)

        if window_size > 0:
            moving_avg = np.convolve(total_loss, np.ones(window_size)/window_size, mode='valid')
            ax.plot(epochs[window_size-1:], moving_avg, color='#F57C00', linewidth=2.5,
                    linestyle='--', label=f'{window_size}-Epoch Moving Average', alpha=0.9)

        ax.set_xlabel('Training Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Combined Loss', fontsize=13, fontweight='bold')
        ax.set_title('Total Training Loss (Policy + Value)', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

        # Add statistics text box
        reduction_pct = ((total_loss[0] - total_loss[-1]) / total_loss[0] * 100)
        stats_text = f'Initial: {total_loss[0]:.4f}\nFinal: {total_loss[-1]:.4f}\nReduction: {reduction_pct:.1f}%'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        total_path = os.path.join(self.run_dir, "loss_total.png")
        plt.savefig(total_path, dpi=300, bbox_inches='tight')
        self.log(f"Total loss plot saved to {total_path}")
        plt.close()

        # ========== FIGURE 4: COMBINED VIEW (ALL LOSSES) ==========
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        ax.plot(epochs, policy_loss, color='#D32F2F', linewidth=2, label='Policy Loss', alpha=0.8)
        ax.plot(epochs, value_loss, color='#388E3C', linewidth=2, label='Value Loss', alpha=0.8)
        ax.plot(epochs, total_loss, color='#1976D2', linewidth=2.5, label='Total Loss', alpha=0.9)

        ax.set_xlabel('Training Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax.set_title('Training Loss Convergence (All Components)', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

        plt.tight_layout()
        combined_path = os.path.join(self.run_dir, "loss_combined.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        self.log(f"Combined loss plot saved to {combined_path}")
        plt.close()

        # Print statistics to log (ASCII-only for Windows compatibility)
        self.log("\n" + "="*60)
        self.log("TRAINING LOSS STATISTICS")
        self.log("="*60)
        self.log(f"Total epochs: {len(epochs)}")
        self.log(f"\nPolicy Loss:")
        self.log(f"  Initial: {policy_loss[0]:.6f} -> Final: {policy_loss[-1]:.6f}")
        self.log(f"  Min: {policy_loss.min():.6f} | Max: {policy_loss.max():.6f}")
        self.log(f"  Mean: {policy_loss.mean():.6f} +/- {policy_loss.std():.6f}")
        self.log(f"\nValue Loss:")
        self.log(f"  Initial: {value_loss[0]:.6f} -> Final: {value_loss[-1]:.6f}")
        self.log(f"  Min: {value_loss.min():.6f} | Max: {value_loss.max():.6f}")
        self.log(f"  Mean: {value_loss.mean():.6f} +/- {value_loss.std():.6f}")
        self.log(f"\nTotal Loss:")
        self.log(f"  Initial: {total_loss[0]:.6f} -> Final: {total_loss[-1]:.6f}")
        self.log(f"  Reduction: {reduction_pct:.2f}%")
        self.log("="*60 + "\n")

    def run_training_loop(self):
        cfg = self.config
        num_episodes = cfg['monte_carlo']['num_episodes']
        tr_cfg = cfg['training']

        num_workers = os.cpu_count() - 1
        if num_workers < 1: num_workers = 1

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

        with open(os.path.join(self.run_dir, "run_config.json"), "w") as f: json.dump(cfg, f, indent=4)
        self.log(f"STARTING TRAINING: {num_episodes} Episodes with {num_workers} parallel workers")

        # Start overall training timer
        training_start_time = time.time()
        episode_times = []
        episode_cpu_memories = []
        episode_gpu_memories = []

        episodes_completed = 0
        parallel_batch_size = num_workers
        current_workers = num_workers  # Track current worker count (can be reduced on OOM)

        while episodes_completed < num_episodes:
            current_batch_size = min(parallel_batch_size, num_episodes - episodes_completed)
            self.log(f"\n--- Spawning Batch of {current_batch_size} Episodes (Total {episodes_completed}/{num_episodes}) ---")

            current_weights = self.network.state_dict()

            # Track batch statistics
            successful_episodes = 0
            oom_errors = 0

            # Dynamic work queue: workers pull episodes as they become available
            # This prevents idle workers when some episodes finish faster than others
            futures = []
            episodes_submitted = 0
            episodes_to_submit = current_batch_size

            with ProcessPoolExecutor(max_workers=current_workers) as executor:
                # Initially submit as many episodes as we have workers (or fewer if batch is small)
                initial_submit = min(current_workers, episodes_to_submit)
                for i in range(initial_submit):
                    global_idx = episodes_completed + i
                    futures.append(
                        executor.submit(run_episode_worker, global_idx, cfg, current_weights, self.run_dir)
                    )
                    episodes_submitted += 1

                for future in as_completed(futures):
                    try:
                        res = future.result()
                        ep_idx = res['episode_idx']
                        traj = res['trajectory']

                        roe_str = np.array2string(res['initial_roe'] * self.config['orbit']['a_chief_km'] * 1000.0, precision=1, separator=', ')
                        self.log(f"Ep {ep_idx+1} Finished | Init Ent: {res['initial_entropy']:.2f} -> Final: {res['final_entropy']:.2f} | Init ROE: {roe_str}")

                        ep_dir = os.path.join(self.run_dir, f"episode_{ep_idx+1:02d}")
                        os.makedirs(ep_dir, exist_ok=True)

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

                        plt.figure(); plt.plot(res['entropy_history'], marker='o'); plt.savefig(os.path.join(ep_dir, "entropy.png")); plt.close()

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

                        if cfg['simulation'].get('visualize', True):
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

                        R = 0
                        for i in reversed(range(len(traj))):
                            t = traj[i]
                            R = t['reward'] + tr_cfg['gamma'] * R
                            self.trainer.add_to_replay_buffer(
                                t['roe'], t['belief'], t['pi'], float(R),
                                t['action'], t['reward'], t['next_roe'], t['time']
                            )

                        successful_episodes += 1

                        # Dynamic work queue: submit next episode if available
                        if episodes_submitted < episodes_to_submit:
                            next_idx = episodes_completed + episodes_submitted
                            futures.append(
                                executor.submit(run_episode_worker, next_idx, cfg, current_weights, self.run_dir)
                            )
                            episodes_submitted += 1
                            self.log(f"Worker freed - submitting episode {next_idx+1}")

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

            if len(self.trainer.replay_buffer) >= self.batch_size:
                self.log("Training Network on updated buffer...")
                train_start = time.time()
                for _ in range(tr_cfg['epochs_per_cycle']):
                    l = self.trainer.train_epoch(5, self.batch_size)
                train_time = time.time() - train_start
                self.log(f"Loss: P={l['policy_loss']:.4f} V={l['value_loss']:.4f} T={l['total_loss']:.4f} | Time: {train_time:.2f}s")

            # Only save checkpoint and advance if at least one episode succeeded
            if successful_episodes > 0:
                self.log(f"Batch completed: {successful_episodes}/{current_batch_size} episodes successful")
                self.trainer.save_checkpoint(episodes_completed + current_batch_size)
                episodes_completed += current_batch_size
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
                    self.log(f"   Retrying episodes: {episodes_completed+1} to {episodes_completed+current_batch_size}")
                    self.log(f"")
                    # Do NOT advance episodes_completed - retry same batch
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
                    episodes_completed += current_batch_size
                else:
                    # Non-OOM failure - skip batch
                    self.log(f"⚠️  Batch failed for non-OOM reasons - skipping")
                    episodes_completed += current_batch_size

        self.plot_history()

        # Save overall training summary
        training_duration = time.time() - training_start_time
        summary = {
            'total_episodes': num_episodes,
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

        with open(os.path.join(self.run_dir, "training_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)

        self.log("\n=== TRAINING SUMMARY ===")
        self.log(f"Total episodes: {num_episodes}")
        self.log(f"Total duration: {training_duration/3600:.2f} hours ({training_duration/60:.1f} minutes)")
        self.log(f"Average episode: {np.mean(episode_times) if episode_times else 0:.1f} seconds")
        self.log(f"Peak CPU memory: {np.max(episode_cpu_memories) if episode_cpu_memories else 0:.1f} MB")
        if episode_gpu_memories:
            self.log(f"Peak GPU memory: {np.max(episode_gpu_memories):.1f} MB")
        self.log(f"Workers used: {num_workers} → {current_workers}")
        self.log("Complete.")
