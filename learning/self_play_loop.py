"""
Self-play learning loop combining MCTS planning with neural network training.

Implements the AlphaZero-style alternating loop:
1. MCTS Planning: Generate training data using MCTS guided by network
2. Network Training: Update policy and value heads on collected data
3. Bootstrap Transition: Gradually shift from rollouts to network predictions

The loop maintains:
- Replay buffer of (state, policy_MCTS, return) tuples
- Network checkpoints for resuming training
- Performance tracking across iterations
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime
import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from learning.policy_value_network import PolicyValueNetwork
from learning.training import SelfPlayTrainer
from mcts.mcts_alphazero import MCTSAlphaZero
from mcts.orbital_mdp_model_gpu import OrbitalMCTSModelGPU, OrbitalState
from camera.camera_observations_gpu import VoxelGrid


class SelfPlayLoop:
    """
    AlphaZero-style self-play learning combining MCTS and network training.

    Process:
    1. Initialize network with random weights
    2. Repeat for N iterations:
       a. MCTS Planning: Generate episode data with MCTS
       b. Add to replay buffer
       c. Network Training: Train on buffer
       d. Evaluate: Check if bootstrapping should activate
       e. Save checkpoint
    """

    def __init__(
        self,
        mdp_model: OrbitalMCTSModelGPU,
        grid: VoxelGrid,
        network: Optional[nn.Module] = None,
        output_dir: str = "./self_play_outputs",
        device: str = "cuda",
        seed: int = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize self-play loop.

        Args:
            mdp_model: MDP model for planning
            grid: Initial voxel grid for belief representation
            network: Policy-value network (creates if None)
            output_dir: Directory for checkpoints and logs
            device: "cuda" or "cpu"
            seed: Random seed for reproducibility
            config: Configuration dictionary with training parameters
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.mdp_model = mdp_model
        self.grid = grid
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize or use provided network
        if network is None:
            self.network = PolicyValueNetwork(
                grid_dims=grid.dims,
                hidden_dim=256,
                num_actions=13,
                num_residual_blocks=3,
                use_batch_norm=True,
                dropout_rate=0.1,
            ).to(device)
        else:
            self.network = network.to(device)

        # Extract training config
        training_cfg = self.config.get("training", {})
        learning_rate = training_cfg.get("learning_rate", 1e-3)
        weight_decay = training_cfg.get("weight_decay", 1e-4)

        # Initialize trainer
        self.trainer = SelfPlayTrainer(
            network=self.network,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            checkpoint_dir=str(self.output_dir / "checkpoints"),
            max_buffer_size=training_cfg.get("replay_buffer_size", 100_000),
            gradient_clip_norm=training_cfg.get("gradient_clip_norm", 1.0),
        )

        # Training history
        self.history = {
            "iteration": [],
            "planning_time": [],
            "training_time": [],
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
            "network_uses": [],
            "buffer_size": [],
        }

        self.logger.info("Self-play loop initialized")
        self.logger.info(f"Network: {self.network.__class__.__name__}")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging to file and console."""
        logger = logging.getLogger("SelfPlayLoop")
        logger.setLevel(logging.INFO)

        # File handler
        log_file = self.output_dir / "self_play.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _generate_episode(
        self,
        initial_state: OrbitalState,
        mcts_iters: int = 100,
        use_network_bootstrap: bool = False,
        bootstrap_threshold: float = 0.8,
        save_trees: bool = False,
    ) -> Tuple[List[Tuple], Dict]:
        """
        Generate one episode of data using MCTS planning.

        Args:
            initial_state: Starting state for episode
            mcts_iters: Number of MCTS iterations per step
            use_network_bootstrap: Enable network value bootstrapping
            bootstrap_threshold: Confidence threshold for bootstrapping
            save_trees: Whether to save MCTS tree visualizations

        Returns:
            List of (state, policy_MCTS, return) tuples
            Dictionary with episode statistics
        """
        import time

        start_time = time.time()

        # Create GPU-optimized AlphaZero MCTS with PUCT formula and policy guidance
        # Network is always used for value bootstrapping (efficient evaluation)
        mcts = MCTSAlphaZero(
            model=self.mdp_model,
            network=self.network,
            use_policy_guidance=True,
            bootstrap_threshold=bootstrap_threshold,
            min_bootstrap_depth=1,
            blend_mode="linear",
            iters=mcts_iters,
            max_depth=5,
            c=1.4,
            gamma=0.95,
            batch_rollouts=True,
            batch_size=10,
            parallel_batching=True,
            device=self.device,
        )

        episode_data = []
        state = initial_state
        cumulative_return = 0.0
        gamma = self.mdp_model.gamma if hasattr(self.mdp_model, 'gamma') else 0.95
        discount = 1.0

        step = 0
        max_steps = self.mdp_model.max_depth  # Episode length from config horizon

        while step < max_steps:
            # MCTS planning
            out_folder = str(self.output_dir / "trees") if save_trees else None
            best_action, best_value, stats = mcts.get_best_root_action(
                state, step, out_folder, return_stats=True
            )

            # Get MCTS policy (visit counts normalized)
            root_visits = stats["root_N_sa"].astype(float)
            policy_mcts = root_visits / (root_visits.sum() + 1e-10)

            # Execute action
            next_state, reward = self.mdp_model.step(state, best_action)

            # Store trajectory with actual reward (not step index!)
            episode_data.append((state, policy_mcts, best_value, reward))

            # Update returns
            cumulative_return += discount * reward
            discount *= gamma

            # Move to next state
            state = next_state
            step += 1

            # Get bootstrap statistics from MCTS
            if hasattr(mcts, 'get_bootstrap_stats'):
                bootstrap_stats = mcts.get_bootstrap_stats()
                self.logger.info(
                    f"Step {step}: Network={bootstrap_stats['network_fraction']:.2%}, "
                    f"Rollout={bootstrap_stats['rollout_fraction']:.2%}, "
                    f"Blend={bootstrap_stats['blend_fraction']:.2%}"
                )

        elapsed = time.time() - start_time

        # Convert episode data to training format (state, policy, return)
        training_data = []
        for step_i, (state, policy_mcts, _, _) in enumerate(episode_data):
            # Compute discounted return from this point onward: R_t = sum_k gamma^k * r_{t+k}
            R = sum(
                gamma ** (j - step_i) * episode_data[j][3]
                for j in range(step_i, len(episode_data))
            )
            training_data.append((state, policy_mcts, R))

        stats = {
            "episode_return": cumulative_return,
            "episode_length": step,
            "episode_time": elapsed,
            "network_bootstrapped": use_network_bootstrap,
        }

        return training_data, stats

    def _add_episode_to_buffer(self, episode_data: List[Tuple]):
        """Add episode data to replay buffer."""
        for state, policy_mcts, episode_return in episode_data:
            self.trainer.add_to_replay_buffer(
                orbital_state=state.roe,
                belief_grid=state.grid.belief,
                policy_mcts=policy_mcts,
                episode_return=episode_return,
            )

    def _train_network(
        self,
        num_epochs: int = 10,
        batch_size: int = 32,
        num_batches_per_epoch: int = 100,
        value_weight: float = 1.0,
        policy_weight: float = 1.0,
    ) -> Dict[str, float]:
        """
        Train network on replay buffer.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            num_batches_per_epoch: Number of batches per epoch
            value_weight: Weight for value loss
            policy_weight: Weight for policy loss

        Returns:
            Dictionary with average losses
        """
        import time

        start_time = time.time()

        avg_losses_all = {
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
        }

        for epoch in range(num_epochs):
            avg_losses = self.trainer.train_epoch(
                num_batches=num_batches_per_epoch,
                batch_size=batch_size,
                value_weight=value_weight,
                policy_weight=policy_weight,
            )

            for key in avg_losses:
                avg_losses_all[key].append(avg_losses[key])

            self.trainer.step_scheduler()

            msg = (
                f"Epoch {epoch+1}/{num_epochs}: "
                f"policy_loss={avg_losses['policy_loss']:.4f}, "
                f"value_loss={avg_losses['value_loss']:.4f}, "
                f"total_loss={avg_losses['total_loss']:.4f}"
            )
            self.logger.info(msg)
            print(msg)  # Also print to console for real-time visibility

        # Compute epoch averages
        final_losses = {
            k: np.mean(v) for k, v in avg_losses_all.items()
        }

        elapsed = time.time() - start_time
        final_losses["training_time"] = elapsed

        return final_losses

    def _should_enable_bootstrapping(self, iteration: int) -> bool:
        """
        Determine if network value bootstrapping should be enabled.

        Generally enable after network has seen sufficient data:
        - After 50% of planned iterations
        - When value loss is below threshold
        - When policy loss has stabilized
        """
        # Simple heuristic: enable after 5 iterations
        return iteration >= 5

    def _generate_loss_plot(self) -> Path:
        """
        Generate PNG plot of training loss curves.

        Returns:
            Path to saved PNG file
        """
        if not self.history["iteration"]:
            self.logger.warning("No training history available for plot")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("AlphaZero Training Metrics", fontsize=16, fontweight='bold')

        # Plot 1: Loss curves
        ax = axes[0, 0]
        iterations = self.history["iteration"]
        ax.plot(iterations, self.history["policy_loss"], 'o-', label='Policy Loss', linewidth=2)
        ax.plot(iterations, self.history["value_loss"], 's-', label='Value Loss', linewidth=2)
        ax.plot(iterations, self.history["total_loss"], '^-', label='Total Loss', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Iterations')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Policy loss only
        ax = axes[0, 1]
        ax.plot(iterations, self.history["policy_loss"], 'o-', color='tab:blue', linewidth=2)
        ax.fill_between(iterations, self.history["policy_loss"], alpha=0.3, color='tab:blue')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Policy Loss (Cross-Entropy)')
        ax.grid(True, alpha=0.3)

        # Plot 3: Value loss only
        ax = axes[1, 0]
        ax.plot(iterations, self.history["value_loss"], 's-', color='tab:orange', linewidth=2)
        ax.fill_between(iterations, self.history["value_loss"], alpha=0.3, color='tab:orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Value Loss (MSE)')
        ax.grid(True, alpha=0.3)

        # Plot 4: Replay buffer size
        ax = axes[1, 1]
        ax.bar(iterations, self.history["buffer_size"], color='tab:green', alpha=0.7, width=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Replay Buffer Growth')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "training_metrics.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def _generate_orbital_video(self, initial_state: OrbitalState) -> Optional[Path]:
        """
        Generate two 3D orbital trajectory videos:
        1. belief_state_evolution.mp4 - Simple 3D voxel grid probabilities
        2. orbital_trajectory_scenario.mp4 - Full orbital scenario with RSO, camera path, and grid

        Returns:
            Path to saved video file, or None if generation failed
        """
        try:
            import imageio
        except ImportError:
            self.logger.warning("imageio not available, skipping orbital video generation")
            return None

        try:
            self.logger.info("Generating orbital trajectory videos...")

            # Run a quick evaluation episode to get trajectory
            from mcts.mcts_alphazero import MCTSAlphaZero

            mcts = MCTSAlphaZero(
                model=self.mdp_model,
                network=self.network,
                use_policy_guidance=True,
                iters=50,  # Fewer iterations for speed
                max_depth=5,
                c=1.4,
                gamma=0.95,
                batch_rollouts=True,
                batch_size=10,
                parallel_batching=True,
                device=self.device,
            )

            state = initial_state
            trajectory_states = [state]
            trajectory_actions = []

            # Run episode and collect states
            for step in range(5):
                best_action, _, _ = mcts.get_best_root_action(state, step, str(self.output_dir), return_stats=True)
                trajectory_actions.append(best_action)
                next_state, _ = self.mdp_model.step(state, best_action)
                trajectory_states.append(next_state)
                state = next_state

            # --- VIDEO: Full Orbital Scenario ---
            self.logger.info("Generating full orbital scenario video...")
            try:
                from camera.camera_observations_gpu import GroundTruthRSO, draw_spacecraft, simulate_observation

                # Create RSO with same configuration
                rso = GroundTruthRSO(grid=self.grid)
                scenario_frames = []

                # Save belief copy to restore after video generation
                belief_copy = self.grid.belief.detach().clone() if isinstance(self.grid.belief, torch.Tensor) else self.grid.belief.copy()

                # Reset belief to uniform distribution for visualization
                if isinstance(self.grid.belief, torch.Tensor):
                    self.grid.belief[:] = 1.0 / self.grid.belief.numel()
                else:
                    self.grid.belief[:] = 1.0 / self.grid.belief.size

                # Generate camera trajectory (circular orbit)
                camera_positions = []
                view_directions = []
                num_frames = len(trajectory_states)  # One frame per step in trajectory
                for i in range(num_frames):
                    angle = (i / num_frames) * 2 * np.pi
                    pos = np.array([5000.0 * np.cos(angle), 5000.0 * np.sin(angle), 3000.0])
                    view = -pos / np.linalg.norm(pos)  # Look at origin
                    camera_positions.append(pos)
                    view_directions.append(view)

                # Get RSO center of mass
                rso_coords = np.argwhere(rso.shape)
                target_com_world = rso_coords.mean(axis=0) if len(rso_coords) > 0 else np.array([0, 0, 0])

                # Generate frames for each step in trajectory, updating belief
                for frame_idx, (camera_pos, view_dir, state) in enumerate(zip(camera_positions, view_directions, trajectory_states)):
                    # Update belief with observation at this position
                    if frame_idx > 0:
                        camera_fn = {
                            'focal_length': 50.0,
                            'sensor_width': 1.0,
                            'sensor_height': 1.0,
                            'image_width': 512,
                            'image_height': 512,
                        }
                        try:
                            simulate_observation(self.grid, rso, camera_fn, camera_pos)
                        except Exception as e:
                            self.logger.debug(f"Could not update belief during video: {e}")

                    fig = plt.figure(figsize=(12, 9))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_zlabel('Z (m)')
                    ax.set_title(f'Orbital Scenario - Step {frame_idx}/{len(trajectory_states)}')

                    # Get belief state voxels
                    if isinstance(self.grid.belief, torch.Tensor):
                        belief_np = self.grid.belief.detach().cpu().numpy()
                    else:
                        belief_np = self.grid.belief

                    certain_mask = (belief_np > 0.7)
                    certain_indices = np.argwhere(certain_mask)

                    # Plot belief voxels
                    if certain_indices.size > 0:
                        certain_world = self.grid.grid_to_world_coords(certain_indices)
                        certain_probabilities = belief_np[
                            certain_indices[:, 0],
                            certain_indices[:, 1],
                            certain_indices[:, 2]
                        ]
                        alphas = np.clip(certain_probabilities * 1.5 - 0.5, 0.3, 1.0)
                        colors = np.array([[0.0, 1.0, 0.0, a] for a in alphas])
                        ax.scatter(certain_world[:, 0], certain_world[:, 1], certain_world[:, 2],
                                  c=colors, marker='o', s=50, label='Belief (P > 0.7)', depthshade=False)

                    # Plot servicer path trajectory
                    cam_hist = np.array(camera_positions[:frame_idx+1])
                    ax.plot(cam_hist[:, 0], cam_hist[:, 1], cam_hist[:, 2],
                           c='blue', linestyle='-', linewidth=2, alpha=0.5, label='Servicer Path')

                    # Plot spacecraft at current position
                    draw_spacecraft(ax, camera_pos, view_dir,
                                   color="gray", scale=(6.0, 4.0, 3.0))
                    # Add legend entry for spacecraft
                    ax.plot([], [], c='gray', linewidth=0, marker='s', markersize=8, label='Servicer Spacecraft')

                    # Plot viewing direction
                    ax.plot([camera_pos[0], target_com_world[0]],
                           [camera_pos[1], target_com_world[1]],
                           [camera_pos[2], target_com_world[2]],
                           c='green', linestyle='--', linewidth=2, label='Viewing Direction')

                    # Add entropy text
                    entropy = self.grid.get_entropy()
                    ax.text2D(0.05, 0.95, f"Entropy: {entropy:.2f}",
                             transform=ax.transAxes, fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.8))

                    # Set view and legend
                    ax.legend(loc='upper right')
                    ax.view_init(elev=20, azim=-60)

                    # Capture frame
                    fig.canvas.draw()
                    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    scenario_frames.append(frame)
                    plt.close(fig)

                # Restore original belief
                self.grid.belief = belief_copy

                if scenario_frames:
                    scenario_path = self.output_dir / "orbital_trajectory_scenario.mp4"
                    imageio.mimsave(str(scenario_path), scenario_frames, fps=1, codec='libx264')
                    self.logger.info(f"Saved orbital scenario video to {scenario_path}")

            except Exception as e:
                self.logger.warning(f"Failed to generate orbital scenario video: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

            return self.output_dir / "orbital_trajectory_scenario.mp4"

        except Exception as e:
            self.logger.warning(f"Failed to generate orbital videos: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def run(
        self,
        initial_state: OrbitalState,
        num_iterations: int = 20,
        mcts_iters: int = 100,
        training_epochs: int = 5,
        training_batches_per_epoch: int = 100,
        save_interval: int = 5,
        eval_interval: int = 5,
        save_trees_interval: int = 10,
    ):
        """
        Run self-play loop.

        Args:
            initial_state: Starting state for episodes
            num_iterations: Number of iterations
            mcts_iters: MCTS iterations per episode step
            training_epochs: Training epochs per iteration
            training_batches_per_epoch: Batches per epoch
            save_interval: Save checkpoint every N iterations
            eval_interval: Evaluate every N iterations
            save_trees_interval: Save MCTS trees every N iterations (reduces disk usage)
        """
        self.logger.info(f"Starting self-play loop for {num_iterations} iterations")

        for iteration in range(num_iterations):
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"ITERATION {iteration+1}/{num_iterations}")
            self.logger.info(f"{'='*70}")

            # Determine if bootstrapping should be enabled
            use_bootstrap = self._should_enable_bootstrapping(iteration)

            # Determine if we should save trees this iteration (every N iterations + last iteration)
            should_save_trees = (iteration % save_trees_interval == 0) or (iteration == num_iterations - 1)

            # 1. Generate episode with MCTS planning
            self.logger.info("Generating episode with MCTS planning...")
            bootstrap_threshold = self.config.get("bootstrap_threshold", 0.8)
            episode_data, episode_stats = self._generate_episode(
                initial_state,
                mcts_iters=mcts_iters,
                use_network_bootstrap=use_bootstrap,
                bootstrap_threshold=bootstrap_threshold,
                save_trees=should_save_trees,
            )

            msg = (
                f"Episode completed: "
                f"return={episode_stats['episode_return']:.4f}, "
                f"length={episode_stats['episode_length']}, "
                f"time={episode_stats['episode_time']:.2f}s"
            )
            self.logger.info(msg)
            print(msg)  # Console output

            # 2. Add to replay buffer
            self._add_episode_to_buffer(episode_data)
            buffer_size = len(self.trainer.replay_buffer)
            self.logger.info(f"Replay buffer size: {buffer_size}")

            # 3. Train network
            if buffer_size >= 32:  # Need enough samples
                self.logger.info("Training network...")
                train_losses = self._train_network(
                    num_epochs=training_epochs,
                    batch_size=32,
                    num_batches_per_epoch=training_batches_per_epoch,
                    value_weight=1.0,
                    policy_weight=1.0,
                )

                msg = (
                    f"Training completed: "
                    f"policy_loss={train_losses['policy_loss']:.4f}, "
                    f"value_loss={train_losses['value_loss']:.4f}, "
                    f"total_loss={train_losses['total_loss']:.4f}, "
                    f"time={train_losses['training_time']:.2f}s"
                )
                self.logger.info(msg)
                print(msg)  # Console output

                # Update history
                self.history["iteration"].append(iteration)
                self.history["policy_loss"].append(train_losses["policy_loss"])
                self.history["value_loss"].append(train_losses["value_loss"])
                self.history["total_loss"].append(train_losses["total_loss"])
                self.history["buffer_size"].append(buffer_size)
            else:
                self.logger.warning(f"Insufficient buffer samples ({buffer_size} < 32)")

            # 4. Save checkpoint (every save_interval iterations)
            if (iteration + 1) % save_interval == 0:
                self.logger.info(f"Saving checkpoint at iteration {iteration+1}")
                self.trainer.save_checkpoint(iteration + 1, is_best=False)

            # 5. Save history and replay buffer after every iteration (for recovery)
            self._save_history()
            self._save_replay_buffer()

        self.logger.info(f"\n{'='*70}")
        self.logger.info("Self-play loop completed!")
        self.logger.info(f"{'='*70}")

        # Generate training metrics visualization
        self.logger.info("\nGenerating training visualizations...")
        try:
            loss_plot_path = self._generate_loss_plot()
            if loss_plot_path:
                self.logger.info(f"Saved training metrics plot to {loss_plot_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate loss plot: {e}")

        # Generate orbital trajectory video
        try:
            video_path = self._generate_orbital_video(initial_state)
            if video_path:
                self.logger.info(f"Saved orbital trajectory video to {video_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate orbital video: {e}")

    def _save_history(self):
        """Save training history to JSON."""
        history_file = self.output_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def _save_replay_buffer(self):
        """Save replay buffer to file for recovery."""
        import pickle
        buffer_file = self.output_dir / "replay_buffer.pkl"
        with open(buffer_file, "wb") as f:
            pickle.dump(self.trainer.replay_buffer, f)
        self.logger.debug(f"Replay buffer saved ({len(self.trainer.replay_buffer)} samples)")


def create_sample_initial_state(grid: VoxelGrid) -> OrbitalState:
    """Create a sample initial orbital state."""
    roe = np.array([0.0, 0.001, 0.0, 0.0, 0.0, 0.0])
    state = OrbitalState(roe=roe, grid=grid.clone())
    return state


if __name__ == "__main__":
    # Example usage
    from camera.camera_observations_gpu import GroundTruthRSO

    # Setup
    grid = VoxelGrid((20, 20, 20), use_torch=True, device="cuda")
    rso = GroundTruthRSO(grid)

    mdp_params = {
        "a_chief": 7000e3,
        "e_chief": 0.0,
        "i_chief": 0.0,
        "omega_chief": 0.0,
        "n_chief": np.sqrt(3.986004418e14 / 7000e3**3),
        "rso": rso,
        "camera_fn": {
            "fov_degrees": 60.0,
            "sensor_res": (32, 32),
            "noise_params": {"hit_false_neg": 0.05, "empty_false_pos": 0.001},
        },
        "grid_dims": (20, 20, 20),
        "lambda_dv": 0.01,
        "time_step": 30.0,
        "max_depth": 5,
        "target_radius": 2000.0,
        "gamma_r": 0.002,
        "r_min_rollout": 500.0,
        "r_max_rollout": 5000.0,
        "use_gpu": True,
    }

    mdp = OrbitalMCTSModelGPU(**mdp_params)
    initial_state = create_sample_initial_state(grid)

    # Run self-play loop
    loop = SelfPlayLoop(
        mdp_model=mdp,
        grid=grid,
        output_dir="./self_play_outputs",
        device="cuda",
    )

    loop.run(
        initial_state=initial_state,
        num_iterations=20,
        mcts_iters=100,
        training_epochs=5,
        training_batches_per_epoch=100,
        save_interval=5,
    )
