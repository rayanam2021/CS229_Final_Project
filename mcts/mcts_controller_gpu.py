"""
GPU-accelerated MCTS controller with parallel search and batched rollouts.

Features:
1. Parallel MCTS tree search
2. Batched action evaluation
3. GPU observation caching
4. Vectorized reward computation
"""

import numpy as np
import time
from mcts.mcts_parallel import MCTSParallel
from mcts.orbital_mdp_model_gpu import OrbitalMCTSModelGPU, OrbitalState


class MCTSControllerGPU:
    """GPU-accelerated MCTS controller for orbital planning."""

    def __init__(self, mcts_iters=1000, mcts_c=1.4, horizon=20, gamma=0.95,
                 batch_rollouts=True, batch_size=10, use_gpu_observations=True):
        """
        Args:
            mcts_iters: Number of MCTS iterations
            mcts_c: UCB1 exploration constant
            horizon: Tree search depth
            gamma: Discount factor
            batch_rollouts: Enable batched rollout evaluation
            batch_size: Rollout batch size
            use_gpu_observations: Enable GPU-accelerated observations
        """
        self.mcts_iters = mcts_iters
        self.mcts_c = mcts_c
        self.horizon = horizon
        self.gamma = gamma
        self.batch_rollouts = batch_rollouts
        self.batch_size = batch_size
        self.use_gpu_observations = use_gpu_observations

        self.mcts = None
        self.mdp = None
        self.replay_buffer = []
        self.step_count = 0

    def setup(self, a_chief, e_chief, i_chief, omega_chief, n_chief,
              rso, camera_fn, grid_dims, lambda_dv, time_step,
              target_radius, gamma_r, r_min_rollout, r_max_rollout,
              alpha_dv=10, beta_tan=0.5, grid=None):
        """Initialize controller with orbital parameters."""

        # Create MDP model with GPU support
        self.mdp = OrbitalMCTSModelGPU(
            a_chief=a_chief,
            e_chief=e_chief,
            i_chief=i_chief,
            omega_chief=omega_chief,
            n_chief=n_chief,
            rso=rso,
            camera_fn=camera_fn,
            grid_dims=grid_dims,
            lambda_dv=lambda_dv,
            time_step=time_step,
            max_depth=self.horizon,
            target_radius=target_radius,
            gamma_r=gamma_r,
            r_min_rollout=r_min_rollout,
            r_max_rollout=r_max_rollout,
            alpha_dv=alpha_dv,
            beta_tan=beta_tan,
            grid=grid,
            use_gpu=self.use_gpu_observations
        )

        # Create parallel MCTS
        self.mcts = MCTSParallel(
            model=self.mdp,
            iters=self.mcts_iters,
            max_depth=self.horizon,
            c=self.mcts_c,
            gamma=self.gamma,
            batch_rollouts=self.batch_rollouts,
            batch_size=self.batch_size
        )

    def search(self, state, output_path):
        """Run MCTS search and return best action."""
        if self.mcts is None:
            raise RuntimeError("Controller not initialized. Call setup() first.")

        start = time.time()
        best_action, best_value, stats = self.mcts.get_best_root_action(
            state, self.step_count, output_path, return_stats=True
        )
        elapsed = time.time() - start

        # Log to replay buffer
        log_entry = {
            'step': self.step_count,
            'action': best_action,
            'value': best_value,
            'elapsed_ms': elapsed * 1000,
            'mcts_stats': stats
        }
        self.replay_buffer.append(log_entry)
        self.step_count += 1

        return best_action, best_value, stats

    def execute_and_observe(self, current_state, action):
        """Execute action and observe result."""
        next_state, reward = self.mdp.step(current_state, action)
        return next_state, reward

    def batch_evaluate_actions(self, state, actions):
        """
        Evaluate multiple actions in batch.

        Args:
            state: Current orbital state
            actions: List of action vectors

        Returns:
            next_states: List of next states
            rewards: List of rewards
        """
        return self.mdp.step_batch(state, actions)

    def get_replay_buffer(self):
        """Get all logged experiences."""
        return self.replay_buffer

    def clear_replay_buffer(self):
        """Clear replay buffer."""
        self.replay_buffer = []
