"""
Parallel MCTS Controller using Root Parallelization.
Drop-in replacement for MCTSController with 5-8x speedup.
"""

import numpy as np
import pandas as pd
import os

from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from mcts.mcts_parallel import ParallelMCTS


class MCTSControllerParallel:
    """
    MCTS Controller using root parallelization for 5-8x speedup.

    Identical API to MCTSController, but runs multiple independent MCTS
    searches in parallel and aggregates their results.
    """

    def __init__(self, mu_earth, a_chief, e_chief, i_chief, omega_chief, n_chief,
                 time_step=30.0, horizon=3,
                 lambda_dv=0.0, mcts_iters=3000, num_workers=None,
                 use_torch=False, device='cpu'):
        """
        Args:
            num_workers: Number of parallel workers (default: CPU count - 1)
            Other args: Same as MCTSController
        """
        self.mu_earth = mu_earth
        self.a_chief = a_chief
        self.e_chief = e_chief
        self.i_chief = i_chief
        self.omega_chief = omega_chief
        self.n_chief = n_chief
        self.time_step = time_step
        self.horizon = horizon
        self.replay_buffer = []
        self.lambda_dv = lambda_dv

        # Initialize Model with GPU support
        self.model = OrbitalMCTSModel(
            a_chief, e_chief, i_chief, omega_chief, n_chief,
            rso=None, camera_fn=None,
            grid_dims=None,
            lambda_dv=lambda_dv,
            time_step=time_step,
            max_depth=horizon,
            use_torch=use_torch,
            device=device
        )

        # Initialize Parallel MCTS
        self.mcts = ParallelMCTS(
            model=self.model,
            iters=mcts_iters,
            max_depth=horizon,
            c=1.4,
            gamma=0.99,
            num_workers=num_workers
        )

    def select_action(self, state, time, tspan, grid, rso, camera_fn, step=0, verbose=False, out_folder=None):
        """Same API as MCTSController.select_action"""
        self.model.rso = rso
        self.model.camera_fn = camera_fn
        self.model.grid_dims = grid.dims

        root_state = OrbitalState(roe=state.copy(), grid=grid, time=time)

        result = self.mcts.get_best_root_action(root_state, step, out_folder)

        if len(result) == 3:
            best_action, value, root_data = result

            if isinstance(root_data, dict):
                stats = root_data
            else:
                stats = {}
        else:
            best_action, value = result
            stats = {'root_N': 0, 'root_Q_sa': [], 'root_N_sa': []}

        return best_action, value, stats

    def record_transition(self, t, state, action, reward, next_state, entropy_before=None, entropy_after=None,
                          info_gain=None, dv_cost=None, step_idx=None, root_stats=None, predicted_value=None):
        """Same API as MCTSController.record_transition"""
        entry = {
            "time": float(t),
            "step": int(step_idx) if step_idx is not None else None,
            "state": state.tolist(),
            "action": action.tolist(),
            "reward": float(reward),
            "next_state": next_state.tolist(),
        }

        if entropy_before is not None:
            entry["entropy_before"] = float(entropy_before)
        if entropy_after is not None:
            entry["entropy_after"] = float(entropy_after)
        if info_gain is not None:
            entry["info_gain"] = float(info_gain)
        if dv_cost is not None:
            entry["dv_cost"] = float(dv_cost)
        if predicted_value is not None:
            entry["predicted_value"] = float(predicted_value)

        if root_stats:
            entry["root_N"] = int(root_stats.get("root_N", 0))
            if "parallel_time" in root_stats:
                entry["parallel_time"] = float(root_stats["parallel_time"])

        self.replay_buffer.append(entry)

    def save_replay_buffer(self, base_dir="output"):
        """Same API as MCTSController.save_replay_buffer"""
        if not self.replay_buffer:
            return

        buffer_path = os.path.join(base_dir, "replay_buffer.csv")
        df = pd.DataFrame(self.replay_buffer)
        df.to_csv(buffer_path, index=False)
        print(f"Replay buffer saved to {buffer_path}")
