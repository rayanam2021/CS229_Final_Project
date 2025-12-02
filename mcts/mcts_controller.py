"""
Standard MCTS Controller using orbital_mdp_model.
"""

import numpy as np
import pandas as pd
import os
import copy
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from roe.propagation import propagateGeomROE, rtn_to_roe
from roe.dynamics import apply_impulsive_dv
from camera.camera_observations import calculate_entropy, simulate_observation, VoxelGrid

from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from mcts.mcts import MCTS
from mcts.mcts_parallel import ParallelMCTS

class MCTSController:
    
    def __init__(self, mu_earth, a_chief, e_chief, i_chief, omega_chief, n_chief,
                 time_step, horizon, alpha_dv, beta_tan, rollout_policy,
                 lambda_dv, branching_factor=13, num_workers=None, mcts_iters=3000, mcts_c=1.4, gamma=0.99,
                 use_parallel_mcts=True, num_processes=None, verbose=False):
        """
        Args:
            mu_earth: Gravitational parameter
            a_chief: Chief semi-major axis
            e_chief: Chief eccentricity
            i_chief: Chief inclination
            omega_chief: Chief argument of perigee
            time_step: Time step duration (seconds)
            horizon: Tree depth (number of decision levels)
            branching_factor: Number of actions per node (typically 13)
            num_workers: (Deprecated, kept for compatibility)
            use_parallel_mcts: Use ParallelMCTS (multiprocessing) instead of sequential MCTS (default: True)
            num_processes: Number of processes for parallel MCTS (default: cpu_count())
        """
        self.mu_earth = mu_earth
        self.a_chief = a_chief
        self.e_chief = e_chief
        self.i_chief = i_chief
        self.omega_chief = omega_chief
        self.n_chief = n_chief
        self.time_step = time_step
        self.horizon = horizon
        self.branching_factor = branching_factor
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.replay_buffer = []
        self.lambda_dv = lambda_dv
        self.use_parallel_mcts = use_parallel_mcts
        self.num_processes = num_processes or cpu_count()
        self.verbose = verbose

        self.model = OrbitalMCTSModel(
                    a_chief, e_chief, i_chief, omega_chief, n_chief,
                    rso=None, camera_fn=None,
                    grid_dims=None,
                    lambda_dv=lambda_dv,
                    time_step=time_step,
                    max_depth=horizon,
                    alpha_dv=alpha_dv,
                    beta_tan=beta_tan,
                )

        # Instantiate MCTS based on parallel/sequential preference
        if use_parallel_mcts:
            if verbose:
                print(f"Using ParallelMCTS with {self.num_processes} processes")
            self.mcts = ParallelMCTS(
                model=self.model,
                iters=mcts_iters,
                max_depth=horizon,
                c=mcts_c,
                gamma=gamma,
                roll_policy=rollout_policy,
                num_processes=self.num_processes
            )
        else:
            if verbose:
                print("Using sequential MCTS")
            self.mcts = MCTS(
                model=self.model,
                iters=mcts_iters,
                max_depth=horizon,
                c=mcts_c,
                gamma=gamma,
                roll_policy=rollout_policy
            )


    def select_action(self, state, time, tspan, grid, rso, camera_fn, step=0, verbose=False, out_folder=None):

        # Update the model with current environment objects
        self.model.rso = rso
        self.model.camera_fn = camera_fn
        self.model.grid_dims = grid.dims

        # Wrap ROEs + belief into an OrbitalState for MCTS
        root_state = OrbitalState(roe=state.copy(), grid=grid, time=time)

        best_action, value, root_stats = self.mcts.get_best_root_action(root_state, step, out_folder, return_stats=True)

        return best_action, value, root_stats
    
    def record_transition(self, t, state, action, reward, next_state, entropy_before=None, entropy_after=None,
                          info_gain=None, dv_cost=None, step_idx=None, root_stats=None, predicted_value=None):
        """
        Store one transition + diagnostics in the replay buffer.

        Args:
            t:         Time of transition (float)
            state:     ROE before action (6,)
            action:    Δv in RTN (3,)
            reward:    Actual reward used in sim
            next_state:ROE after action + propagation (6,)
            entropy_before/after: grid entropies
            info_gain: entropy_before - entropy_after
            dv_cost:   ||Δv||
            step_idx:  integer step index
            root_stats:dict from MCTS.get_best_root_action
            predicted_value: MCTS root value estimate
        """
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

        if root_stats is not None:
            entry["root_N"] = int(root_stats.get("root_N", 0))
            entry["root_best_idx"] = int(root_stats.get("best_idx", -1))

            q_sa = root_stats.get("root_Q_sa", None)
            n_sa = root_stats.get("root_N_sa", None)
            if q_sa is not None:
                entry["root_Q_sa"] = np.asarray(q_sa).tolist()
            if n_sa is not None:
                entry["root_N_sa"] = np.asarray(n_sa).tolist()

        self.replay_buffer.append(entry)

    
    def save_replay_buffer(self, base_dir="output"):
        """
        Write replay buffer to CSV in timestamped folder.
        """
        # Use unified output folder if provided
        if hasattr(self, 'output_folder') and self.output_folder:
            folder = self.output_folder
        else:
            folder = base_dir
        os.makedirs(folder, exist_ok=True)

        df = pd.DataFrame(self.replay_buffer)
        csv_path = os.path.join(folder, "replay_buffer.csv")
        df.to_csv(csv_path, index=False)

        print(f"Saved replay buffer with {len(df)} entries to {csv_path}")
        return csv_path
