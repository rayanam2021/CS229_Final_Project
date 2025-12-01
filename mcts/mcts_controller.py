"""
Standard MCTS Controller using orbital_mdp_model.
"""

import numpy as np
import pandas as pd
import os
from multiprocessing import cpu_count
from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from mcts.mcts import MCTS

class MCTSController:
    
    def __init__(self, mu_earth, a_chief, e_chief, i_chief, omega_chief, n_chief, time_step=30.0, horizon=3, branching_factor=13, num_workers=None):
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
            num_workers: (Not used in serial MCTS but kept for interface compatibility)
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
        self.lambda_dv = 0

        self.model = OrbitalMCTSModel(
                    a_chief, e_chief, i_chief, omega_chief, n_chief,
                    rso=None, camera_fn=None,
                    grid_dims=None,
                    lambda_dv=0.0,
                    time_step=time_step,
                    max_depth=horizon,
                )

        self.mcts = MCTS(model=self.model, iters=3000, max_depth=horizon)

    def select_action(self, state, time, tspan, grid, rso, camera_fn, step=0, verbose=False, out_folder=None):

        # Update the model with current environment objects
        self.model.rso = rso
        self.model.camera_fn = camera_fn
        self.model.grid_dims = grid.dims

        # Wrap ROEs + belief + time into an OrbitalState for MCTS
        root_state = OrbitalState(roe=state.copy(), grid=grid, time=time)

        best_action, value = self.mcts.get_best_root_action(root_state, step, out_folder)

        return best_action, value, [best_action]
    
    def record_transition(self, t, state, action, reward, next_state):
        """
        Store (s, a, r, s') to replay buffer.
        """
        self.replay_buffer.append({
            "time": t,
            "state": state.tolist(),
            "action": action.tolist(),
            "reward": float(reward),
            "next_state": next_state,
        })
    
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