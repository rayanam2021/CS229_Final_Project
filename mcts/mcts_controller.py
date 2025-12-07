import numpy as np
import pandas as pd
import os
from multiprocessing import cpu_count

from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from mcts.mcts import MCTS

class MCTSController:
    
    def __init__(self, mu_earth, a_chief, e_chief, i_chief, omega_chief, n_chief, 
                 time_step=30.0, horizon=3, branching_factor=13, num_workers=None, 
                 lambda_dv=0.0, mcts_iters=3000):
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

        # Initialize Model
        self.model = OrbitalMCTSModel(
            a_chief, e_chief, i_chief, omega_chief, n_chief,
            rso=None, camera_fn=None,
            grid_dims=None,
            lambda_dv=lambda_dv,
            time_step=time_step,
            max_depth=horizon
        )

        # Initialize MCTS
        self.mcts = MCTS(
            model=self.model,
            iters=mcts_iters,
            max_depth=horizon,
            c=1.4,       
            gamma=0.99,  
        )

    def select_action(self, state, time, tspan, grid, rso, camera_fn, step=0, verbose=False, out_folder=None):
        self.model.rso = rso
        self.model.camera_fn = camera_fn
        self.model.grid_dims = grid.dims

        root_state = OrbitalState(roe=state.copy(), grid=grid, time=time)

        result = self.mcts.get_best_root_action(root_state, step, out_folder)
        
        if len(result) == 3:
            best_action, value, root_data = result
            
            if isinstance(root_data, dict):
                stats = root_data
            elif hasattr(root_data, 'N'):
                stats = {
                    'root_N': root_data.N,
                    'root_Q_sa': root_data.Q_sa,
                    'root_N_sa': root_data.N_sa,
                    'best_idx': int(np.argmax(root_data.Q_sa)) if len(root_data.Q_sa) > 0 else -1
                }
            else:
                # Unknown format
                stats = {}
        else:
            best_action, value = result
            stats = {'root_N': 0, 'root_Q_sa': [], 'root_N_sa': []}

        return best_action, value, stats
    
    def record_transition(self, t, state, action, reward, next_state, entropy_before=None, entropy_after=None,
                          info_gain=None, dv_cost=None, step_idx=None, root_stats=None, predicted_value=None):
        """
        Store one transition + diagnostics in the replay buffer.
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
            
            q_sa = root_stats.get("root_Q_sa", None)
            n_sa = root_stats.get("root_N_sa", None)
            if q_sa is not None:
                entry["root_Q_sa"] = np.asarray(q_sa).tolist()
            if n_sa is not None:
                entry["root_N_sa"] = np.asarray(n_sa).tolist()

        self.replay_buffer.append(entry)

    def save_replay_buffer(self, base_dir="output"):
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