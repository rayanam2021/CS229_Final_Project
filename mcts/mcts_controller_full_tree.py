"""
Full MCTS Tree Search Controller for Orbital Camera RSO Characterization.
PARALLELIZED VERSION.
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


def _evaluate_child_action_worker(args):
    """Top-level worker function for multiprocessing (Windows-safe)."""
    (parent_state, action, horizon_level, tspan, parent_grid, rso, camera_fn, a_chief, e_chief, i_chief, omega_chief, n_chief, horizon, lambda_dv) = args
    
    # We approximate the start of the step as relative t=0 for the worker logic
    # unless absolute time is threaded through the tree nodes.
    # For now, we assume local propagation logic (f ~ 0 relative to start of step)
    # or consistent relative time steps.
    t_start = 0.0
    t_burn = np.array([t_start])

    # Apply action
    child_state_impulse = apply_impulsive_dv(
        parent_state, action, a_chief, n_chief, t_burn,
        e=e_chief, i=i_chief, omega=omega_chief
    )
    
    # Propagate (tspan is the duration vector, usually [dt])
    rho_rtn_child, rhodot_rtn_child = propagateGeomROE(
        child_state_impulse, a_chief, e_chief, i_chief, omega_chief, n_chief, 
        tspan, t0=t_start
    )
    
    pos_child = rho_rtn_child[:, 0] * 1000 
    vel_child = rhodot_rtn_child[:, 0] * 1000 
    
    # Map back to ROE
    child_state_propagated = np.array(rtn_to_roe(
        rho_rtn_child[:, 0], rhodot_rtn_child[:, 0], 
        a_chief, n_chief, tspan
    ))

    # Create child node
    child = MCTSNode(child_state_propagated, horizon_level=horizon_level + 1, max_horizon=horizon, action_taken=action)

    # Reconstruct grid for this branch
    child.grid = VoxelGrid(grid_dims=parent_grid.dims)
    child.grid.belief = parent_grid.belief.copy()
    child.grid.log_odds = parent_grid.log_odds.copy()

    # Propagate and observe
    entropy_before = calculate_entropy(child.grid.belief)
    simulate_observation(child.grid, rso, camera_fn, pos_child)
    entropy_after = calculate_entropy(child.grid.belief)

    information_gain = entropy_before - entropy_after
    child.entropy_at_node = information_gain
    
    dv_cost = float(np.linalg.norm(action))
    child.value = float(information_gain - lambda_dv * dv_cost)

    return {
        'child': child,
        'information_gain': information_gain,
        'branch_grid_belief': child.grid.belief.copy() if horizon_level + 1 < horizon else None
    }


class MCTSNode:
    def __init__(self, state, horizon_level, max_horizon, action_taken=None):
        self.state = state.copy()
        self.horizon_level = horizon_level
        self.max_horizon = max_horizon
        self.action_taken = action_taken
        self.belief = None
        self.children = []
        self.value = 0.0
        self.entropy_at_node = 0.0
        self.is_leaf = (horizon_level == max_horizon)
    
    def __repr__(self):
        return f"MCTSNode(level={self.horizon_level}, value={self.value:.4f}, children={len(self.children)})"


class MCTSController:
    
    def __init__(self, mu_earth, a_chief, e_chief, i_chief, omega_chief, n_chief, time_step=30.0, horizon=3, branching_factor=13, num_workers=None):
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

    def select_action(self, state, time, tspan, grid, rso, camera_fn, verbose=False, out_folder=None):
        if verbose:
            print(f"\n[MCTS] Building tree at time={time:.1f}s, state={np.round(state, 5)}")

        root = self._build_tree(state, tspan, grid, rso, camera_fn, verbose)
        best_path = self._extract_best_path(root)
        best_action = best_path[0] if best_path else np.zeros(3)
        best_value = root.value

        if verbose:
            print(f"[MCTS] Tree complete. Root value: {best_value:.6f}")
            print(f"[MCTS] Selected first action: {np.round(best_action, 3)}")
            
            try:
                log_path = os.path.join(out_folder, "root_children.txt")
                with open(log_path, "w") as fh:
                    fh.write(f"Root value: {best_value:.6f}\n")
                    fh.write(f"Selected action: {np.round(best_action,3)}\n")
                    fh.write("action,value,entropy,dv\n")
                    for ch in root.children:
                        dv_cost = float(np.linalg.norm(ch.action_taken)) if ch.action_taken is not None else 0.0
                        fh.write(f"{np.array2string(ch.action_taken, precision=4, separator=',')},{ch.value:.6f},{ch.entropy_at_node:.6f},{dv_cost:.6f}\n")
            except Exception as e:
                print(f"[MCTS] Failed to write decision log: {e}")

        return best_action, best_value, best_path
    
    def _build_tree(self, state, tspan, real_grid, rso, camera_fn, verbose=False):
        root = MCTSNode(state, horizon_level=0, max_horizon=self.horizon)
        root.grid = VoxelGrid(grid_dims=real_grid.dims)
        root.grid.belief = real_grid.belief.copy()
        root.grid.log_odds = real_grid.log_odds.copy()
        self._build_tree_recursive(root, tspan, rso, camera_fn, verbose)
        return root
    
    def _evaluate_child_action(self, args):
        (parent_state, action, horizon_level, tspan, parent_grid, rso, camera_fn) = args

        t_start = 0.0
        t_burn = np.array([t_start])

        child_state_impulse = apply_impulsive_dv(
            parent_state, action, self.a_chief, self.n_chief, t_burn,
            e=self.e_chief, i=self.i_chief, omega=self.omega_chief
        )
        
        rho_rtn_child, rhodot_rtn_child = propagateGeomROE(
            child_state_impulse, self.a_chief, self.e_chief, self.i_chief, self.omega_chief, self.n_chief, 
            tspan, t0=t_start
        )
        
        pos_child = rho_rtn_child[:, 0] * 1000
        vel_child = rhodot_rtn_child[:, 0] * 1000
        
        child_state_propagated = np.array(rtn_to_roe(
            rho_rtn_child[:, 0], rhodot_rtn_child[:, 0], 
            self.a_chief, self.n_chief, tspan
        ))
               
        child = MCTSNode(child_state_propagated, horizon_level=horizon_level + 1, max_horizon=self.horizon, action_taken=action)
        
        child.grid = VoxelGrid(grid_dims=parent_grid.dims)
        child.grid.belief = parent_grid.belief.copy()
        child.grid.log_odds = parent_grid.log_odds.copy()
        
        entropy_before = calculate_entropy(child.grid.belief)
        simulate_observation(child.grid, rso, camera_fn, pos_child)
        entropy_after = calculate_entropy(child.grid.belief)
        
        information_gain = entropy_before - entropy_after
        child.entropy_at_node = information_gain

        dv_cost = float(np.linalg.norm(action))
        child.value = float(information_gain - self.lambda_dv * dv_cost)

        return {
            'child': child,
            'information_gain': information_gain,
            'branch_grid_belief': child.grid.belief.copy() if horizon_level + 1 < self.horizon else None
        }
    
    def _build_tree_recursive(self, node, tspan, rso, camera_fn, verbose=False):
        actions = self._generate_actions()
        
        eval_args = [
            (
                node.state,
                action,
                node.horizon_level,
                tspan,
                node.grid,
                rso,
                camera_fn
            )
            for action in actions
        ]
        
        if self.num_workers > 1 and node.horizon_level < self.horizon - 1:
            eval_args_worker = [
                (
                    node.state,
                    action,
                    node.horizon_level,
                    tspan,
                    node.grid,
                    rso,
                    camera_fn,
                    self.a_chief,
                    self.e_chief,
                    self.i_chief,
                    self.omega_chief,
                    self.n_chief,
                    self.horizon,
                    self.lambda_dv,
                )
                for action in actions
            ]

            with Pool(processes=self.num_workers) as pool:
                results = pool.map(_evaluate_child_action_worker, eval_args_worker)
        else:
            results = [self._evaluate_child_action(args) for args in eval_args]
        
        for result in results:
            child = result['child']
            information_gain = result['information_gain']
            
            if verbose and node.horizon_level < 2:
                print(f"  Level {node.horizon_level + 1}, information_gain={information_gain:.6f}")
            
            if node.horizon_level + 1 < self.horizon:
                branch_grid = VoxelGrid(grid_dims=node.grid.dims)
                branch_grid.belief = result['branch_grid_belief']
                self._build_tree_recursive(child, tspan, rso, camera_fn)
            else:
                child.value = information_gain
            
            node.children.append(child)
        
        if node.children:
            child_values = [child.value for child in node.children]
            best_child_value = max(child_values)
            if node.horizon_level == 0:
                node.value = best_child_value
            else:
                node.value = best_child_value + node.entropy_at_node
        else:
            node.value = node.entropy_at_node
    
    def _extract_best_path(self, node):
        path = []
        current = node
        while current.children:
            best_child = max(current.children, key=lambda c: c.value)
            if best_child.action_taken is not None:
                path.append(best_child.action_taken)
            current = best_child
        return path
    
    def _generate_actions(self):
        delta_v_small = 0.01
        delta_v_large = 0.05
        actions = [np.zeros(3)]
        for axis in range(3):
            for mag in [delta_v_small, delta_v_large]:
                e = np.zeros(3)
                e[axis] = mag
                actions.append(e.copy())
                actions.append(-e.copy())
        return actions
    
    def record_transition(self, t, state, action, reward, next_state):
        self.replay_buffer.append({
            "time": t,
            "state": state.tolist(),
            "action": action.tolist(),
            "reward": float(reward),
            "next_state": next_state,
        })
    
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