"""
Full MCTS Tree Search Controller for Orbital Camera RSO Characterization.

This implements a complete Monte Carlo Tree Search with:
- Tree structure (with depth of horizon)
- All 13 actions branching at each level
- 13^3 = 2,197 total paths through the tree
- Bottom-up value propagation
- Optimal first-action selection
- Parallel evaluation of child nodes for speedup
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
    """Top-level worker function for multiprocessing (Windows-safe).

    Args tuple:
        parent_state, action, horizon_level, tspan, real_grid_belief,
        grid_dims, rso, camera_fn, a_chief, e_chief, i_chief, omega_chief,
        n_chief, horizon
    Returns same dict as the instance method version.
    """
    (parent_state, action, horizon_level, tspan, parent_grid, rso, camera_fn, a_chief, e_chief, i_chief, omega_chief, n_chief, horizon, lambda_dv) = args
    
    tspan0 = np.array([0.0])

    # Apply action to get propagated child state
    child_state_impulse = apply_impulsive_dv(parent_state, action, a_chief, n_chief, tspan0)
    rho_rtn_child, rhodot_rtn_child = propagateGeomROE(child_state_impulse, a_chief, e_chief, i_chief, omega_chief, n_chief, tspan)
    pos_child = rho_rtn_child[:, 0] * 1000  # Convert to meters
    vel_child = rhodot_rtn_child[:, 0] * 1000  # Convert to meters per second
    child_state_propagated = np.array(rtn_to_roe(rho_rtn_child[:, 0], rhodot_rtn_child[:, 0], a_chief, n_chief, tspan0))

    # Create child node
    child = MCTSNode(child_state_propagated, horizon_level=horizon_level + 1, max_horizon=horizon, action_taken=action)

    # Reconstruct grid for this branch
    child.grid = VoxelGrid(grid_dims=parent_grid.dims)
    child.grid.belief = parent_grid.belief.copy()
    child.grid.log_odds = parent_grid.log_odds.copy()

    # Propagate and observe
    entropy_before = calculate_entropy(child.grid.belief)

    # Simulate observation
    simulate_observation(child.grid, rso, camera_fn, pos_child)
    entropy_after = calculate_entropy(child.grid.belief)

    information_gain = entropy_before - entropy_after
    child.entropy_at_node = information_gain
    # Include ΔV cost into the child's immediate value to discourage
    # unnecessary maneuvers. The planning metric uses entropy gain minus
    # a cost proportional to the magnitude of the ΔV action.
    dv_cost = float(np.linalg.norm(action))
    child.value = float(information_gain - lambda_dv * dv_cost)

    return {
        'child': child,
        'information_gain': information_gain,
        'branch_grid_belief': child.grid.belief.copy() if horizon_level + 1 < horizon else None
    }


class MCTSNode:
    """
    Represents a node in the MCTS tree.
    
    Each node corresponds to:
    - A specific state (ROE)
    - A horizon level
    - An action that was taken to reach this node
    """
    
    def __init__(self, state, horizon_level, max_horizon, action_taken=None):
        self.state = state.copy()
        self.horizon_level = horizon_level
        self.max_horizon = max_horizon
        self.action_taken = action_taken  # Action that led to this node
        self.belief = None  # Voxel grid belief at this node
        self.children = []  # Will contain 13 child nodes (one per action)
        self.value = 0.0  # Best value achievable from this node onward
        self.entropy_at_node = 0.0
        self.is_leaf = (horizon_level == max_horizon)
    
    def __repr__(self):
        return f"MCTSNode(level={self.horizon_level}, value={self.value:.4f}, children={len(self.children)})"


class MCTSController:
    """
    Full MCTS Tree Search Controller with Parallel Evaluation.
    
    Builds a complete 3-level tree with all 13 actions at each level,
    evaluates all 13^3 = 2,197 paths using parallel processing, 
    and selects the optimal first action.
    
    Parallel evaluation: Child nodes are evaluated in parallel using multiprocessing
    to speed up tree building.
    """
    
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
            num_workers: Number of parallel workers (None = use all CPU cores)
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
        self.lambda_dv = 0  # ΔV cost weight

    
    def select_action(self, state, time, tspan, grid, rso, camera_fn, verbose=False, out_folder=None):
        """
        Selects the best action by building and evaluating the full MCTS tree.
        
        Args:
            state: Current ROE state (6D)
            time: Current simulation time (seconds)
            tspan: Time span array for propagation
            grid: VoxelGrid (real belief state)
            rso: GroundTruthRSO (target)
            camera_fn: Camera parameters dict
            verbose: Print debug info
        
        Returns:
            best_action: 3D ΔV vector (m/s)
            best_value: Expected cumulative value
            best_path: Sequence of actions forming optimal path
        """
        if verbose:
            print(f"\n[MCTS] Building tree at time={time:.1f}s, state={np.round(state, 5)}")

        # Build the tree
        root = self._build_tree(state, tspan, grid, rso, camera_fn, verbose)

        # Extract best path and first action
        best_path = self._extract_best_path(root)
        best_action = best_path[0] if best_path else np.zeros(3)
        best_value = root.value

        # Decision logging: print why the best first action was chosen
        if verbose:
            print(f"[MCTS] Tree complete. Root value: {best_value:.6f}")
            print(f"[MCTS] Best path (all actions): {[np.round(a, 3) for a in best_path]}")
            print(f"[MCTS] Selected first action: {np.round(best_action, 3)}")
            print("[MCTS] Root children (action, child.value, entropy_at_node, dv_cost):")
            for ch in root.children:
                dv_cost = float(np.linalg.norm(ch.action_taken)) if ch.action_taken is not None else 0.0
                print(f"  action={np.round(ch.action_taken,3)}, value={ch.value:.6f}, entropy={ch.entropy_at_node:.6f}, dv={dv_cost:.6f}")

            # Save a timestamped decision log to output/
            try:
                log_path = os.path.join(out_folder, "root_children.txt")
                with open(log_path, "w") as fh:
                    fh.write(f"Root value: {best_value:.6f}\n")
                    fh.write(f"Selected action: {np.round(best_action,3)}\n")
                    fh.write("action,value,entropy,dv\n")
                    for ch in root.children:
                        dv_cost = float(np.linalg.norm(ch.action_taken)) if ch.action_taken is not None else 0.0
                        fh.write(f"{np.array2string(ch.action_taken, precision=4, separator=',')},{ch.value:.6f},{ch.entropy_at_node:.6f},{dv_cost:.6f}\n")
                print(f"[MCTS] Decision log written to {log_path}")
            except Exception as e:
                print(f"[MCTS] Failed to write decision log: {e}")

        return best_action, best_value, best_path
    
    def _build_tree(self, state, tspan, real_grid, rso, camera_fn, verbose=False):
        """
        Recursively builds the full MCTS tree with parallel child evaluation.
        
        Args:
            state: Current ROE state
            tspan: Time span array for propagation
            real_grid: Real belief grid
            rso: Ground truth target
            camera_fn: Camera parameters
            verbose: Debug printing
        
        Returns:
            root: MCTSNode representing the root of the tree
        """
        root = MCTSNode(state, horizon_level=0, max_horizon=self.horizon)
        root.grid = VoxelGrid(grid_dims=real_grid.dims)
        root.grid.belief = real_grid.belief.copy()
        root.grid.log_odds = real_grid.log_odds.copy()
        self._build_tree_recursive(root, tspan, rso, camera_fn, verbose)
        return root
    
    def _evaluate_child_action(self, args):
        """
        Worker function for parallel evaluation of a single child action.
        
        Args:
            args: Tuple of (parent_state, action, horizon_level, time, real_grid_belief, 
                           grid_dims, rso, camera_fn)
        
        Returns:
            child_data: Dictionary with child node information
        """
        (parent_state, action, horizon_level, tspan, parent_grid, rso, camera_fn) = args

        tspan0 = np.array([0.0])

        # Apply action to get propagated child state
        child_state_impulse = apply_impulsive_dv(parent_state, action, self.a_chief, self.n_chief, tspan0)
        rho_rtn_child, rhodot_rtn_child = propagateGeomROE(child_state_impulse, self.a_chief, self.e_chief, self.i_chief, self.omega_chief, self.n_chief, tspan)
        pos_child = rho_rtn_child[:, 0] * 1000  # Convert to meters
        vel_child = rhodot_rtn_child[:, 0] * 1000  # Convert to meters per second
        child_state_propagated = np.array(rtn_to_roe(rho_rtn_child[:, 0], rhodot_rtn_child[:, 0], self.a_chief, self.n_chief, tspan0))
               
        # Create child node
        child = MCTSNode(child_state_propagated, horizon_level=horizon_level + 1, max_horizon=self.horizon, action_taken=action)
        
        # Reconstruct grid for this branch
        child.grid = VoxelGrid(grid_dims=parent_grid.dims)
        child.grid.belief = parent_grid.belief.copy()
        child.grid.log_odds = parent_grid.log_odds.copy()
        
        # Propagate and observe
        entropy_before = calculate_entropy(child.grid.belief)
        
        # Simulate observation
        simulate_observation(child.grid, rso, camera_fn, pos_child)
        entropy_after = calculate_entropy(child.grid.belief)
        
        information_gain = entropy_before - entropy_after
        child.entropy_at_node = information_gain

        # Include ΔV cost in the immediate child value to encourage
        # information-efficient actions during planning (serial path).
        dv_cost = float(np.linalg.norm(action))
        child.value = float(information_gain - self.lambda_dv * dv_cost)

        return {
            'child': child,
            'information_gain': information_gain,
            'branch_grid_belief': child.grid.belief.copy() if horizon_level + 1 < self.horizon else None
        }
    
    def _build_tree_recursive(self, node, tspan, rso, camera_fn, verbose=False):
        """
        Recursively builds tree with parallel child evaluation.
        
        At each node:
        1. If leaf (horizon == self.horizon): compute entropy gain and return
        2. If internal: create 13 children in parallel and recurse
        3. Set node value = max(children values) + entropy at this node
        
        Args:
            node: Current MCTSNode to expand
            time: Simulation time
            real_grid: Real belief (for deep copy at each branch)
            rso: Ground truth
            camera_fn: Camera config
            verbose: Debug output
        """
        # Generate 13 actions
        actions = self._generate_actions()
        
        # Prepare arguments for parallel evaluation
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
        
        # Evaluate children in parallel (Windows-safe worker)
        if self.num_workers > 1 and node.horizon_level < self.horizon - 1:
            # Prepare worker args that include controller parameters
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

            # Use multiprocessing Pool with top-level worker function
            with Pool(processes=self.num_workers) as pool:
                results = pool.map(_evaluate_child_action_worker, eval_args_worker)
        else:
            # Serial evaluation for leaf parents or single worker
            results = [self._evaluate_child_action(args) for args in eval_args]
        
        # Process results and recurse
        for result in results:
            child = result['child']
            information_gain = result['information_gain']
            
            if verbose and node.horizon_level < 2:
                print(f"  Level {node.horizon_level + 1}, information_gain={information_gain:.6f}")
            
            # Recurse if not at leaf level
            if node.horizon_level + 1 < self.horizon:
                # Reconstruct grid for recursion
                branch_grid = VoxelGrid(grid_dims=node.grid.dims)
                branch_grid.belief = result['branch_grid_belief']
                
                self._build_tree_recursive(child, tspan, rso, camera_fn)
            else:
                # Leaf node: value is just information gain
                child.value = information_gain
            
            node.children.append(child)
        
        # Set this node's value = max of children values + entropy at this node
        if node.children:
            child_values = [child.value for child in node.children]
            best_child_value = max(child_values)
            if node.horizon_level == 0:
                # ROOT NODE: Value is just the best child path value
                # Do NOT add entropy_at_node because that's the current observation
                # (already taken), not part of future lookahead
                node.value = best_child_value
            else:
                # INTERNAL NODE: Value includes this node's entropy plus best future
                node.value = best_child_value + node.entropy_at_node
        else:
            node.value = node.entropy_at_node
    
    def _extract_best_path(self, node):
        """
        Extracts the optimal path from root to leaf by greedily following
        the child with the highest value at each level.
        
        Args:
            node: Root node of the tree
        
        Returns:
            path: List of actions forming the optimal path
        """
        path = []
        current = node
        
        while current.children:
            # Find child with highest value
            best_child = max(current.children, key=lambda c: c.value)
            if best_child.action_taken is not None:
                path.append(best_child.action_taken)
            current = best_child
        
        return path
    
    def _generate_actions(self):
        """
        Generate 13 candidate actions (no-op, ±small/large in 3 RTN directions).
        
        Returns:
            actions: List of 13 actions (3D numpy arrays in m/s)
        """
        delta_v_small = 0.01  # m/s
        delta_v_large = 0.05  # m/s
        actions = [np.zeros(3)]  # no-op
        
        for axis in range(3):
            for mag in [delta_v_small, delta_v_large]:
                e = np.zeros(3)
                e[axis] = mag
                actions.append(e.copy())
                actions.append(-e.copy())
        
        return actions
    
    def record_transition(self, t, state, action, reward, next_state):
        """
        Store (s, a, r, s') to replay buffer.
        
        Args:
            t: Time of transition
            state: Initial state (6D ROE)
            action: Action taken (3D ΔV)
            reward: Reward received
            next_state: Resulting state (6D ROE)
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
        
        Args:
            base_dir: Output directory
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
