import numpy as np
from roe.propagation import propagateGeomROE, rtn_to_roe
from roe.dynamics import apply_impulsive_dv
from camera.camera_observations import simulate_observation, calculate_entropy, VoxelGrid

class OrbitalState:
    def __init__(self, roe, grid, time):
        self.roe = np.array(roe, dtype=float)
        self.grid = grid
        self.time = time 

class OrbitalMCTSModel:
    def __init__(self, a_chief, e_chief, i_chief, omega_chief, n_chief,
                 rso, camera_fn, grid_dims, lambda_dv, time_step, max_depth, grid=None):
        
        self.a_chief = a_chief
        self.e_chief = e_chief
        self.i_chief = i_chief
        self.omega_chief = omega_chief
        self.n_chief = n_chief

        self.rso = rso
        self.camera_fn = camera_fn
        self.grid_dims = grid_dims
        self.grid = grid
        self.lambda_dv = lambda_dv
        self.time_step = time_step
        self.max_depth = max_depth
        
        # Initialize cache (can be regenerated if deleted)
        self._cached_actions = self._generate_actions()
        self.action_space_size = len(self._cached_actions)

    def _generate_actions(self):
        """
        Generates the 13 discrete actions (No-op + +/- directions).
        """
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

    def get_all_actions(self):
        """
        Returns list of all possible action vectors.
        Refills the cache if it has been deleted or is None.
        """
        if not hasattr(self, '_cached_actions') or self._cached_actions is None:
            self._cached_actions = self._generate_actions()
            
        return self._cached_actions

    def actions(self, state):
        """Returns valid actions for a state."""
        return self.get_all_actions()

    def step(self, state, action):
        """
        Apply action, propagate dynamics, and compute reward.
        """
        # 1. Apply Impulsive Maneuver
        t_burn = np.array([state.time])
        
        child_state_impulse = apply_impulsive_dv(
            state.roe, action, self.a_chief, self.n_chief, t_burn,
            e=self.e_chief, i=self.i_chief, omega=self.omega_chief
        )

        # 2. Propagate Natural Motion
        next_time = state.time + self.time_step
        t_target = np.array([next_time])
        
        rho_rtn_child, rhodot_rtn_child = propagateGeomROE(
            child_state_impulse,
            self.a_chief, self.e_chief, self.i_chief,
            self.omega_chief, self.n_chief,
            t_target,
            t0=state.time
        )

        pos_child = rho_rtn_child[:, 0] * 1000.0  # Convert km to meters

        # 3. Map RTN back to ROE state
        next_roe = np.array(
            rtn_to_roe(
                rho_rtn_child[:, 0],
                rhodot_rtn_child[:, 0],
                self.a_chief,
                self.n_chief,
                t_target
            )
        )

        # 4. Update Belief Grid
        # Deep copy grid to ensure branches don't affect each other
        grid = VoxelGrid(self.grid_dims)
        grid.belief[:] = state.grid.belief[:]     
        grid.log_odds[:] = state.grid.log_odds[:]

        # --- Reward Calculation (Verbose) ---
        entropy_before = calculate_entropy(grid.belief)
        
        # Update grid based on observation at new position
        simulate_observation(grid, self.rso, self.camera_fn, pos_child)
        
        entropy_after = calculate_entropy(grid.belief)

        info_gain = entropy_before - entropy_after
        dv_cost = float(np.linalg.norm(action))

        reward = info_gain - self.lambda_dv * dv_cost

        # 5. Create Next State
        next_state = OrbitalState(roe=next_roe, grid=grid, time=next_time)

        return next_state, reward

    def rollout_policy(self, state):
        """
        Simple random rollout policy for Standard MCTS (non-AlphaZero).
        Selects a random action from the available action space.
        """
        actions = self.get_all_actions()
        idx = np.random.randint(len(actions))
        return actions[idx]