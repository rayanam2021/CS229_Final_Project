import numpy as np
from roe.propagation import rtn_to_roe, ROEDynamics, map_roe_to_rtn
from roe.dynamics import apply_impulsive_dv
from camera.camera_observations import simulate_observation, calculate_entropy, VoxelGrid

class OrbitalState:
    def __init__(self, roe, grid, time):
        self.roe = np.array(roe, dtype=float)
        self.grid = grid
        self.time = time 

class OrbitalMCTSModel:
    def __init__(self, a_chief, e_chief, i_chief, omega_chief, n_chief,
                 rso, camera_fn, grid_dims, lambda_dv, time_step, max_depth,
                 alpha_dv=10, beta_tan=0.5):

        self.a_chief = a_chief
        self.e_chief = e_chief
        self.i_chief = i_chief
        self.omega_chief = omega_chief
        self.n_chief = n_chief

        self.rso = rso
        self.camera_fn = camera_fn
        self.grid_dims = grid_dims
        self.lambda_dv = lambda_dv
        self.time_step = time_step
        self.max_depth = max_depth

        self.dyn_model = ROEDynamics(a_chief, e_chief, i_chief, omega_chief)
        self._cached_actions = self._generate_actions()

        self.alpha_dv = alpha_dv
        self.beta_tan = beta_tan

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

    def get_all_actions(self):
        if not hasattr(self, '_cached_actions') or self._cached_actions is None:
            self._cached_actions = self._generate_actions()
        return self._cached_actions

    def actions(self, state):
        return self.get_all_actions()

    def step(self, state, action):
        """
        Apply action, propagate dynamics analytically, and compute reward.
        """
        # 1. Apply Impulsive Maneuver (Instantaneous change in ROE)
        t_burn = np.array([state.time])
        
        roe_after_impulse = apply_impulsive_dv(
            state.roe, action, self.a_chief, self.n_chief, t_burn,
            e=self.e_chief, i=self.i_chief, omega=self.omega_chief
        )
        # Calculate the magnitude of the change in dimensionless ROE
        # delta_roe = np.linalg.norm(roe_after_impulse - state.roe)
        # action_mag = np.linalg.norm(action)
        
        # if action_mag > 1e-6:
        #     # Scale ROE change to estimated meters (approximate) just for visibility
        #     roe_change_meters = delta_roe * self.a_chief * 1000.0
        #     print(f" [DEBUG] BURN: |dv|={action_mag:.3f} m/s -> ROE Change ~ {roe_change_meters:.2f} m")
        
        #     if delta_roe < 1e-9:
        #         print(" [WARNING] Action taken but ROE did not change! Check dynamics.py scaling.")

        # 2. Propagate Natural Motion (Analytically)
        # Using self.dyn_model which is now guaranteed to exist
        next_roe = self.dyn_model.propagate(roe_after_impulse, self.time_step, second_order=True)
        next_time = state.time + self.time_step

        # 3. Calculate Position for Observation (RTN)
        f_target = self.n_chief * next_time # Mean anomaly approximation
        
        r_vec, _ = map_roe_to_rtn(next_roe, self.a_chief, self.n_chief, f=f_target, omega=self.omega_chief)
        pos_child = r_vec * 1000.0  # Convert km to meters for the camera

        # 4. Update Belief Grid
        grid = VoxelGrid(self.grid_dims)
        grid.belief[:] = state.grid.belief[:]     
        grid.log_odds[:] = state.grid.log_odds[:]

        entropy_before = calculate_entropy(grid.belief)
        simulate_observation(grid, self.rso, self.camera_fn, pos_child)
        entropy_after = calculate_entropy(grid.belief)

        info_gain = entropy_before - entropy_after
        dv_cost = float(np.linalg.norm(action))

        # Reward calculation
        reward = info_gain - self.lambda_dv * dv_cost

        # 5. Create Next State
        next_state = OrbitalState(roe=next_roe, grid=grid, time=next_time)

        return next_state, reward

    def random_rollout_policy(self, state):
        """Random action selection"""
        actions = self.actions(state)
        n = len(actions)
        idx = np.random.randint(n)
        return actions[idx]

    def custom_rollout_policy(self, state):
        """Heuristic rollout policy with GPU acceleration"""
        actions = self.actions(state)
        scores = []

        for a in actions:
            dv_norm = np.linalg.norm(a)
            s = -self.alpha_dv * dv_norm

            if dv_norm > 0:
                main_axis = int(np.argmax(np.abs(a)))
                if main_axis in (1, 2):
                    s += self.beta_tan

            scores.append(s)

        scores = np.array(scores)
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()

        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]