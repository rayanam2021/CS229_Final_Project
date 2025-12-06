import numpy as np
from roe.propagation import propagateGeomROE, rtn_to_roe, ROEDynamics, map_roe_to_rtn
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
                 grid=None, initial_entropy=None):
        
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
        
        self.initial_entropy = initial_entropy if initial_entropy else 1.0 
        
        self.dyn_model = ROEDynamics(a_chief, e_chief, i_chief, omega_chief)
        self._cached_actions = self._generate_actions()
        self.action_space_size = len(self._cached_actions)

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
        t_burn = np.array([state.time])
        roe_after_impulse = apply_impulsive_dv(
            state.roe, action, self.a_chief, self.n_chief, t_burn,
            e=self.e_chief, i=self.i_chief, omega=self.omega_chief
        )

        next_roe = self.dyn_model.propagate(roe_after_impulse, self.time_step, second_order=True)
        next_time = state.time + self.time_step

        f_target = self.n_chief * next_time 
        r_vec, _ = map_roe_to_rtn(next_roe, self.a_chief, self.n_chief, f=f_target, omega=self.omega_chief)
        pos_child = r_vec * 1000.0

        grid = VoxelGrid(self.grid_dims)
        grid.belief[:] = state.grid.belief[:]     
        grid.log_odds[:] = state.grid.log_odds[:]

        entropy_before = calculate_entropy(grid.belief)
        simulate_observation(grid, self.rso, self.camera_fn, pos_child)
        entropy_after = calculate_entropy(grid.belief)

        info_gain = entropy_before - entropy_after
        dv_cost = float(np.linalg.norm(action))

        # Normalize the gain by the episode's initial entropy
        norm_gain = info_gain / self.initial_entropy
        
        reward = norm_gain - self.lambda_dv * dv_cost

        next_state = OrbitalState(roe=next_roe, grid=grid, time=next_time)

        return next_state, reward

    def rollout_policy(self, state):
        actions = self.get_all_actions()
        idx = np.random.randint(len(actions))
        return actions[idx]