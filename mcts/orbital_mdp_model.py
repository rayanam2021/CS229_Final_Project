import numpy as np
from roe.propagation import propagateGeomROE, rtn_to_roe
from roe.dynamics import apply_impulsive_dv
from camera.camera_observations import simulate_observation, calculate_entropy, VoxelGrid

class OrbitalState:
    def __init__(self, roe, grid):
        self.roe = np.array(roe, dtype=float)
        self.grid = grid


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

    def actions(self, state):
        delta_v_small = 0.01
        delta_v_large = 0.05
        actions = [np.zeros(3)]
        for axis in range(3):
            for mag in [delta_v_small, delta_v_large]:
                e = np.zeros(3)
                e[axis] = mag
                actions.append( e.copy())
                actions.append(-e.copy())
        return actions

    def step(self, state, action):
        # 1) propagate orbital state from state.roe
        tspan0 = np.array([0.0])
        child_state_impulse = apply_impulsive_dv(
            state.roe, action, self.a_chief, self.n_chief, tspan0
        )

        rho_rtn_child, rhodot_rtn_child = propagateGeomROE(
            child_state_impulse,
            self.a_chief, self.e_chief, self.i_chief,
            self.omega_chief, self.n_chief,
            np.array([self.time_step])
        )

        pos_child = rho_rtn_child[:, 0] * 1000.0  # m

        next_roe = np.array(
            rtn_to_roe(
                rho_rtn_child[:, 0],
                rhodot_rtn_child[:, 0],
                self.a_chief,
                self.n_chief,
                tspan0
            )
        )

        # 2) update voxel grid belief (deep copy branch-specific grid)
        grid = VoxelGrid(self.grid_dims)
        grid.belief[:]    = state.grid.belief[:]     # copy from current state
        grid.log_odds[:]  = state.grid.log_odds[:]

        entropy_before = calculate_entropy(grid.belief)
        simulate_observation(grid, self.rso, self.camera_fn, pos_child)
        entropy_after = calculate_entropy(grid.belief)

        info_gain = entropy_before - entropy_after
        dv_cost   = float(np.linalg.norm(action))

        reward = info_gain - self.lambda_dv * dv_cost

        # 3) pack ROE and belief into a new OrbitalState
        next_state = OrbitalState(roe=next_roe, grid=grid)

        return next_state, reward

    def rollout_policy(self, state):
        actions = self.actions(state)
        n = len(actions)
        idx = np.random.randint(n)
        return actions[idx]

