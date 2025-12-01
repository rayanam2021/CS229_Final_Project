import numpy as np
from roe.propagation import propagateGeomROE, rtn_to_roe
from roe.dynamics import apply_impulsive_dv
from camera.camera_observations import simulate_observation, calculate_entropy, VoxelGrid

class OrbitalState:
    def __init__(self, roe, grid, time):
        self.roe = np.array(roe, dtype=float)
        self.grid = grid
        self.time = time # Track absolute time in the state


class OrbitalMCTSModel:
    def __init__(self, a_chief, e_chief, i_chief, omega_chief, n_chief,
                 rso, camera_fn, grid_dims, lambda_dv, time_step, max_depth,
                 target_radius, gamma_r, r_min_rollout, r_max_rollout,
                 alpha_dv=10, beta_tan=0.5, grid=None):

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

        self.alpha_dv = alpha_dv
        self.beta_tan = beta_tan
        self.target_radius=target_radius
        self.gamma_r=gamma_r
        self.r_min_rollout=r_min_rollout
        self.r_max_rollout=r_max_rollout

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
        # 1) Apply Impulse
        # Use state.time for the maneuver epoch (absolute time)
        t_burn = np.array([state.time])
        
        child_state_impulse = apply_impulsive_dv(
            state.roe, action, self.a_chief, self.n_chief, t_burn,
            e=self.e_chief, i=self.i_chief, omega=self.omega_chief
        )

        # 2) Propagate
        # We propagate forward by one time_step from the current state.time
        # The target absolute time is state.time + time_step
        next_time = state.time + self.time_step
        t_target = np.array([next_time])
        
        rho_rtn_child, rhodot_rtn_child = propagateGeomROE(
            child_state_impulse,
            self.a_chief, self.e_chief, self.i_chief,
            self.omega_chief, self.n_chief,
            t_target,
            t0=state.time
        )

        pos_child = rho_rtn_child[:, 0] * 1000.0  # m

        # 3) Convert back to ROE
        # Convert at the NEW absolute time
        next_roe = np.array(
            rtn_to_roe(
                rho_rtn_child[:, 0],
                rhodot_rtn_child[:, 0],
                self.a_chief,
                self.n_chief,
                t_target
            )
        )

        # 4) Update Belief (deep copy branch-specific grid)
        grid = VoxelGrid(self.grid_dims)
        grid.belief[:]    = state.grid.belief[:]     # copy from current state
        grid.log_odds[:]  = state.grid.log_odds[:]

        entropy_before = calculate_entropy(grid.belief)
        simulate_observation(grid, self.rso, self.camera_fn, pos_child)
        entropy_after = calculate_entropy(grid.belief)

        info_gain = entropy_before - entropy_after
        dv_cost   = float(np.linalg.norm(action))

        reward = info_gain - self.lambda_dv * dv_cost

        # 5) Pack ROE, belief, and new time into OrbitalState
        next_state = OrbitalState(roe=next_roe, grid=grid, time=next_time)

        return next_state, reward

    # def rollout_policy(self, state):
    #     actions = self.actions(state)
    #     n = len(actions)
    #     idx = np.random.randint(n)
    #     return actions[idx]

    def rollout_policy(self, state):
        """
        Heuristic rollout policy:
        - Prefer smaller |dv|
        - Prefer tangential/normal actions (parallax)
        """
        actions = self.actions(state)
        scores = []

        for a in actions:
            dv_norm = np.linalg.norm(a)

            # base score from |dv|
            s = -self.alpha_dv * dv_norm

            # tangential / normal bonus: axis 0 = radial, 1/2 = tangential/normal
            if dv_norm > 0:
                main_axis = int(np.argmax(np.abs(a)))
                if main_axis in (1, 2):
                    s += self.beta_tan

            scores.append(s)

        # softmax over scores
        scores = np.array(scores)
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()

        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]