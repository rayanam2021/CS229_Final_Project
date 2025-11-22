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
        - Prefer radius near target_radius
        - Reject actions that push radius outside [r_min_rollout, r_max_rollout]
        """

        actions = self.actions(state)

        # we will potentially filter some actions out for rollouts
        scores = []
        valid_actions = []

        # time span for a single rollout step radius check
        tspan_impulse = np.array([0.0])
        tspan_prop = np.array([self.time_step])

        for a in actions:
            dv_norm = np.linalg.norm(a)

            # base score from |dv| (Boltzmann: exp(-alpha_dv * |dv|))
            s = -self.alpha_dv * dv_norm

            # tangential / normal bonus: axis 0 = radial, 1/2 = tangential/normal
            if dv_norm > 0:
                main_axis = int(np.argmax(np.abs(a)))
                if main_axis in (1, 2):
                    s += self.beta_tan

            # --- radius-based term: estimate child radius with 1-step propagation ---
            # Start from current ROE, apply impulsive dv, propagate for one time_step
            # state.roe is assumed to be the current ROE vector
            roe_child = apply_impulsive_dv(
                state.roe,
                a,
                self.a_chief,
                self.n_chief,
                tspan_impulse
            )

            rho_rtn_child, _ = propagateGeomROE(
                roe_child,
                self.a_chief,
                self.e_chief,
                self.i_chief,
                self.omega_chief,
                self.n_chief,
                tspan_prop
            )

            # rho_rtn_child is in km; convert to meters and take last column
            if rho_rtn_child.ndim == 2:
                pos_child = rho_rtn_child[:, -1] * 1000.0
            else:
                pos_child = rho_rtn_child * 1000.0

            radius = float(np.linalg.norm(pos_child))

            # rollout-only radius guardrails
            if (radius < self.r_min_rollout) or (radius > self.r_max_rollout):
                # skip this action in rollouts
                continue

            # penalize deviation from target radius
            s -= self.gamma_r * abs(radius - self.target_radius)

            valid_actions.append(a)
            scores.append(s)

        # if everything got filtered out (should be rare), fall back to simple |dv| heuristic
        if not valid_actions:
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

        # softmax over scores for valid actions
        scores = np.array(scores)
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()

        idx = np.random.choice(len(valid_actions), p=probs)
        return valid_actions[idx]
