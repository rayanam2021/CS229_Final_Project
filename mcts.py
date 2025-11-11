import math
import random
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class MCTSConfig:
    simulations: int = 1000     # number of simulations per action decision
    max_depth: int = 50         # depth cap per simulation
    c_ucb: float = 1.414213562  # exploration constant (sqrt(2) is a good default)
    gamma: float = 0.99         # discount factor

class MCTS:
    """
    Model-based, anytime Monte Carlo Tree Search (MDP version).
    Requires a generative model: actions(s), step(s,a)->(s', r, done).
    """

    def __init__(self, actions, step, rollout_policy=None, cfg: MCTSConfig = MCTSConfig()):
        self.actions = actions
        self.step = step
        self.rollout_policy = rollout_policy or self._random_rollout_policy
        self.cfg = cfg

        # Visit counts and value estimates
        self.N_sa = defaultdict(int)     # N(s,a)
        self.N_s  = defaultdict(int)     # N(s)
        self.Q_sa = defaultdict(float)   # Q(s,a)
        # Children: map (s) -> set of (a, s') seen
        self.children = defaultdict(set)

    # ----- Public API -----
    def plan(self, s0):
        """Run self.cfg.simulations simulations from s0, then return best action by Q."""
        for _ in range(self.cfg.simulations):
            self._simulate(s0, depth=self.cfg.max_depth)

        # Choose action with max Q(s0, a); break ties by visit count
        best_a = None
        best_q = -float("inf")
        for a in self.actions(s0):
            q = self.Q_sa[(s0, a)]
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    # ----- Core simulation (Selection → Expansion → Simulation → Backup) -----
    def _simulate(self, s, depth):
        if depth == 0:
            return 0.0

        A = list(self.actions(s))
        if not A:
            return 0.0

        # If state unvisited (expansion), do a rollout to initialize a value
        if self.N_s[s] == 0:
            v = self._rollout(s, self.cfg.max_depth)
            self.N_s[s] += 1
            return v

        # Selection with UCB1
        a = self._ucb_action(s, A)

        # One-step transition from generative model
        s_next, r, done = self.step(s, a)
        self.children[s].add((a, s_next))

        if done:
            q_return = r
        else:
            q_return = r + self.cfg.gamma * self._simulate(s_next, depth - 1)

        # Backup
        key = (s, a)
        self.N_sa[key] += 1
        self.N_s[s] += 1
        # Incremental mean
        self.Q_sa[key] += (q_return - self.Q_sa[key]) / self.N_sa[key]
        return q_return

    # ----- Policies -----
    def _ucb_action(self, s, A):
        # UCB1: Q + c * sqrt(ln N(s) / N(s,a)); treat N(s,a)=0 as +inf bonus
        ln_Ns = math.log(max(1, self.N_s[s]))
        best, best_score = None, -float("inf")
        for a in A:
            n_sa = self.N_sa[(s, a)]
            if n_sa == 0:
                return a  # optimistic exploration
            exploit = self.Q_sa[(s, a)]
            explore = self.cfg.c_ucb * math.sqrt(ln_Ns / n_sa)
            score = exploit + explore
            if score > best_score:
                best, best_score = a, score
        return best

    def _random_rollout_policy(self, s):
        # Default rollout: pick a random legal action (or None if terminal)
        acts = list(self.actions(s))
        return random.choice(acts) if acts else None

    def _rollout(self, s, depth):
        """Default rollout with discounting."""
        ret, g = 0.0, 1.0
        d = depth
        state = s
        while d > 0:
            a = self.rollout_policy(state)
            if a is None:
                break
            state, r, done = self.step(state, a)
            ret += g * r
            g *= self.cfg.gamma
            if done:
                break
            d -= 1
        return ret


# -------------------- Minimal example: stochastic chain --------------------
if __name__ == "__main__":
    # Toy environment: states 0..4, goal at 4 with reward +1, otherwise 0.
    # Actions: {+1, -1} with 90% success, 10% slip (do nothing).
    S_MAX = 4

    def actions(s):
        return [] if s == S_MAX else [-1, +1]

    def step(s, a):
        if s == S_MAX:
            return s, 0.0, True
        slip = (random.random() < 0.10)
        s_next = s if slip else max(0, min(S_MAX, s + a))
        r = 1.0 if s_next == S_MAX else 0.0
        done = (s_next == S_MAX)
        return s_next, r, done

    # Slightly smarter rollout: prefer +1
    def rollout_policy(s):
        A = actions(s)
        if not A:
            return None
        return +1 if +1 in A else random.choice(A)

    mcts = MCTS(actions, step, rollout_policy, MCTSConfig(simulations=500, max_depth=20, c_ucb=1.2, gamma=0.99))
    s = 0
    total = 0.0
    steps = 0
    while s != S_MAX and steps < 30:
        a = mcts.plan(s)
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    print(f"Reached goal={s==S_MAX} in {steps} steps, total reward={total}")
