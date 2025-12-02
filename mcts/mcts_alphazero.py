"""
True AlphaZero-style MCTS with PUCT formula.

Implements the exact MCTS algorithm from AlphaGo Zero / AlphaZero papers:
- Uses policy network priors to guide tree search (PUCT formula)
- Combines Q-values with policy exploration bonus
- Gradually transitions from policy guidance to empirical values as simulations increase

Key innovation: Policy P(a|s) directly influences action selection during planning.

PUCT Formula:
    PUCT(s,a) = Q(s,a) + U(s,a)

    where:
    U(s,a) = c * P(a|s) * √N(s) / (1 + N(s,a))

    - Q(s,a): Mean value from rollouts/network
    - P(a|s): Policy prior from neural network
    - N(s): Total visits to state
    - N(s,a): Visits to action a
    - c: Exploration constant (typically 1.0-2.0)

Reference: Silver et al., "Mastering Chess and Shogi by Self-Play with a General
Reinforcement Learning Algorithm" (2018)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from mcts.mcts_parallel import MCTSParallel, Node


class MCTSAlphaZero(MCTSParallel):
    """
    True AlphaZero-style MCTS with PUCT formula for policy-guided tree search.

    Key differences from standard UCB1:
    1. Policy priors P(a|s) directly boost exploration of promising actions
    2. Exploration bonus decreases as visits increase (N_a dependent)
    3. Network confidence implicitly built into the formula
    """

    def __init__(
        self,
        model,
        network: Optional[nn.Module] = None,
        use_policy_guidance: bool = True,
        bootstrap_threshold: float = 0.8,
        min_bootstrap_depth: int = 1,
        blend_mode: str = "linear",
        iters: int = 1000,
        max_depth: int = 5,
        c: float = 1.4,
        gamma: float = 1.0,
        batch_rollouts: bool = True,
        batch_size: int = 10,
        parallel_batching: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize AlphaZero-style MCTS with PUCT.

        Args:
            model: MDP model
            network: Policy-value neural network
            use_policy_guidance: Enable policy priors in PUCT formula
            bootstrap_threshold: Confidence threshold for value bootstrapping (0-1)
            min_bootstrap_depth: Minimum tree depth before using network value
            blend_mode: How to blend rollout and network values
            iters: Number of MCTS iterations
            max_depth: Tree depth limit
            c: PUCT exploration constant
            gamma: Discount factor
            batch_rollouts: Enable batched rollouts
            batch_size: Batch size for rollouts
            parallel_batching: Enable parallel tree search
            device: "cuda" or "cpu"
        """
        super().__init__(
            model=model,
            iters=iters,
            max_depth=max_depth,
            c=c,
            gamma=gamma,
            batch_rollouts=batch_rollouts,
            batch_size=batch_size,
            parallel_batching=parallel_batching,
        )

        self.network = network
        self.use_policy_guidance = use_policy_guidance and (network is not None)
        self.bootstrap_threshold = bootstrap_threshold
        self.min_bootstrap_depth = min_bootstrap_depth
        self.blend_mode = blend_mode
        self.device = device

        # Statistics for monitoring policy guidance
        self.policy_guidance_count = 0
        self.network_value_count = 0
        self.rollout_value_count = 0
        self.blend_value_count = 0

    def _get_policy_and_value(
        self,
        state,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Get policy priors and value prediction from the network.

        Returns:
            policy_probs: (13,) policy probabilities for action guidance
            value: Scalar value estimate
            confidence: Confidence in the prediction (0-1)
        """
        if not self.use_policy_guidance or self.network is None:
            # Return uniform policy if network unavailable
            return np.ones(13) / 13, 0.0, 0.0

        try:
            self.network.eval()
            with torch.no_grad():
                # Get policy and value from network
                orbital_state = torch.tensor(
                    state.roe,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)

                belief_grid = torch.tensor(
                    state.grid.belief,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)

                # Handle 3D grid flattening
                if belief_grid.dim() > 2:
                    belief_grid = belief_grid.view(1, -1)

                policy_logits, value = self.network(orbital_state, belief_grid)

                value_scalar = float(value.squeeze().cpu().item())
                policy_probs = torch.softmax(policy_logits, dim=1).squeeze().cpu().numpy()

                # Estimate confidence based on policy entropy
                policy_entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-10))
                max_entropy = np.log(len(policy_probs))
                confidence = 1.0 - (policy_entropy / max_entropy)

                return policy_probs, value_scalar, confidence
        except Exception as e:
            print(f"Network inference failed: {e}")
            return np.ones(13) / 13, 0.0, 0.0

    def _select_puct_action_index(self, node):
        """
        PUCT action selection (AlphaZero-style).

        PUCT(s,a) = Q(s,a) + U(s,a)

        where U(s,a) = c * P(a|s) * √N(s) / (1 + N(s,a))

        This formula:
        - Encourages exploring actions with high policy probability
        - Reduces exploration of actions as they are visited more
        - Naturally balances policy priors with empirical Q-values
        """
        if not self.use_policy_guidance:
            # Fall back to standard UCB1 if policy guidance disabled
            return self._select_ucb1_action_index(node)

        # Get policy priors from network
        policy_probs, _, _ = self._get_policy_and_value(node.state)

        total_N = max(node.N, 1)  # Avoid division by zero

        # PUCT formula components
        exploitation = node.Q_sa  # Q-values from rollouts
        exploration = self.c * policy_probs * np.sqrt(total_N) / (1.0 + node.N_sa)

        # Combined PUCT score
        puct_sa = exploitation + exploration

        # Select action with highest PUCT score
        action_idx = int(np.argmax(puct_sa))
        self.policy_guidance_count += 1

        return action_idx

    def _blend_values(
        self,
        rollout_value: float,
        network_value: float,
        confidence: float,
    ) -> float:
        """
        Blend rollout and network values based on confidence.

        Same blending strategy as MCTSWithNetworkBootstrap.
        """
        if self.blend_mode == "rollout":
            return rollout_value
        elif self.blend_mode == "network":
            return network_value
        elif self.blend_mode == "linear":
            # Linear blend: more confident = more network weight
            if confidence >= self.bootstrap_threshold:
                return network_value
            else:
                # Gradual transition
                weight = confidence / self.bootstrap_threshold
                return (1 - weight) * rollout_value + weight * network_value
        elif self.blend_mode == "weighted":
            # Confidence-weighted average
            alpha = confidence
            return (1 - alpha) * rollout_value + alpha * network_value
        else:
            return rollout_value

    def _rollout_with_bootstrap(
        self,
        state,
        depth: int,
        use_network: bool = True,
    ) -> float:
        """
        Rollout with optional network value bootstrapping.

        Uses rollout initially, then transitions to network predictions
        when confidence is high and depth is sufficient.

        Args:
            state: Current state
            depth: Current depth in tree
            use_network: Whether to consider network bootstrap

        Returns:
            Value estimate (rollout, network, or blend)
        """
        # Decide whether to use network
        should_use_network = (
            use_network
            and self.use_policy_guidance
            and depth >= self.min_bootstrap_depth
        )

        if not should_use_network:
            # Standard rollout
            value = self._rollout(state, depth)
            self.rollout_value_count += 1
            return value

        # Get network prediction
        _, network_value, confidence = self._get_policy_and_value(state)

        # Decide whether network confidence is sufficient
        if confidence >= self.bootstrap_threshold:
            # Use network value directly
            self.network_value_count += 1
            return network_value
        else:
            # Run rollout and blend with network
            rollout_value = self._rollout(state, depth)
            blended_value = self._blend_values(rollout_value, network_value, confidence)
            self.blend_value_count += 1
            return blended_value

    def _rollout_batch_with_bootstrap(
        self,
        states: List,
        depths: List[int],
        use_network: bool = True,
    ) -> np.ndarray:
        """
        Batch rollout with optional network value bootstrapping.

        Separates states that can use network predictions from those
        that need full rollouts, processes each group efficiently.
        """
        if not use_network or not self.use_policy_guidance:
            # Fall back to standard batch rollout
            return self._rollout_batch(states, depths)

        batch_size = len(states)
        returns = np.zeros(batch_size)

        # Separate states by whether they can use network bootstrap
        network_indices = []
        rollout_indices = []

        for i, (state, depth) in enumerate(zip(states, depths)):
            if depth >= self.min_bootstrap_depth:
                network_indices.append(i)
            else:
                rollout_indices.append(i)

        # Process network predictions for eligible states
        if network_indices:
            network_states = [states[i] for i in network_indices]
            network_values = []

            for state in network_states:
                _, network_value, confidence = self._get_policy_and_value(state)
                if confidence >= self.bootstrap_threshold:
                    network_values.append(network_value)
                    self.network_value_count += 1
                else:
                    # Fall back to rollout for this state
                    rollout_value = self._rollout(state, depths[network_indices[len(network_values)]])
                    blended = self._blend_values(rollout_value, network_value, confidence)
                    network_values.append(blended)
                    self.blend_value_count += 1

            for idx, val in zip(network_indices, network_values):
                returns[idx] = val

        # Process standard rollouts for remaining states
        if rollout_indices:
            rollout_states = [states[i] for i in rollout_indices]
            rollout_depths = [depths[i] for i in rollout_indices]
            rollout_values = self._rollout_batch(rollout_states, rollout_depths)

            for idx, val in zip(rollout_indices, rollout_values):
                returns[idx] = val
                self.rollout_value_count += 1

        return returns

    def _search(self, node, depth):
        """Single MCTS search iteration with PUCT policy guidance."""
        if depth == self.max_depth:
            value = 0.0
            self._backpropagate(node, value)
            return value

        if node.untried_action_indices:
            # Expansion phase: try untried action
            a_idx = node.untried_action_indices.pop()
            child = self._expand(node, a_idx)

            # Use network bootstrap when available
            if self.batch_rollouts and depth + 1 < self.max_depth:
                states = [child.state]
                depths = [depth + 1]
                returns = self._rollout_batch_with_bootstrap(states, depths)
                value = returns[0]
            else:
                value = self._rollout_with_bootstrap(child.state, depth + 1)

            self._backpropagate(child, value)
            return value

        if node.children:
            # Selection phase: use PUCT to select child
            a_idx = self._select_puct_action_index(node)

            child = None
            for ch in node.children:
                if ch.action_index == a_idx:
                    child = ch
                    break

            return self._search(child, depth + 1)

        value = 0.0
        self._backpropagate(node, value)
        return value

    def get_best_root_action(self, root_state, step, out_folder, return_stats=True):
        """Main MCTS search interface with PUCT and statistics."""
        # Reset statistics
        self.policy_guidance_count = 0
        self.network_value_count = 0
        self.rollout_value_count = 0
        self.blend_value_count = 0

        # Run MCTS
        root_actions = self.mdp.actions(root_state)
        root = Node(root_state, actions=root_actions, action_index=None, parent=None)

        for _ in range(self.max_path_searches):
            if self.parallel_batching:
                self._search_parallel_batch(root, depth=0)
            else:
                self._search(root, depth=0)

        if len(root.actions) == 0:
            return np.zeros(3), 0.0

        best_idx = int(np.argmax(root.Q_sa))
        best_action = root.actions[best_idx]
        best_value = float(root.Q_sa[best_idx])

        if return_stats:
            total_value_evals = (
                self.network_value_count +
                self.rollout_value_count +
                self.blend_value_count
            )

            stats = {
                "root_N": int(root.N),
                "root_Q_sa": root.Q_sa.copy(),
                "root_N_sa": root.N_sa.copy(),
                "best_idx": best_idx,
                "best_action": best_action,
                "predicted_value": best_value,
                "alphazero_stats": {
                    "policy_guidance_selections": self.policy_guidance_count,
                    "network_value_uses": self.network_value_count,
                    "rollout_value_uses": self.rollout_value_count,
                    "blend_value_uses": self.blend_value_count,
                    "total_value_evals": total_value_evals,
                    "network_fraction": (
                        self.network_value_count / total_value_evals
                        if total_value_evals > 0 else 0.0
                    ),
                },
            }
            return best_action, best_value, stats

        return best_action, best_value

    def get_puct_stats(self) -> Dict[str, float]:
        """Get statistics on PUCT usage and value bootstrapping."""
        total_value = (
            self.network_value_count +
            self.rollout_value_count +
            self.blend_value_count
        )
        return {
            "policy_guidance_selections": self.policy_guidance_count,
            "network_value_uses": self.network_value_count,
            "rollout_value_uses": self.rollout_value_count,
            "blend_value_uses": self.blend_value_count,
            "total_value_evaluations": total_value,
            "network_fraction": (
                self.network_value_count / total_value
                if total_value > 0 else 0.0
            ),
            "rollout_fraction": (
                self.rollout_value_count / total_value
                if total_value > 0 else 0.0
            ),
            "blend_fraction": (
                self.blend_value_count / total_value
                if total_value > 0 else 0.0
            ),
        }
