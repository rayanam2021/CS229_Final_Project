"""
MCTS integration with neural network bootstrapping.

This module provides two approaches:

1. MCTSWithNetworkBootstrap (standard MCTS with value bootstrapping):
   - Uses standard UCB1 for action selection
   - Network value predictions replace expensive rollouts
   - Policy used for confidence estimation only

2. MCTSAlphaZero (true AlphaZero-style MCTS):
   - Uses PUCT formula with policy priors
   - Policy network DIRECTLY guides action selection
   - Network value predictions bootstrap rollouts
   - See mcts/mcts_alphazero.py for full implementation

For true AlphaZero behavior, use MCTSAlphaZero instead.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from mcts.mcts_parallel import MCTSParallel, Node


class MCTSWithNetworkBootstrap(MCTSParallel):
    """
    MCTS with neural network value bootstrapping.

    Extends MCTSParallel to optionally use network value predictions
    instead of full rollouts, reducing computation while maintaining
    search quality.

    The network is used when:
    - use_network_bootstrap=True
    - Network value accuracy is above bootstrap_threshold
    - Current depth is >= min_bootstrap_depth
    """

    def __init__(
        self,
        model,
        network: Optional[nn.Module] = None,
        use_network_bootstrap: bool = False,
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
        Initialize MCTS with network bootstrapping.

        Args:
            model: MDP model
            network: Policy-value neural network (optional)
            use_network_bootstrap: Enable network value bootstrapping
            bootstrap_threshold: Confidence threshold for using network values (0-1)
                Higher values = more conservative, rely on rollouts longer
            min_bootstrap_depth: Minimum tree depth before using network bootstrap
            blend_mode: How to combine rollout and network values:
                - "rollout": Use only rollouts
                - "network": Use only network
                - "linear": Linearly blend based on confidence
                - "weighted": Weighted average by visit counts
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
        self.use_network_bootstrap = use_network_bootstrap and (network is not None)
        self.bootstrap_threshold = bootstrap_threshold
        self.min_bootstrap_depth = min_bootstrap_depth
        self.blend_mode = blend_mode
        self.device = device

        # Statistics for monitoring network confidence
        self.network_use_count = 0
        self.rollout_use_count = 0
        self.blend_use_count = 0

    def _get_network_value(
        self,
        state,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Get value and policy predictions from the network.

        Returns:
            value: Scalar value estimate
            policy_probs: (13,) policy probabilities
            confidence: Estimated confidence in the prediction (0-1)
        """
        if not self.use_network_bootstrap or self.network is None:
            return 0.0, np.ones(13) / 13, 0.0

        try:
            self.network.eval()
            with torch.no_grad():
                # Get value and policy from network
                # This assumes the state has attributes for orbital_state and belief_grid
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
                # High entropy = low confidence, Low entropy = high confidence
                policy_entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-10))
                max_entropy = np.log(len(policy_probs))
                confidence = 1.0 - (policy_entropy / max_entropy)

                return value_scalar, policy_probs, confidence
        except Exception as e:
            print(f"Network inference failed: {e}")
            return 0.0, np.ones(13) / 13, 0.0

    def _blend_values(
        self,
        rollout_value: float,
        network_value: float,
        confidence: float,
    ) -> float:
        """
        Blend rollout and network values.

        Args:
            rollout_value: Value from rollout
            network_value: Value from network
            confidence: Network confidence (0-1)

        Returns:
            Blended value estimate
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
        Rollout with optional network bootstrapping.

        Uses full rollout initially, then transitions to network predictions
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
            and self.use_network_bootstrap
            and depth >= self.min_bootstrap_depth
        )

        if not should_use_network:
            # Standard rollout
            value = self._rollout(state, depth)
            self.rollout_use_count += 1
            return value

        # Get network prediction
        network_value, policy_probs, confidence = self._get_network_value(state)

        # Decide whether network confidence is sufficient
        if confidence >= self.bootstrap_threshold:
            # Use network value directly
            self.network_use_count += 1
            return network_value
        else:
            # Run rollout and blend with network
            rollout_value = self._rollout(state, depth)
            blended_value = self._blend_values(rollout_value, network_value, confidence)
            self.blend_use_count += 1
            return blended_value

    def _rollout_batch_with_bootstrap(
        self,
        states: List,
        depths: List[int],
        use_network: bool = True,
    ) -> np.ndarray:
        """
        Batch rollout with optional network bootstrapping.

        Separates states that can use network predictions from those
        that need full rollouts, processes each group efficiently.

        Args:
            states: List of states
            depths: List of depths
            use_network: Whether to consider network bootstrap

        Returns:
            Array of value estimates
        """
        if not use_network or not self.use_network_bootstrap:
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
                network_value, _, confidence = self._get_network_value(state)
                if confidence >= self.bootstrap_threshold:
                    network_values.append(network_value)
                    self.network_use_count += 1
                else:
                    # Fall back to rollout for this state
                    rollout_value = self._rollout(state, depths[network_indices[len(network_values)]])
                    blended = self._blend_values(rollout_value, network_value, confidence)
                    network_values.append(blended)
                    self.blend_use_count += 1

            for idx, val in zip(network_indices, network_values):
                returns[idx] = val

        # Process standard rollouts for remaining states
        if rollout_indices:
            rollout_states = [states[i] for i in rollout_indices]
            rollout_depths = [depths[i] for i in rollout_indices]
            rollout_values = self._rollout_batch(rollout_states, rollout_depths)

            for idx, val in zip(rollout_indices, rollout_values):
                returns[idx] = val
                self.rollout_use_count += 1

        return returns

    def _search(self, node, depth):
        """Single MCTS search iteration with network bootstrap."""
        if depth == self.max_depth:
            value = 0.0
            self._backpropagate(node, value)
            return value

        if node.untried_action_indices:
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
            a_idx = self._select_ucb1_action_index(node)

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
        """Main MCTS search interface with statistics."""
        # Reset statistics
        self.network_use_count = 0
        self.rollout_use_count = 0
        self.blend_use_count = 0

        # Run standard MCTS
        best_action, best_value, stats = super().get_best_root_action(
            root_state, step, out_folder, return_stats=True
        )

        # Add bootstrap statistics
        if return_stats:
            total_uses = self.network_use_count + self.rollout_use_count + self.blend_use_count
            stats["network_bootstrap_stats"] = {
                "network_uses": self.network_use_count,
                "rollout_uses": self.rollout_use_count,
                "blend_uses": self.blend_use_count,
                "total_evaluations": total_uses,
                "network_fraction": (
                    self.network_use_count / total_uses if total_uses > 0 else 0.0
                ),
            }
            return best_action, best_value, stats

        return best_action, best_value

    def get_bootstrap_stats(self) -> Dict[str, float]:
        """Get statistics on network bootstrap usage."""
        total = self.network_use_count + self.rollout_use_count + self.blend_use_count
        return {
            "network_uses": self.network_use_count,
            "rollout_uses": self.rollout_use_count,
            "blend_uses": self.blend_use_count,
            "total_evaluations": total,
            "network_fraction": self.network_use_count / total if total > 0 else 0.0,
            "rollout_fraction": self.rollout_use_count / total if total > 0 else 0.0,
            "blend_fraction": self.blend_use_count / total if total > 0 else 0.0,
        }
