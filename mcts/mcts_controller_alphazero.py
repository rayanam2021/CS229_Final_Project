"""
High-level controller for AlphaZero-style MCTS planning.

Provides a simple interface to MCTSAlphaZero with sensible defaults
for orbital active sensing problems.
"""

import numpy as np
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from mcts.mcts_alphazero import MCTSAlphaZero
from mcts.orbital_mdp_model_gpu import OrbitalMCTSModelGPU


class MCTSControllerAlphaZero:
    """
    High-level interface for AlphaZero-style MCTS planning.

    Combines orbital dynamics model with policy-value network
    for efficient planning using PUCT formula.
    """

    def __init__(
        self,
        mcts_iters: int = 500,
        horizon: int = 20,
        use_policy_guidance: bool = True,
        bootstrap_threshold: float = 0.8,
        batch_rollouts: bool = True,
        batch_size: int = 10,
        parallel_batching: bool = False,
        c: float = 1.4,
        gamma: float = 0.95,
    ):
        """
        Initialize AlphaZero MCTS controller.

        Args:
            mcts_iters: Number of MCTS iterations per search
            horizon: Maximum tree depth
            use_policy_guidance: Enable PUCT with policy priors
            bootstrap_threshold: Confidence threshold for value bootstrapping
            batch_rollouts: Enable batched rollout evaluation
            batch_size: Batch size for rollouts
            parallel_batching: Enable parallel tree search
            c: PUCT exploration constant
            gamma: Discount factor
        """
        self.mcts_iters = mcts_iters
        self.horizon = horizon
        self.use_policy_guidance = use_policy_guidance
        self.bootstrap_threshold = bootstrap_threshold
        self.batch_rollouts = batch_rollouts
        self.batch_size = batch_size
        self.parallel_batching = parallel_batching
        self.c = c
        self.gamma = gamma

        self.mdp = None
        self.network = None
        self.mcts = None

    def setup(
        self,
        mdp: OrbitalMCTSModelGPU,
        network: Optional[nn.Module] = None,
    ):
        """
        Set up the controller with an MDP model and optional network.

        Args:
            mdp: Orbital dynamics model
            network: Policy-value network (optional)
        """
        self.mdp = mdp
        self.network = network

        # Create MCTS instance
        self.mcts = MCTSAlphaZero(
            model=mdp,
            network=network,
            use_policy_guidance=self.use_policy_guidance and (network is not None),
            bootstrap_threshold=self.bootstrap_threshold,
            min_bootstrap_depth=1,
            blend_mode="linear",
            iters=self.mcts_iters,
            max_depth=self.horizon,
            c=self.c,
            gamma=self.gamma,
            batch_rollouts=self.batch_rollouts,
            batch_size=self.batch_size,
            parallel_batching=self.parallel_batching,
            device="cuda",
        )

    def search(
        self,
        state,
        step: int = 0,
        out_folder: str = "./trees",
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Run MCTS planning from current state.

        Args:
            state: Current orbital state
            step: Current simulation step (for logging)
            out_folder: Folder for tree visualizations

        Returns:
            best_action: Recommended action
            best_value: Predicted value of state
            stats: Planning statistics including PUCT usage
        """
        if self.mcts is None:
            raise RuntimeError("Must call setup() before search()")

        best_action, best_value, stats = self.mcts.get_best_root_action(
            state, step, out_folder, return_stats=True
        )

        return best_action, best_value, stats

    def get_stats(self) -> Dict[str, Any]:
        """Get AlphaZero MCTS statistics."""
        if self.mcts is None:
            return {}

        puct_stats = self.mcts.get_puct_stats()
        return {
            "puct_stats": puct_stats,
            "configuration": {
                "mcts_iters": self.mcts_iters,
                "horizon": self.horizon,
                "use_policy_guidance": self.use_policy_guidance,
                "bootstrap_threshold": self.bootstrap_threshold,
                "c": self.c,
                "gamma": self.gamma,
            },
        }
