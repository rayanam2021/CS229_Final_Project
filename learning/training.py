"""
Training loop for AlphaZero-style self-play learning.

Alternates between:
1. MCTS planning with network bootstrapping
2. Network training on MCTS data

The training loss combines policy and value objectives:
    L(θ) = (R − V_θ(s))² − π_MCTS(a|s)⊤ log π_θ(a|s)

where:
    - R: discounted return from episode
    - V_θ(s): network value prediction
    - π_MCTS(a|s): improved policy from MCTS visit counts
    - π_θ(a|s): network policy prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional
import os
from datetime import datetime


class SelfPlayTrainer:
    """
    Trainer for AlphaZero-style self-play with MCTS and neural network.

    Maintains:
    - Policy-value network
    - Replay buffer of (state, π_MCTS, R) tuples
    - Training loop alternating between MCTS and network updates
    """

    def __init__(
        self,
        network: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None,
        max_buffer_size: int = 100_000,
        gradient_clip_norm: float = 1.0,
    ):
        """
        Initialize the trainer.

        Args:
            network: Policy-value network module
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            device: "cuda" or "cpu"
            checkpoint_dir: Directory to save checkpoints
            max_buffer_size: Maximum replay buffer size
            gradient_clip_norm: Gradient clipping threshold
        """
        self.network = network.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir or "./checkpoints"
        self.gradient_clip_norm = gradient_clip_norm
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler (optional)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )

        # Replay buffer
        self.replay_buffer: List[Tuple] = []
        self.max_buffer_size = max_buffer_size

        # Training history
        self.training_history = {
            "epoch": [],
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
            "lr": [],
        }

    def add_to_replay_buffer(
        self,
        orbital_state: np.ndarray,
        belief_grid: np.ndarray,
        policy_mcts: np.ndarray,
        episode_return: float,
    ):
        """
        Add a training sample to the replay buffer.

        Args:
            orbital_state: (6,) ROE coordinates
            belief_grid: (grid_size^3,) or (grid_size, grid_size, grid_size) flattened probabilities
            policy_mcts: (13,) MCTS-improved policy (visit counts normalized)
            episode_return: scalar discounted return R from that point onward
        """
        sample = (orbital_state, belief_grid, policy_mcts, episode_return)
        self.replay_buffer.append(sample)

        # Trim buffer if too large
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size :]

    def clear_replay_buffer(self):
        """Clear the replay buffer."""
        self.replay_buffer = []

    def prepare_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a batch from the replay buffer.

        Returns:
            orbital_states: (batch, 6)
            belief_grids: (batch, grid_flat_size)
            policies_mcts: (batch, 13)
            returns: (batch,)
        """
        if len(self.replay_buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.replay_buffer)} < {batch_size}")

        # Sample randomly from buffer
        indices = np.random.choice(len(self.replay_buffer), size=batch_size, replace=False)
        samples = [self.replay_buffer[i] for i in indices]

        # Unpack and convert to tensors
        orbital_states = torch.tensor(
            np.array([s[0] for s in samples]), dtype=torch.float32, device=self.device
        )
        # Handle belief grids that may be CUDA tensors or numpy arrays
        belief_grids_list = []
        for s in samples:
            grid = s[1]
            if isinstance(grid, torch.Tensor):
                # Keep tensor on its device, move to target device only once
                belief_grids_list.append(grid.detach())
            else:
                # Convert numpy to tensor on target device directly
                belief_grids_list.append(torch.tensor(grid, dtype=torch.float32))

        # Stack and move to target device once (avoid repeated transfers)
        if belief_grids_list and isinstance(belief_grids_list[0], torch.Tensor):
            belief_grids = torch.stack(belief_grids_list).to(self.device)
        else:
            belief_grids = torch.tensor(
                np.array(belief_grids_list), dtype=torch.float32, device=self.device
            )
        policies_mcts = torch.tensor(
            np.array([s[2] for s in samples]), dtype=torch.float32, device=self.device
        )
        returns = torch.tensor(
            np.array([s[3] for s in samples]), dtype=torch.float32, device=self.device
        )

        return orbital_states, belief_grids, policies_mcts, returns

    def train_step(self, batch_size: int = 32, value_weight: float = 1.0, policy_weight: float = 1.0) -> Dict[str, float]:
        """
        Perform one training step on a batch from replay buffer.

        Loss function:
            L(θ) = value_weight * (R − V_θ(s))² − policy_weight * π_MCTS(a|s)⊤ log π_θ(a|s)

        Args:
            batch_size: Size of batch to train on
            value_weight: Weight for value loss
            policy_weight: Weight for policy loss

        Returns:
            Dictionary with loss values
        """
        if len(self.replay_buffer) < batch_size:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "total_loss": 0.0,
            }

        orbital_states, belief_grids, policies_mcts, returns = self.prepare_batch(batch_size)

        # Forward pass
        policy_logits, values = self.network(orbital_states, belief_grids)

        # Policy loss: cross-entropy with MCTS policy
        log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies_mcts * log_probs).sum(dim=1).mean()

        # Value loss: MSE between predicted and actual return
        value_loss = torch.nn.functional.mse_loss(values.squeeze(), returns)

        # Total loss
        total_loss = policy_weight * policy_loss + value_weight * value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.gradient_clip_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def train_epoch(
        self,
        num_batches: int = 100,
        batch_size: int = 32,
        value_weight: float = 1.0,
        policy_weight: float = 1.0,
    ) -> Dict[str, float]:
        """
        Train for one epoch (multiple steps on the replay buffer).

        Args:
            num_batches: Number of batches to process
            batch_size: Batch size
            value_weight: Weight for value loss
            policy_weight: Weight for policy loss

        Returns:
            Average losses for the epoch
        """
        self.network.train()

        epoch_losses = {
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
        }

        for _ in range(num_batches):
            losses = self.train_step(batch_size, value_weight, policy_weight)
            for key in epoch_losses:
                epoch_losses[key].append(losses[key])

        # Average
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

        return avg_losses

    def evaluate(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate network on a batch from replay buffer (no gradients).

        Args:
            batch_size: Batch size

        Returns:
            Dictionary with loss values
        """
        self.network.eval()

        with torch.no_grad():
            orbital_states, belief_grids, policies_mcts, returns = self.prepare_batch(batch_size)
            policy_logits, values = self.network(orbital_states, belief_grids)

            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -(policies_mcts * log_probs).sum(dim=1).mean()
            value_loss = torch.nn.functional.mse_loss(values.squeeze(), returns)
            total_loss = policy_loss + value_loss

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save network checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "training_history": self.training_history,
        }

        name = "best" if is_best else f"epoch_{epoch}"
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{name}.pt")
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load network from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.training_history = checkpoint["training_history"]

    def get_value_and_policy(
        self,
        orbital_state: np.ndarray,
        belief_grid: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Get value estimate and policy prediction for a state (no gradients).

        Args:
            orbital_state: (6,)
            belief_grid: (grid_size^3,) or (grid_size, grid_size, grid_size)

        Returns:
            value: scalar
            policy_probs: (13,) softmax probabilities over actions
        """
        self.network.eval()

        with torch.no_grad():
            orbital_t = torch.tensor(orbital_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            belief_t = torch.tensor(belief_grid, dtype=torch.float32, device=self.device).unsqueeze(0)

            if belief_t.dim() == 3:  # (1, grid_size, grid_size, grid_size)
                belief_t = belief_t.view(1, -1)
            elif belief_t.dim() == 2 and belief_t.shape[1] != orbital_state.shape[0]:
                # Already flattened, just reshape to (1, -1)
                belief_t = belief_t.view(1, -1)

            policy_logits, value = self.network(orbital_t, belief_t)

            value_scalar = value.squeeze().item()
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze().cpu().numpy()

        return value_scalar, policy_probs

    def log_training_history(self, epoch: int, losses: Dict[str, float]):
        """Log training history."""
        self.training_history["epoch"].append(epoch)
        for key in ["policy_loss", "value_loss", "total_loss"]:
            if key in losses:
                self.training_history[key].append(losses[key])
        self.training_history["lr"].append(self.optimizer.param_groups[0]["lr"])

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()


class ReplayBuffer:
    """Simple replay buffer for storing training samples."""

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self.buffer: List[Tuple] = []

    def add(
        self,
        orbital_state: np.ndarray,
        belief_grid: np.ndarray,
        policy_mcts: np.ndarray,
        episode_return: float,
    ):
        """Add sample to buffer."""
        self.buffer.append((orbital_state, belief_grid, policy_mcts, episode_return))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples: {len(self.buffer)} < {batch_size}")

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]

        orbital_states = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
        belief_grids = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
        policies_mcts = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32)
        returns = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32)

        return orbital_states, belief_grids, policies_mcts, returns

    def clear(self):
        """Clear buffer."""
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
