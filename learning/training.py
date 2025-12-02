import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import os

class SelfPlayTrainer:
    """
    Trainer for AlphaZero-style self-play with MCTS and neural network.
    Updated to store full transition data (s, a, r, s', t).
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
        self.network = network.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir or "./checkpoints"
        self.gradient_clip_norm = gradient_clip_norm
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )

        # Replay buffer stores: (state, belief, policy, return, action, reward, next_state, time)
        self.replay_buffer: List[Tuple] = []
        self.max_buffer_size = max_buffer_size

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
        action: np.ndarray,       
        reward: float,            
        next_orbital_state: np.ndarray, 
        time: float               
    ):
        """
        Add a complete transition sample to the replay buffer.
        """
        sample = (orbital_state, belief_grid, policy_mcts, episode_return, action, reward, next_orbital_state, time)
        self.replay_buffer.append(sample)

        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size :]

    def clear_replay_buffer(self):
        self.replay_buffer = []

    def prepare_batch(self, batch_size: int = 32):
        """
        Prepare a batch from the replay buffer.
        Returns tensors for all stored fields.
        """
        if len(self.replay_buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.replay_buffer)} < {batch_size}")

        indices = np.random.choice(len(self.replay_buffer), size=batch_size, replace=False)
        samples = [self.replay_buffer[i] for i in indices]

        # 1. State Inputs (for Network)
        orbital_states = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32, device=self.device)
        
        belief_grids_list = []
        for s in samples:
            grid = s[1]
            if isinstance(grid, torch.Tensor):
                belief_grids_list.append(grid.detach())
            else:
                belief_grids_list.append(torch.tensor(grid, dtype=torch.float32))

        if belief_grids_list and isinstance(belief_grids_list[0], torch.Tensor):
            belief_grids = torch.stack(belief_grids_list).to(self.device)
        else:
            belief_grids = torch.tensor(np.array(belief_grids_list), dtype=torch.float32, device=self.device)

        # 2. Targets (for Loss)
        policies_mcts = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32, device=self.device)
        returns = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32, device=self.device)

        # 3. Extra Data (unused in standard loss but kept for potential aux tasks)
        actions = torch.tensor(np.array([s[4] for s in samples]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([s[5] for s in samples]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([s[6] for s in samples]), dtype=torch.float32, device=self.device)
        times = torch.tensor(np.array([s[7] for s in samples]), dtype=torch.float32, device=self.device)

        return orbital_states, belief_grids, policies_mcts, returns, actions, rewards, next_states, times

    def train_step(self, batch_size: int = 32, value_weight: float = 1.0, policy_weight: float = 1.0) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        # Unpack all data
        orbital_states, belief_grids, policies_mcts, returns, _, _, _, _ = self.prepare_batch(batch_size)

        # Forward pass
        policy_logits, values = self.network(orbital_states, belief_grids)

        # Policy loss
        log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies_mcts * log_probs).sum(dim=1).mean()

        # Value loss
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

    def train_epoch(self, num_batches: int = 100, batch_size: int = 32, value_weight: float = 1.0, policy_weight: float = 1.0) -> Dict[str, float]:
        self.network.train()
        epoch_losses = {"policy_loss": [], "value_loss": [], "total_loss": []}

        for _ in range(num_batches):
            losses = self.train_step(batch_size, value_weight, policy_weight)
            for key in epoch_losses:
                epoch_losses[key].append(losses[key])

        # --- UPDATED: Calculate averages AND store them in history ---
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        # Append to internal history
        for key in ["policy_loss", "value_loss", "total_loss"]:
            if key in self.training_history:
                self.training_history[key].append(float(avg_losses[key]))
        
        return avg_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "training_history": self.training_history,
        }
        name = "best" if is_best else f"checkpoint_ep_{epoch}"
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.training_history = checkpoint["training_history"]

    def step_scheduler(self):
        self.scheduler.step()