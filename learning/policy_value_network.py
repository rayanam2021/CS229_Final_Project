"""
AlphaZero-inspired Policy-Value Neural Network for Active Sensing.

This module implements a neural network that predicts:
1. Policy π(a|s) - probability distribution over 13 possible actions
2. Value V(s) - expected discounted return from state s

The network takes as input:
- Orbital state (ROE): 6 dimensions
- Belief grid: 3D voxel probabilities (flattened or encoded)
- Derived features: position magnitude, action space encoding, etc.

The architecture uses shared hidden layers, then splits into:
- Policy head: outputs logits for 13 actions
- Value head: outputs scalar value estimate
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PolicyValueNetwork(nn.Module):
    """
    AlphaZero-style policy-value network for orbital active sensing.

    Input:
        - orbital_state: (batch, 6) ROE coordinates
        - belief_grid: (batch, grid_size^3) flattened voxel probabilities
        - (optional) derived_features: (batch, num_features)

    Output:
        - policy_logits: (batch, 13) action logits
        - value: (batch, 1) scalar value estimate
    """

    def __init__(
        self,
        grid_dims: Tuple[int, int, int] = (20, 20, 20),
        hidden_dim: int = 256,
        num_actions: int = 13,
        num_residual_blocks: int = 3,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize the policy-value network.

        Args:
            grid_dims: Dimensions of the voxel grid (assumed cubic)
            hidden_dim: Size of hidden layers
            num_actions: Number of possible actions (default 13 for RTN maneuvers)
            num_residual_blocks: Number of residual blocks in trunk
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.grid_dims = grid_dims
        self.grid_size = grid_dims[0]  # Assuming cubic grid
        self.grid_flat_size = int(np.prod(grid_dims))
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.use_batch_norm = use_batch_norm

        # Input encoding
        self.orbital_state_dim = 6  # ROE coordinates
        self.input_dim = self.orbital_state_dim + self.grid_flat_size

        # Trunk: shared feature extraction
        self.trunk = self._build_trunk(num_residual_blocks, dropout_rate)

        # Policy head: outputs logits for each action
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Value head: outputs scalar value estimate
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _build_trunk(self, num_residual_blocks: int, dropout_rate: float) -> nn.Module:
        """Build the trunk with initial layer and residual blocks."""
        layers = []

        # Initial projection layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.ReLU())

        # Residual blocks
        for _ in range(num_residual_blocks):
            layers.append(
                ResidualBlock(self.hidden_dim, self.use_batch_norm, dropout_rate)
            )

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize network weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        orbital_state: torch.Tensor,
        belief_grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            orbital_state: (batch, 6) ROE coordinates
            belief_grid: (batch, grid_size^3) or (batch, grid_size, grid_size, grid_size)

        Returns:
            policy_logits: (batch, 13)
            value: (batch, 1)
        """
        batch_size = orbital_state.size(0)

        # Flatten belief grid if needed
        if belief_grid.dim() > 2:
            belief_flat = belief_grid.view(batch_size, -1)
        else:
            belief_flat = belief_grid

        # Concatenate inputs
        x = torch.cat([orbital_state, belief_flat], dim=1)

        # Trunk: shared feature extraction
        features = self.trunk(x)

        # Policy and value heads
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value


class ResidualBlock(nn.Module):
    """Residual block with batch norm and dropout."""

    def __init__(self, hidden_dim: int, use_batch_norm: bool = True, dropout_rate: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out = out + residual
        out = F.relu(out)

        return out


class ConvPolicyValueNetwork(nn.Module):
    """
    Policy-Value network with CNN feature extraction for grid-based belief.

    Uses 3D convolutions to process the voxel grid belief, then combines with
    orbital state for policy and value prediction.
    """

    def __init__(
        self,
        grid_dims: Tuple[int, int, int] = (20, 20, 20),
        hidden_dim: int = 128,
        num_actions: int = 13,
        num_conv_filters: int = 32,
        use_batch_norm: bool = True,
    ):
        """
        Initialize conv-based policy-value network.

        Args:
            grid_dims: Dimensions of the voxel grid
            hidden_dim: Size of MLP hidden layers
            num_actions: Number of actions (13)
            num_conv_filters: Number of conv filters per layer
            use_batch_norm: Use batch normalization
        """
        super().__init__()

        self.grid_dims = grid_dims
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # 3D Convolutional feature extraction
        self.conv1 = nn.Conv3d(1, num_conv_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(num_conv_filters, num_conv_filters * 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(num_conv_filters) if use_batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm3d(num_conv_filters * 2) if use_batch_norm else nn.Identity()

        # Calculate flattened conv output size
        # After 2 conv layers with padding=1, spatial dims stay the same
        conv_output_size = num_conv_filters * 2 * int(np.prod(grid_dims))

        # Combine conv features with orbital state
        self.fc_trunk = nn.Sequential(
            nn.Linear(conv_output_size + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy and value heads
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        orbital_state: torch.Tensor,
        belief_grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            orbital_state: (batch, 6)
            belief_grid: (batch, grid_size, grid_size, grid_size) or (batch, 1, grid_size, grid_size, grid_size)

        Returns:
            policy_logits: (batch, 13)
            value: (batch, 1)
        """
        batch_size = orbital_state.size(0)

        # Ensure belief_grid has channel dimension
        if belief_grid.dim() == 4:
            belief_grid = belief_grid.unsqueeze(1)

        # Conv feature extraction
        x_conv = F.relu(self.bn1(self.conv1(belief_grid)))
        x_conv = F.relu(self.bn2(self.conv2(x_conv)))
        x_conv_flat = x_conv.view(batch_size, -1)

        # Combine with orbital state
        x = torch.cat([x_conv_flat, orbital_state], dim=1)

        # Trunk
        features = self.fc_trunk(x)

        # Heads
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value
