import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNetwork(nn.Module):
    def __init__(self, grid_dims=(20, 20, 20), num_actions=13, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv3d(16, 32, 3, 2, 1)
        self.conv3 = nn.Conv3d(32, 64, 3, 2, 1)
        self.fc_grid = nn.Linear(5 * 5 * 5 * 64, hidden_dim)
        self.fc_roe = nn.Linear(6, hidden_dim)
        self.fc_shared = nn.Linear(hidden_dim * 2, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, roe, grid):
        if grid.ndim == 4: grid = grid.unsqueeze(1)
        elif grid.ndim == 3: grid = grid.unsqueeze(0).unsqueeze(0)
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc_grid(x.view(x.size(0), -1)))
        y = F.relu(self.fc_roe(roe))
        shared = F.relu(self.fc_shared(torch.cat([x, y], dim=1)))
        return self.policy_head(shared), self.value_head(shared)