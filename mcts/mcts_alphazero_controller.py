import numpy as np
import torch
import os
# Requires: pip install graphviz
from graphviz import Digraph

class Node:
    def __init__(self, state, parent=None, action_idx=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action_idx = action_idx
        self.prior = prior  # P(s, a) from network
        
        self.children = {}  # Map action_idx -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.value_mean = 0.0 # Q(s, a)

    def is_expanded(self):
        return len(self.children) > 0

class MCTSAlphaZeroCPU:
    """
    Serial CPU implementation of AlphaZero MCTS.
    Supports both CPU and CUDA devices.
    """
    def __init__(self, model, network, c_puct=1.4, n_iters=100, gamma=0.99, device="cpu"):
        self.model = model
        self.network = network
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.gamma = gamma
        self.device = device

    def search(self, root_state):
        # Initialize root node
        root = Node(state=root_state, prior=1.0)
        
        # Expand root immediately
        self._expand_node(root)
        
        # Add Dirichlet noise
        if root.is_expanded():
            actions = list(root.children.keys())
            noise = np.random.dirichlet([0.3] * len(actions)) 
            for i, action_idx in enumerate(actions):
                root.children[action_idx].prior = 0.95 * root.children[action_idx].prior + 0.05 * noise[i]

        for _ in range(self.n_iters):
            node = root
            search_path = [node]

            while node.is_expanded():
                action_idx, node = self._select_child(node)
                search_path.append(node)

            value = self._expand_node(node)
            self._backpropagate(search_path, value)

        counts = np.zeros(self.model.action_space_size)
        for action_idx, child in root.children.items():
            counts[action_idx] = child.visit_count
            
        if np.sum(counts) == 0:
            return np.ones(self.model.action_space_size) / self.model.action_space_size, root.value_mean, root

        pi = counts / np.sum(counts)
        return pi, root.value_mean, root

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action_idx, child in node.children.items():
            u = self.c_puct * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
            q = child.value_mean
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child
        
        return best_action, best_child

    def _expand_node(self, node):
        # Handle ROE (always numpy array)
        roe_tensor = torch.tensor(node.state.roe, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Handle grid belief (might be numpy array or torch tensor)
        if isinstance(node.state.grid.belief, torch.Tensor):
            # Already a tensor - just ensure correct device and add batch dimension
            grid_tensor = node.state.grid.belief.to(self.device).unsqueeze(0)
        else:
            # Numpy array - convert to tensor on correct device
            grid_tensor = torch.tensor(node.state.grid.belief, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            policy_logits, value = self.network(roe_tensor, grid_tensor)

        # Move tensors to CPU before converting to numpy (required for CUDA tensors)
        policy_probs = torch.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
        value_scalar = value.cpu().item()

        # CRITICAL: Free GPU tensors immediately to prevent memory leak
        del roe_tensor, grid_tensor, policy_logits, value

        actions = self.model.get_all_actions()

        for i, action_vec in enumerate(actions):
            next_state, _ = self.model.step(node.state, action_vec)
            child = Node(state=next_state, parent=node, action_idx=i, prior=policy_probs[i])
            node.children[i] = child

        return value_scalar

    def _backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            node.value_mean = node.value_sum / node.visit_count
            value = node.reward + self.gamma * value if hasattr(node, 'reward') else value * self.gamma 

    def export_tree_to_dot(self, root, episode, step, output_path):
        """Exports the MCTS tree to a Graphviz DOT format with wide layout."""
        try:
            dot = Digraph(comment=f'MCTS Tree Ep{episode} Step{step}')
            
            dot.attr(rankdir='TB', ratio='auto', nodesep='0.5', ranksep='1.0')

            def add_node(node, node_id):
                color_hex = "#ffffff"
                if node.visit_count > 0:
                    val = np.clip(node.value_mean, -1, 1) 
                    r = int(255 * (1 - max(0, val)))
                    g = int(255 * (1 - max(0, -val)))
                    b = 200
                    color_hex = f"#{r:02x}{g:02x}{b:02x}"

                label = f"N={node.visit_count}\nQ={node.value_mean:.2f}\nP={node.prior:.2f}"
                dot.node(node_id, label, style='filled', fillcolor=color_hex, shape='ellipse', fontsize='10')

                sorted_children = sorted(node.children.items(), key=lambda x: x[1].visit_count, reverse=True)
                
                for action_idx, child in sorted_children:
                    if child.visit_count > 0:
                        child_id = f"{node_id}_{action_idx}"
                        
                        act_vec = self.model.get_all_actions()[action_idx]
                        mag = np.linalg.norm(act_vec)
                        if mag < 1e-6: act_str = "No-Op"
                        else: act_str = f"|dv|={mag:.2f}"
                        
                        dot.edge(node_id, child_id, label=act_str, fontsize='8')
                        add_node(child, child_id)

            add_node(root, "root")
            
            filename = f"tree_ep{episode}_step{step}"
            dot.render(filename=filename, directory=output_path, format='png', cleanup=True)
        except Exception as e:
            print(f"Graphviz export failed: {e}")
