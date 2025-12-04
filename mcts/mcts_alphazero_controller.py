import numpy as np
import torch
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
    """
    def __init__(self, model, network, c_puct=1.4, n_iters=100, gamma=0.99):
        self.model = model
        self.network = network
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.gamma = gamma
        self.device = "cpu"

    def search(self, root_state):
        # Initialize root node
        root = Node(state=root_state, prior=1.0)
        
        # Expand root immediately
        self._expand_node(root)
        
        # Add Dirichlet noise
        if root.is_expanded():
            actions = list(root.children.keys())
            # Change 0.3 to something higher if you want more chaotic exploration
            noise = np.random.dirichlet([0.3] * len(actions)) 
            for i, action_idx in enumerate(actions):
                # Increase noise weight from 0.25 to 0.5 for early training
                root.children[action_idx].prior = 0.75 * root.children[action_idx].prior + 0.25 * noise[i]

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
        roe_tensor = torch.tensor(node.state.roe, dtype=torch.float32).unsqueeze(0)
        grid_tensor = torch.tensor(node.state.grid.belief, dtype=torch.float32).unsqueeze(0)
        
        self.network.eval()
        with torch.no_grad():
            policy_logits, value = self.network(roe_tensor, grid_tensor)
        
        policy_probs = torch.softmax(policy_logits, dim=1).squeeze().numpy()
        value_scalar = value.item()

        actions = self.model.get_all_actions() 
        
        for i, action_vec in enumerate(actions):
            # Note: We store state in child for simplicity in this implementation
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
            
            # --- FIXED LAYOUT PARAMS ---
            # Remove strict size limits to allow tree to grow
            # Use 'nodesep' to spread out siblings horizontally
            dot.attr(rankdir='TB', ratio='auto', nodesep='0.5', ranksep='1.0')

            def add_node(node, node_id):
                # Color node based on value
                color_hex = "#ffffff"
                if node.visit_count > 0:
                    val = np.clip(node.value_mean, -1, 1) 
                    # Map -1..1 to Red..Green
                    r = int(255 * (1 - max(0, val)))
                    g = int(255 * (1 - max(0, -val)))
                    b = 200
                    color_hex = f"#{r:02x}{g:02x}{b:02x}"

                # Simplified label
                label = f"N={node.visit_count}\nQ={node.value_mean:.2f}\nP={node.prior:.2f}"
                dot.node(node_id, label, style='filled', fillcolor=color_hex, shape='ellipse', fontsize='10')

                # --- FIXED: Sort but DO NOT prune to top 5 ---
                sorted_children = sorted(node.children.items(), key=lambda x: x[1].visit_count, reverse=True)
                
                for action_idx, child in sorted_children:
                    # Only visualize nodes with visits to keep tree readable
                    if child.visit_count > 0:
                        child_id = f"{node_id}_{action_idx}"
                        
                        # Action string as edge label
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