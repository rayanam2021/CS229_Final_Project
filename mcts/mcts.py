import numpy as np
import itertools
import os

from graphviz import Digraph

class Node:
    _ids = itertools.count()
    def __init__(self, state, actions, reward=0.0, action_index=None, parent=None):
        self.id = next(Node._ids)
        self.state = state
        self.parent = parent

        self.action = None                  # Action that led here (np.array)
        self.action_index = action_index    # index of action that led here in the parent's action list
        self.actions = list(actions)
        if action_index is not None:
            # parent's action list index; the actual vector is stored here just for convenience
            self.action = self.actions[action_index] if 0 <= action_index < len(self.actions) else None

        # Children and expansion tracking
        self.children = []
        self.untried_action_indices = list(range(len(self.actions)))  # indices of actions not expanded yet

        # Statistics for UCB1 and value estimates
        num_actions = len(self.actions)
        self.N = 0                          # visits to this state
        self.Q_sa = np.zeros(num_actions)   # mean return per action index
        self.N_sa = np.zeros(num_actions, dtype=int)  # visits per action index

        self.reward = reward  # reward for getting to this node

class MCTS:
    def __init__(self, model, iters=1000, max_depth=5, c=1.4, gamma=1.0):
        self.max_path_searches = iters
        self.max_depth = max_depth
        self.c = c
        self.gamma = gamma
        self.mdp = model

    def get_best_root_action(self, root_state, step, out_folder):
        root_actions = self.mdp.actions(root_state)
        root = Node(root_state, actions=root_actions, action_index=None, parent=None)

        for _ in range(self.max_path_searches):
            self._search(root, depth=0)

        # Best action at root = argmax Q_sa over actions
        if len(root.actions) == 0:
            # No available actions
            return np.zeros(3), 0.0

        best_idx = int(np.argmax(root.Q_sa))
        best_action = root.actions[best_idx]
        best_value = float(root.Q_sa[best_idx])

        self.export_tree_to_dot(root, step, out_folder)

        return best_action, best_value

    def _select_ucb1_action_index(self, node):
        total_N = node.N
        # should not be called unless all actions have N_sa[i] > 0.
        ucb1_sa = node.Q_sa + self.c * np.sqrt(np.log(total_N) / np.maximum(node.N_sa, 1))
        return int(np.argmax(ucb1_sa))

    def _expand(self, node, action_index):
        action = node.actions[action_index]
        next_state, reward = self.mdp.step(node.state, action)
        next_actions = self.mdp.actions(next_state)

        child = Node(
            state=next_state,
            actions=next_actions,
            reward=reward,
            action_index=action_index,
            parent=node,
        )

        node.children.append(child)

        return child

    def _rollout(self, state, depth):
        total_return = 0.0
        discount = 1.0
        d = depth

        while d < self.max_depth:
            actions = self.mdp.actions(state)
            if not actions:
                break  # terminal

            action = self.mdp.rollout_policy(state)
            next_state, reward = self.mdp.step(state, action)

            total_return += discount * reward
            discount *= self.gamma

            state = next_state
            d += 1

        return total_return

    def _backpropagate(self, node, simulation_return):
        G = simulation_return
        current = node

        while current is not None:
            current.N += 1

            if current.parent is not None and current.action_index is not None:
                # Bellman: r + gamma * G_child
                G = current.reward + self.gamma * G

                a_idx = current.action_index
                parent = current.parent

                parent.N_sa[a_idx] += 1
                n_sa = parent.N_sa[a_idx]
                q_old = parent.Q_sa[a_idx]
                parent.Q_sa[a_idx] = q_old + (G - q_old) / n_sa

            current = current.parent

    def _search(self, node, depth):
        # Max depth reached. Backpropagate and return 0 value
        if depth == self.max_depth:
            value = 0.0
            self._backpropagate(node, value)
            return value

        # Expansion: if we still have untried actions at this node, expand one (inf bonus in ucb1)
        if node.untried_action_indices:
            a_idx = node.untried_action_indices.pop()
            child = self._expand(node, a_idx)

            # We moved one step deeper in the tree. Rollout from depth+1 (child)
            value = self._rollout(child.state, depth + 1)
            self._backpropagate(child, value)
            return value

        # Node is fully expanded (no actiones left to try) and has children, select
        # best action via UCB1 to keep building the tree
        if node.children:
            a_idx = self._select_ucb1_action_index(node)

            # Find child corresponding to this action index
            # (there should be exactly one)
            child = None
            for ch in node.children:
                if ch.action_index == a_idx:
                    child = ch
                    break

            return self._search(child, depth + 1)

        # No children and no untried actions: no actions here (terminal), return 0 as terminal value.
        value = 0.0
        self._backpropagate(node, value)
        return value

    def export_tree_to_dot(self, root, step, output_path):
        dot = Digraph()

        def add(node):
            label = f"ID={node.id}\nN={node.N}"
            dot.node(str(node.id), label)

            for child in node.children:
                qs = ", ".join([f"{q:.2f}" for q in node.Q_sa])
                edge_label = (
                    f"a={child.action_index}\n"
                    f"r={child.reward:.2f}\n"   # immediate rewards returned by model at that step
                    f"Q={node.Q_sa[child.action_index]:.2f}"
                )
                dot.edge(str(node.id), str(child.id), edge_label)
                add(child)

        add(root)
        
        out_dir = os.path.join(output_path,"trees")
        os.makedirs(out_dir, exist_ok=True)
        filename = "step"+str(step)

        dot.render(filename=filename, directory=out_dir,format='png', cleanup=True)