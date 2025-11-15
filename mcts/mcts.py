import numpy as np


class Node:
    def __init__(self, state, actions, action_index=None, parent=None):
        self.state = state
        self.parent = parent

        # Action that led here (np.array) and its index in the parent's action list
        self.action_index = action_index
        self.action = None
        self.actions = list(actions)  # list of np arrays
        if action_index is not None:
            # parent's action list index; the actual vector is stored here just for convenience
            self.action = self.actions[action_index] if 0 <= action_index < len(self.actions) else None

        # Children and expansion tracking
        self.children = []  # list[Node]
        self.untried_action_indices = list(range(len(self.actions)))  # indices of actions not expanded yet

        # Statistics for UCB and value estimates
        num_actions = len(self.actions)
        self.N = 0                          # visits to this state
        self.Q_sa = np.zeros(num_actions)   # mean return per action index
        self.N_sa = np.zeros(num_actions, dtype=int)  # visits per action index

    def __repr__(self):
        return f"Node(N={self.N}, num_children={len(self.children)})"


class MCTS:
    def __init__(self, model, iters=1000, max_depth=5, c=1.4, gamma=1.0):
        self.max_iterations = iters
        self.max_depth = max_depth
        self.c = c
        self.gamma = gamma
        self.mdp = model

    def get_best_root_action(self, root_state):
        root_actions = self.mdp.actions(root_state)
        root = Node(root_state, actions=root_actions, action_index=None, parent=None)

        for _ in range(self.max_iterations):
            self._search(root, depth=0)

        # Best action at root = argmax Q_sa over actions
        if len(root.actions) == 0:
            # No available actions → return zero action and value 0
            return np.zeros(3), 0.0

        best_idx = int(np.argmax(root.Q_sa))
        best_action = root.actions[best_idx]
        best_value = float(root.Q_sa[best_idx])
        return best_action, best_value

    def _select_ucb_action_index(self, node):
        total_N = node.N
        # Avoid division by zero; but by design, this should not be called
        # unless all actions have N_sa[i] > 0.
        ucb_values = node.Q_sa + self.c * np.sqrt(
            np.log(total_N) / np.maximum(node.N_sa, 1)
        )
        return int(np.argmax(ucb_values))

    def _expand(self, node, action_index):
        action = node.actions[action_index]
        next_state, _ = self.mdp.step(node.state, action)
        next_actions = self.mdp.actions(next_state)

        child = Node(
            state=next_state,
            actions=next_actions,
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
                # Apply one step of discount going up the tree
                G = self.gamma * G

                a_idx = current.action_index
                parent = current.parent

                parent.N_sa[a_idx] += 1
                n_sa = parent.N_sa[a_idx]
                q_old = parent.Q_sa[a_idx]
                parent.Q_sa[a_idx] = q_old + (G - q_old) / n_sa

            current = current.parent

    def _search(self, node, depth):
        # 1) Global depth limit: treat as horizon state (value 0)
        if depth == self.max_depth:
            value = 0.0
            self._backpropagate(node, value)
            return value

        # 2) Expansion: if we still have untried actions at this node, expand one
        if node.untried_action_indices:
            a_idx = node.untried_action_indices.pop()
            child = self._expand(node, a_idx)

            # We moved one step deeper in the tree → rollout starts at depth+1
            value = self._rollout(child.state, depth + 1)
            self._backpropagate(child, value)
            return value

        # 3) Node is fully expanded and has children → select via UCB and recurse
        if node.children:
            a_idx = self._select_ucb_action_index(node)

            # Find child corresponding to this action index
            # (there should be exactly one)
            child = None
            for ch in node.children:
                if ch.action_index == a_idx:
                    child = ch
                    break

            if child is None:
                # Fallback: shouldn't happen, but avoid crashing
                value = self._rollout(node.state, depth)
                self._backpropagate(node, value)
                return value

            return self._search(child, depth + 1)

        # 4) No actions available (terminal state) → rollout returns 0
        value = 0.0
        self._backpropagate(node, value)
        return value
