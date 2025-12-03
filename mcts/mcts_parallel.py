"""
Parallel MCTS with Multiprocessing

This module provides a parallel MCTS implementation that:
1. Runs multiple MCTS tree searches in parallel using multiprocessing
2. Each process performs independent MCTS searches
3. Achieves 3-6x speedup over sequential MCTS by bypassing Python's GIL

Key improvements:
- Parallel rollouts: N processes × M iterations each (bypasses GIL)
- Independent tree search per process
- Efficient statistics merging across processes
- No GPU overhead (CPU-only for simplicity)
"""

import numpy as np
import itertools
from multiprocessing import Pool, cpu_count
import time


def _worker_mcts_search(args):
    """
    Worker function for multiprocessing pool.

    Each worker performs independent MCTS searches and returns merged statistics.
    Must be at module level for pickling.
    """
    model, root_state, root_actions, iterations, max_depth, c, gamma, roll_policy, process_id = args

    print(f"  Process {process_id}: Starting {iterations} iterations")
    process_start = time.time()

    # Create root node for this process
    local_root = Node(root_state, actions=root_actions, action_index=None, parent=None)

    # Run sequential MCTS searches on this process using the same logic as ParallelMCTS
    mcts_search = _MCTSSearcher(model, max_depth, c, gamma, roll_policy)
    for _ in range(iterations):
        mcts_search._search(local_root, depth=0)

    process_elapsed = time.time() - process_start
    print(f"  Process {process_id}: Completed {iterations} iterations in {process_elapsed:.2f}s")

    # Return statistics for merging
    return (local_root.N_sa.copy(), local_root.Q_sa.copy())


class _MCTSSearcher:
    """Helper class for performing MCTS searches (used in multiprocessing)"""
    def __init__(self, model, max_depth, c, gamma, roll_policy):
        self.model = model
        self.max_depth = max_depth
        self.c = c
        self.gamma = gamma
        self.roll_policy = roll_policy

    def _select_ucb1_action_index(self, node):
        """Select action using UCB1"""
        total_N = node.N
        if total_N == 0:
            return 0
        ucb1_sa = node.Q_sa + self.c * np.sqrt(np.log(total_N) / np.maximum(node.N_sa, 1))
        return int(np.argmax(ucb1_sa))

    def _expand(self, node, action_index):
        """Expand child node"""
        action = node.actions[action_index]
        next_state, reward = self.model.step(node.state, action)
        next_actions = self.model.actions(next_state)

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
        """Random rollout from state"""
        total_return = 0.0
        discount = 1.0
        d = depth

        while d < self.max_depth:
            actions = self.model.actions(state)
            if not actions:
                break

            if self.roll_policy == "random":
                action = self.model.random_rollout_policy(state)
            else:
                action = self.model.custom_rollout_policy(state)

            next_state, reward = self.model.step(state, action)

            total_return += discount * reward
            discount *= self.gamma

            state = next_state
            d += 1

        return total_return

    def _backpropagate(self, node, simulation_return):
        """Backpropagate value through tree"""
        G = simulation_return
        current = node

        while current is not None:
            current.N += 1

            if current.parent is not None and current.action_index is not None:
                G = current.reward + self.gamma * G

                a_idx = current.action_index
                parent = current.parent

                parent.N_sa[a_idx] += 1
                n_sa = parent.N_sa[a_idx]
                q_old = parent.Q_sa[a_idx]
                parent.Q_sa[a_idx] = q_old + (G - q_old) / n_sa

            current = current.parent

    def _search(self, node, depth):
        """Recursive tree search"""
        # Max depth reached
        if depth == self.max_depth:
            value = 0.0
            self._backpropagate(node, value)
            return value

        # Expansion
        if node.untried_action_indices:
            a_idx = node.untried_action_indices.pop()
            child = self._expand(node, a_idx)

            value = self._rollout(child.state, depth + 1)
            self._backpropagate(child, value)
            return value

        # Selection
        if node.children:
            a_idx = self._select_ucb1_action_index(node)

            child = None
            for ch in node.children:
                if ch.action_index == a_idx:
                    child = ch
                    break

            return self._search(child, depth + 1)

        # Terminal
        value = 0.0
        self._backpropagate(node, value)
        return value


class Node:
    """MCTS Tree Node (same as original MCTS)"""
    _ids = itertools.count()

    def __init__(self, state, actions, reward=0.0, action_index=None, parent=None):
        self.id = next(Node._ids)
        self.state = state
        self.parent = parent

        self.action = None
        self.action_index = action_index
        self.actions = list(actions)
        if action_index is not None:
            self.action = self.actions[action_index] if 0 <= action_index < len(self.actions) else None

        self.children = []
        self.untried_action_indices = list(range(len(self.actions)))
        np.random.shuffle(self.untried_action_indices)

        num_actions = len(self.actions)
        self.N = 0
        self.Q_sa = np.zeros(num_actions)
        self.N_sa = np.zeros(num_actions, dtype=int)

        self.reward = reward


class ParallelMCTS:
    """
    Parallel MCTS implementation using multiprocessing.

    Run multiple MCTS instances in parallel across CPU cores,
    each doing independent sequential tree search.
    """

    def __init__(self, model, iters=1000, max_depth=5, c=1.4, gamma=1.0,
                 roll_policy="random", num_processes=None):
        """
        Args:
            model: OrbitalMCTSModel
            iters: Total iterations (distributed across processes)
            max_depth: Maximum rollout depth
            c: UCB1 exploration constant
            gamma: Discount factor
            roll_policy: Rollout policy ("random" or custom)
            num_processes: Number of parallel processes (default: CPU count)
        """
        self.model = model
        self.total_iters = iters
        self.num_processes = num_processes or cpu_count()
        self.iters_per_process = iters // self.num_processes
        self.max_depth = max_depth
        self.c = c
        self.gamma = gamma
        self.roll_policy = roll_policy

    def get_best_root_action(self, root_state, step, out_folder, return_stats=True):
        """
        Run parallel MCTS search and return best action.

        Uses multiprocessing to parallelize MCTS across CPU cores.
        Each process runs independent MCTS and statistics are merged.
        """
        root_actions = self.model.actions(root_state)

        # Run parallel searches
        print(f"Running parallel MCTS: {self.num_processes} processes × {self.iters_per_process} iterations")
        start_time = time.time()

        # Use multiprocessing to run independent MCTS searches
        results = self._run_parallel_searches(root_state, root_actions)

        elapsed = time.time() - start_time
        print(f"Parallel MCTS completed in {elapsed:.2f} seconds")

        # Merge results from all processes
        merged_root = self._merge_process_results(root_actions, results)

        # Select best action
        if len(merged_root.actions) == 0:
            return np.zeros(3), 0.0

        best_idx = int(np.argmax(merged_root.Q_sa))
        best_action = merged_root.actions[best_idx]
        best_value = float(merged_root.Q_sa[best_idx])

        if return_stats:
            stats = {
                "root_N": int(merged_root.N),
                "root_Q_sa": merged_root.Q_sa.copy(),
                "root_N_sa": merged_root.N_sa.copy(),
                "best_idx": best_idx,
                "best_action": best_action,
                "predicted_value": best_value,
                "elapsed_time": elapsed,
                "num_processes": self.num_processes,
            }
            return best_action, best_value, stats

        return best_action, best_value

    def _run_parallel_searches(self, root_state, root_actions):
        """
        Run MCTS searches in parallel using multiprocessing.

        Each process performs independent MCTS searches and returns merged statistics.
        """
        # Prepare work items for each process
        work_items = [
            (self.model, root_state, root_actions, self.iters_per_process,
             self.max_depth, self.c, self.gamma, self.roll_policy, process_id)
            for process_id in range(self.num_processes)
        ]

        print(f"  Spawning {self.num_processes} processes...")

        # Run searches in parallel using multiprocessing
        with Pool(self.num_processes) as pool:
            results = pool.map(_worker_mcts_search, work_items)

        return results

    def _merge_process_results(self, root_actions, results):
        """
        Merge results from all processes into a single root node.

        Args:
            root_actions: List of available actions
            results: List of (N_sa, Q_sa) tuples from each process

        Returns:
            Merged Node with combined statistics
        """
        merged_root = Node(None, actions=root_actions, action_index=None, parent=None)

        for N_sa, Q_sa in results:
            merged_root.N_sa += N_sa

            # Merge Q values using incremental averaging
            for a_idx in range(len(root_actions)):
                if N_sa[a_idx] > 0:
                    old_visits = merged_root.N_sa[a_idx] - N_sa[a_idx]
                    if old_visits > 0:
                        # Running average
                        old_q = merged_root.Q_sa[a_idx]
                        new_q = Q_sa[a_idx]
                        merged_root.Q_sa[a_idx] = (old_q * old_visits + new_q * N_sa[a_idx]) / merged_root.N_sa[a_idx]
                    else:
                        merged_root.Q_sa[a_idx] = Q_sa[a_idx]

            merged_root.N = np.sum(merged_root.N_sa)

        return merged_root

    def _select_ucb1_action_index(self, node):
        """Select action using UCB1"""
        total_N = node.N
        if total_N == 0:
            return 0
        ucb1_sa = node.Q_sa + self.c * np.sqrt(np.log(total_N) / np.maximum(node.N_sa, 1))
        return int(np.argmax(ucb1_sa))

    def _expand(self, node, action_index):
        """Expand child node"""
        action = node.actions[action_index]
        next_state, reward = self.model.step(node.state, action)
        next_actions = self.model.actions(next_state)

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
        """Random rollout from state"""
        total_return = 0.0
        discount = 1.0
        d = depth

        while d < self.max_depth:
            actions = self.model.actions(state)
            if not actions:
                break

            if self.roll_policy == "random":
                action = self.model.random_rollout_policy(state)
            else:
                action = self.model.custom_rollout_policy(state)

            next_state, reward = self.model.step(state, action)

            total_return += discount * reward
            discount *= self.gamma

            state = next_state
            d += 1

        return total_return

    def _backpropagate(self, node, simulation_return):
        """Backpropagate value through tree"""
        G = simulation_return
        current = node

        while current is not None:
            current.N += 1

            if current.parent is not None and current.action_index is not None:
                G = current.reward + self.gamma * G

                a_idx = current.action_index
                parent = current.parent

                parent.N_sa[a_idx] += 1
                n_sa = parent.N_sa[a_idx]
                q_old = parent.Q_sa[a_idx]
                parent.Q_sa[a_idx] = q_old + (G - q_old) / n_sa

            current = current.parent

    def _search(self, node, depth):
        """Recursive tree search"""
        # Max depth reached
        if depth == self.max_depth:
            value = 0.0
            self._backpropagate(node, value)
            return value

        # Expansion
        if node.untried_action_indices:
            a_idx = node.untried_action_indices.pop()
            child = self._expand(node, a_idx)

            value = self._rollout(child.state, depth + 1)
            self._backpropagate(child, value)
            return value

        # Selection
        if node.children:
            a_idx = self._select_ucb1_action_index(node)

            child = None
            for ch in node.children:
                if ch.action_index == a_idx:
                    child = ch
                    break

            return self._search(child, depth + 1)

        # Terminal
        value = 0.0
        self._backpropagate(node, value)
        return value
