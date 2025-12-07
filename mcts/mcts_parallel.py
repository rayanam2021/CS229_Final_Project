"""
Parallel MCTS using Root Parallelization.

Instead of running 3000 sequential simulations, we run multiple independent
MCTS searches in parallel (e.g., 8 workers × 375 sims each = 3000 total),
then aggregate their statistics.

Benefits:
- 5-8x speedup with multiprocessing
- Better exploration (different random paths)
- No shared state complexity (each worker has own tree)
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from mcts.mcts import MCTS, Node
import multiprocessing
import time

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be done before any CUDA operations
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

def _run_mcts_worker(worker_id, root_state_data, mdp_params, iters_per_worker, max_depth, c, gamma):
    """
    Worker function to run MCTS in a separate process.

    Returns aggregated statistics from this worker's tree.
    """
    # Recreate MDP in worker process
    from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState

    mdp = OrbitalMCTSModel(
        a_chief=mdp_params['a_chief'],
        e_chief=mdp_params['e_chief'],
        i_chief=mdp_params['i_chief'],
        omega_chief=mdp_params['omega_chief'],
        n_chief=mdp_params['n_chief'],
        rso=mdp_params['rso'],
        camera_fn=mdp_params['camera_fn'],
        grid_dims=mdp_params['grid_dims'],
        lambda_dv=mdp_params['lambda_dv'],
        time_step=mdp_params['time_step'],
        max_depth=max_depth,
        use_torch=mdp_params.get('use_torch', False),
        device=mdp_params.get('device', 'cpu')
    )

    # Recreate root state
    from camera.camera_observations import VoxelGrid
    use_torch = mdp_params.get('use_torch', False)
    device = mdp_params.get('device', 'cpu')

    grid = VoxelGrid(
        grid_dims=mdp_params['grid_dims'],
        use_torch=use_torch,
        device=device
    )

    # Restore grid state
    if use_torch:
        import torch
        grid.belief = torch.tensor(root_state_data['grid_belief'], dtype=torch.float32, device=device)
        grid.log_odds = torch.tensor(root_state_data['grid_log_odds'], dtype=torch.float32, device=device)
    else:
        grid.belief = root_state_data['grid_belief']
        grid.log_odds = root_state_data['grid_log_odds']

    root_state = OrbitalState(
        roe=root_state_data['roe'],
        grid=grid,
        time=root_state_data['time']
    )

    # Run MCTS
    mcts = MCTS(model=mdp, iters=iters_per_worker, max_depth=max_depth, c=c, gamma=gamma)

    root_actions = mdp.actions(root_state)
    root = Node(root_state, actions=root_actions, action_index=None, parent=None)

    # Run simulations
    for _ in range(iters_per_worker):
        mcts._search(root, depth=0)

    # Return aggregated statistics
    return {
        'worker_id': worker_id,
        'Q_sa': root.Q_sa.copy(),
        'N_sa': root.N_sa.copy(),
        'N': root.N,
        'num_actions': len(root_actions)
    }


class ParallelMCTS:
    """
    Root-parallelized MCTS controller.

    Runs multiple independent MCTS searches in parallel, then aggregates results.
    """

    def __init__(self, model, iters=3000, max_depth=5, c=1.4, gamma=1.0, num_workers=None):
        """
        Args:
            model: OrbitalMCTSModel instance
            iters: Total number of MCTS simulations
            max_depth: Maximum tree depth
            c: UCB exploration constant
            gamma: Discount factor
            num_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.mdp = model
        self.total_iters = iters
        self.max_depth = max_depth
        self.c = c
        self.gamma = gamma

        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        self.num_workers = min(num_workers, iters)  # Don't use more workers than iterations

        # Distribute iterations across workers
        self.iters_per_worker = iters // self.num_workers
        self.remainder = iters % self.num_workers

        print(f"[ParallelMCTS] {self.num_workers} workers, {self.iters_per_worker} iters/worker (+{self.remainder} remainder)")

    def get_best_root_action(self, root_state, step, out_folder, return_stats=True):
        """
        Run parallel MCTS and aggregate results.

        Args:
            root_state: Initial OrbitalState
            step: Current step number
            out_folder: Output folder for visualization
            return_stats: Whether to return statistics

        Returns:
            best_action: Best action according to aggregated statistics
            best_value: Estimated value of best action
            stats: Statistics dictionary (if return_stats=True)
        """
        # Prepare root state data for workers
        if self.mdp.use_torch:
            grid_belief = root_state.grid.belief.cpu().numpy()
            grid_log_odds = root_state.grid.log_odds.cpu().numpy()
        else:
            grid_belief = root_state.grid.belief.copy()
            grid_log_odds = root_state.grid.log_odds.copy()

        root_state_data = {
            'roe': root_state.roe.copy(),
            'grid_belief': grid_belief,
            'grid_log_odds': grid_log_odds,
            'time': root_state.time
        }

        # Prepare MDP parameters
        mdp_params = {
            'a_chief': self.mdp.a_chief,
            'e_chief': self.mdp.e_chief,
            'i_chief': self.mdp.i_chief,
            'omega_chief': self.mdp.omega_chief,
            'n_chief': self.mdp.n_chief,
            'rso': self.mdp.rso,
            'camera_fn': self.mdp.camera_fn,
            'grid_dims': self.mdp.grid_dims,
            'lambda_dv': self.mdp.lambda_dv,
            'time_step': self.mdp.time_step,
            'use_torch': self.mdp.use_torch,
            'device': self.mdp.device
        }

        # Launch parallel workers
        start_time = time.time()
        worker_results = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(self.num_workers):
                # Give remainder iterations to first worker
                iters = self.iters_per_worker + (self.remainder if i == 0 else 0)

                futures.append(executor.submit(
                    _run_mcts_worker,
                    worker_id=i,
                    root_state_data=root_state_data,
                    mdp_params=mdp_params,
                    iters_per_worker=iters,
                    max_depth=self.max_depth,
                    c=self.c,
                    gamma=self.gamma
                ))

            # Collect results
            for future in futures:
                worker_results.append(future.result())

        parallel_time = time.time() - start_time

        # Aggregate statistics across all workers
        num_actions = worker_results[0]['num_actions']

        aggregated_Q_sa = np.zeros(num_actions)
        aggregated_N_sa = np.zeros(num_actions, dtype=int)
        total_N = 0

        for result in worker_results:
            # Weighted average of Q values by visit counts
            aggregated_Q_sa += result['Q_sa'] * result['N_sa']
            aggregated_N_sa += result['N_sa']
            total_N += result['N']

        # Normalize Q values
        mask = aggregated_N_sa > 0
        aggregated_Q_sa[mask] /= aggregated_N_sa[mask]

        # Select best action
        best_idx = int(np.argmax(aggregated_Q_sa))
        best_action = self.mdp.actions(root_state)[best_idx]
        best_value = float(aggregated_Q_sa[best_idx])

        if return_stats:
            stats = {
                "root_N": int(total_N),
                "root_Q_sa": aggregated_Q_sa.copy(),
                "root_N_sa": aggregated_N_sa.copy(),
                "best_idx": best_idx,
                "best_action": best_action,
                "predicted_value": best_value,
                "parallel_time": parallel_time,
                "num_workers": self.num_workers
            }
            return best_action, best_value, stats

        return best_action, best_value
