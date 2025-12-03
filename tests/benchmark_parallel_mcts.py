"""
Benchmark parallel independent MCTS vs sequential MCTS.
"""

import sys
import numpy as np
import time
from multiprocessing import cpu_count

sys.path.insert(0, '/home/saveasmtz/Documents/CS229_Final_Project')

from camera.camera_observations import VoxelGrid, GroundTruthRSO
from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from mcts.mcts_gpu import MCTSPU
from mcts.mcts_controller_gpu import MCTSControllerGPU, MCTSControllerParallelIndependent


def create_test_environment():
    """Create a simple test environment."""
    # Orbital parameters
    mu_earth = 3.986e14  # m^3/s^2
    a_chief = 6.8e6  # m
    e_chief = 0.0
    i_chief = 0.0
    omega_chief = 0.0
    n_chief = np.sqrt(mu_earth / a_chief**3)

    time_step = 60.0  # seconds
    horizon = 3  # steps
    mcts_iters = 100  # iterations per MCTS

    # Create grid and RSO
    grid = VoxelGrid(grid_dims=(20, 20, 20))
    rso = GroundTruthRSO(grid)

    # Camera configuration
    camera_fn = {
        'fov_degrees': 60,
        'sensor_res': (64, 64),
        'noise_params': {
            'p_hit_given_occupied': 0.95,
            'p_hit_given_empty': 0.001,
        }
    }

    # Initial state
    state = np.zeros(6)  # ROE state
    time_val = 0.0

    return {
        'mu_earth': mu_earth,
        'a_chief': a_chief,
        'e_chief': e_chief,
        'i_chief': i_chief,
        'omega_chief': omega_chief,
        'n_chief': n_chief,
        'time_step': time_step,
        'horizon': horizon,
        'mcts_iters': mcts_iters,
        'grid': grid,
        'rso': rso,
        'camera_fn': camera_fn,
        'state': state,
        'time': time_val,
    }


def benchmark_sequential_mcts(env):
    """Benchmark sequential MCTS."""
    print(f"\n{'='*70}")
    print(f"Sequential MCTS (GPU)")
    print(f"{'='*70}")

    controller = MCTSControllerGPU(
        mu_earth=env['mu_earth'],
        a_chief=env['a_chief'],
        e_chief=env['e_chief'],
        i_chief=env['i_chief'],
        omega_chief=env['omega_chief'],
        n_chief=env['n_chief'],
        time_step=env['time_step'],
        horizon=env['horizon'],
        alpha_dv=0.1,
        beta_tan=0.2,
        rollout_policy='random',
        lambda_dv=0.1,
        mcts_iters=env['mcts_iters'],
        mcts_c=1.4,
        gamma=0.99,
        gpu_batch_size=16,
        device="cpu",  # Use CPU for fair comparison
        verbose=True
    )

    times = []
    for trial in range(3):
        t0 = time.time()
        action, value, stats = controller.select_action(
            env['state'], env['time'], None, env['grid'], env['rso'],
            env['camera_fn'], step=0, verbose=False
        )
        t1 = time.time()
        times.append(t1 - t0)
        print(f"  Trial {trial+1}: {(t1-t0)*1000:.1f} ms, action={action}, value={value:.4f}")

    avg_time = np.mean(times)
    print(f"  Average: {avg_time*1000:.1f} ms (σ={np.std(times)*1000:.1f} ms)")
    return avg_time


def benchmark_parallel_independent_mcts(env, num_trees=None):
    """Benchmark parallel independent MCTS."""
    num_trees = num_trees or max(2, cpu_count() - 1)
    print(f"\n{'='*70}")
    print(f"Parallel Independent MCTS ({num_trees} trees)")
    print(f"{'='*70}")

    controller = MCTSControllerParallelIndependent(
        mu_earth=env['mu_earth'],
        a_chief=env['a_chief'],
        e_chief=env['e_chief'],
        i_chief=env['i_chief'],
        omega_chief=env['omega_chief'],
        n_chief=env['n_chief'],
        time_step=env['time_step'],
        horizon=env['horizon'],
        alpha_dv=0.1,
        beta_tan=0.2,
        rollout_policy='random',
        lambda_dv=0.1,
        mcts_iters=env['mcts_iters'],
        mcts_c=1.4,
        gamma=0.99,
        gpu_batch_size=16,
        device="cpu",
        num_trees=num_trees,
        verbose=True
    )

    times = []
    for trial in range(3):
        t0 = time.time()
        action, value, stats = controller.select_action(
            env['state'], env['time'], None, env['grid'], env['rso'],
            env['camera_fn'], step=0, verbose=False
        )
        t1 = time.time()
        times.append(t1 - t0)
        print(f"  Trial {trial+1}: {(t1-t0)*1000:.1f} ms, action={action}, value={value:.4f}, "
              f"mean={stats['mean_value']:.4f}")

    avg_time = np.mean(times)
    print(f"  Average: {avg_time*1000:.1f} ms (σ={np.std(times)*1000:.1f} ms)")
    return avg_time


if __name__ == "__main__":
    print("Parallel Independent MCTS Benchmark")
    print("=" * 70)

    env = create_test_environment()

    # Sequential baseline
    seq_time = benchmark_sequential_mcts(env)

    # Parallel with default number of trees
    num_trees = max(2, cpu_count() - 1)
    par_time = benchmark_parallel_independent_mcts(env, num_trees=num_trees)

    # Results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Sequential:  {seq_time*1000:.1f} ms")
    print(f"Parallel:    {par_time*1000:.1f} ms ({num_trees} trees)")
    print(f"Speedup:     {seq_time/par_time:.2f}x")
    print(f"{'='*70}")
