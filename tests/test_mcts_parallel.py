"""
Test Parallel MCTS vs Sequential MCTS

This script compares:
1. Sequential MCTS (original implementation)
2. Parallel MCTS with 4 threads
3. Parallel MCTS with 8 threads

Expected results: 3-4x speedup with 4 threads, 4-6x with 8 threads
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from mcts.mcts import MCTS
from mcts.mcts_parallel import ParallelMCTS
from camera.camera_observations import VoxelGrid, GroundTruthRSO

def setup_test():
    """Create test environment"""
    # Orbital parameters (from scenario)
    a_chief = 6778e3  # meters
    e_chief = 0.002
    i_chief = 51.6
    omega_chief = 0
    n_chief = 14.0 / 1440  # mean motion in rev/min converted to rev/sec

    # Camera and grid
    camera_fn = {
        'fov_degrees': 30.0,
        'sensor_res': (32, 32),  # Smaller for faster test
        'noise_params': {
            'p_hit_given_occupied': 0.95,
            'p_hit_given_empty': 0.001,
        }
    }
    grid = VoxelGrid(grid_dims=(20, 20, 20))
    rso = GroundTruthRSO(grid)

    # Initial state
    roe_init = np.array([0, 0, 0, 0, 0, 0])
    initial_state = OrbitalState(roe=roe_init, grid=grid, time=0.0)

    # MDP model
    mdp = OrbitalMCTSModel(
        a_chief=a_chief,
        e_chief=e_chief,
        i_chief=i_chief,
        omega_chief=omega_chief,
        n_chief=n_chief,
        rso=rso,
        camera_fn=camera_fn,
        grid_dims=(20, 20, 20),
        lambda_dv=0.1,
        time_step=60.0,
        max_depth=3,  # Shallow for fast test
        grid=grid
    )

    return initial_state, mdp, camera_fn


def test_sequential_mcts(state, mdp, iters=100):
    """Test sequential MCTS"""
    print("\n" + "="*70)
    print("SEQUENTIAL MCTS (1 thread)")
    print("="*70)

    mcts = MCTS(model=mdp, iters=iters, max_depth=3, c=1.4, gamma=1.0)

    start_time = time.time()
    action, value, stats = mcts.get_best_root_action(state, 0, ".", return_stats=True)
    elapsed = time.time() - start_time

    print(f"Iterations: {iters}")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Best action: {action}")
    print(f"Predicted value: {value:.4f}")
    print(f"Root visits: {stats['root_N']}")

    return elapsed, action, value


def test_parallel_mcts(state, mdp, total_iters=100, num_processes=4):
    """Test parallel MCTS"""
    print("\n" + "="*70)
    print(f"PARALLEL MCTS ({num_processes} processes)")
    print("="*70)

    mcts = ParallelMCTS(
        model=mdp,
        iters=total_iters,
        max_depth=3,
        c=1.4,
        gamma=1.0,
        num_processes=num_processes
    )

    start_time = time.time()
    action, value, stats = mcts.get_best_root_action(state, 0, ".", return_stats=True)
    elapsed = time.time() - start_time

    print(f"Total iterations: {total_iters} ({total_iters//num_processes} per process)")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Best action: {action}")
    print(f"Predicted value: {value:.4f}")
    print(f"Root visits: {stats['root_N']}")

    return elapsed, action, value


def run_benchmark():
    """Run full benchmark"""
    import multiprocessing

    print("\n" + "="*80)
    print("PARALLEL MCTS BENCHMARK")
    print("="*80)

    state, mdp, camera_fn = setup_test()

    # Smaller test for development
    test_iters = 100
    cpu_count = multiprocessing.cpu_count()

    print(f"Test configuration:")
    print(f"  Iterations: {test_iters}")
    print(f"  Camera resolution: 32×32 (1024 rays per observation)")
    print(f"  Max depth: 3")
    print(f"  Grid: 20×20×20 voxels")
    print(f"  CPU cores available: {cpu_count}")

    # Sequential baseline
    seq_time, seq_action, seq_value = test_sequential_mcts(state, mdp, iters=test_iters)

    # Parallel 4 processes
    par4_time, par4_action, par4_value = test_parallel_mcts(
        state, mdp, total_iters=test_iters, num_processes=4
    )

    # Parallel with all available cores
    par_max_time, par_max_action, par_max_value = test_parallel_mcts(
        state, mdp, total_iters=test_iters, num_processes=cpu_count
    )

    # Results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\nTiming Results:")
    print(f"  Sequential (1 process):    {seq_time:.2f}s (baseline)")
    print(f"  Parallel (4 processes):    {par4_time:.2f}s (speedup: {seq_time/par4_time:.2f}x)")
    print(f"  Parallel ({cpu_count} processes): {par_max_time:.2f}s (speedup: {seq_time/par_max_time:.2f}x)")

    print(f"\nAction Consistency:")
    print(f"  Sequential action:  {seq_action}")
    print(f"  4-process action:   {par4_action}")
    print(f"  {cpu_count}-process action:  {par_max_action}")

    print(f"\nValue Consistency:")
    print(f"  Sequential:  {seq_value:.4f}")
    print(f"  4-process:   {par4_value:.4f}")
    print(f"  {cpu_count}-process:  {par_max_value:.4f}")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    speedup_4 = seq_time / par4_time
    speedup_max = seq_time / par_max_time
    efficiency_4 = speedup_4 / 4
    efficiency_max = speedup_max / cpu_count

    print(f"\nParallelization Efficiency:")
    print(f"  4 processes:   {speedup_4:.2f}x speedup ({efficiency_4*100:.1f}% efficiency)")
    print(f"  {cpu_count} processes: {speedup_max:.2f}x speedup ({efficiency_max*100:.1f}% efficiency)")

    if speedup_max > cpu_count * 0.75:
        print(f"\n✅ Parallel MCTS scales excellently!")
    elif speedup_max > cpu_count * 0.5:
        print(f"\n✅ Parallel MCTS is working well!")
    elif speedup_max > cpu_count * 0.3:
        print(f"\n⚠️  Parallel MCTS provides moderate speedup")
    else:
        print(f"\n⚠️  Parallel MCTS overhead is significant - may benefit from optimization")

    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
