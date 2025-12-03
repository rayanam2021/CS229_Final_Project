"""
GPU-Accelerated MCTS Benchmark vs CPU MCTS

Comprehensive performance comparison:
1. Sequential MCTS (CPU baseline)
2. Parallel MCTS (multi-CPU)
3. GPU-Accelerated MCTS (GPU with batch operations)

Metrics:
- Search time per iteration
- Total wallclock time
- Device utilization
- Speedup factors

Expected Results:
- GPU MCTS: 5-8x faster than sequential CPU MCTS
- GPU MCTS: 1.5-3x faster than parallel CPU MCTS
"""

import sys
import os
import time
import numpy as np
import json
from datetime import datetime
from multiprocessing import cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from mcts.mcts import MCTS
from mcts.mcts_parallel import ParallelMCTS
from mcts.mcts_gpu import MCTSPU
from mcts.mcts_parallel_gpu import ParallelMCTSGPU
from camera.camera_observations import VoxelGrid, GroundTruthRSO
import torch


class BenchmarkResults:
    """Store and analyze benchmark results"""
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().isoformat()

    def add_result(self, name, config, timing, stats):
        """Add a benchmark result"""
        self.results[name] = {
            "config": config,
            "timing": timing,
            "stats": stats,
        }

    def get_speedup(self, baseline_name, test_name):
        """Calculate speedup relative to baseline"""
        baseline_time = self.results[baseline_name]["timing"]["total"]
        test_time = self.results[test_name]["timing"]["total"]
        return baseline_time / test_time

    def get_efficiency(self, name, num_workers):
        """Calculate parallel efficiency"""
        speedup = self.results[name]["timing"]["speedup"]
        return (speedup / num_workers) * 100

    def save_json(self, filepath):
        """Save results to JSON"""
        output = {
            "timestamp": self.timestamp,
            "results": self.results,
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {filepath}")

    def print_summary(self):
        """Print formatted summary"""
        print("\n" + "="*100)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*100)

        # Timing table
        print("\nTiming Results:")
        print(f"{'Method':<30} {'Time (s)':<15} {'Per Iter (ms)':<15} {'Device':<15}")
        print("-" * 75)

        for name, result in self.results.items():
            timing = result["timing"]
            stats = result["stats"]
            device = stats.get("device", "CPU")
            per_iter = (timing["total"] / timing["iterations"]) * 1000

            print(f"{name:<30} {timing['total']:<15.2f} {per_iter:<15.2f} {device:<15}")

        # Speedup comparison
        if len(self.results) > 1:
            baseline_name = list(self.results.keys())[0]
            baseline_time = self.results[baseline_name]["timing"]["total"]

            print(f"\nSpeedup vs Sequential Baseline ({baseline_name}):")
            print(f"{'Method':<30} {'Speedup':<15} {'Efficiency':<15}")
            print("-" * 60)

            for name, result in self.results.items():
                test_time = result["timing"]["total"]
                speedup = baseline_time / test_time
                config = result["config"]
                num_workers = config.get("num_processes", config.get("batch_size", 1))

                if name == baseline_name:
                    print(f"{name:<30} {'1.00x (baseline)':<15} {'N/A':<15}")
                else:
                    efficiency = (speedup / num_workers) * 100 if num_workers > 1 else 100
                    print(f"{name:<30} {speedup:<15.2f}x {efficiency:<15.1f}%")

        # GPU information
        print(f"\nGPU Information:")
        if torch.cuda.is_available():
            print(f"  CUDA Available: Yes")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print(f"  CUDA Available: No")

        print("\n" + "="*100)


def setup_test(grid_dims=(20, 20, 20), max_depth=3, camera_res=32):
    """Create test environment"""
    # Orbital parameters
    a_chief = 6778e3  # meters
    e_chief = 0.002
    i_chief = 51.6
    omega_chief = 0
    n_chief = 14.0 / 1440  # mean motion in rev/min converted to rev/sec

    # Camera and grid
    camera_fn = {
        'fov_degrees': 30.0,
        'sensor_res': (camera_res, camera_res),
        'noise_params': {
            'p_hit_given_occupied': 0.95,
            'p_hit_given_empty': 0.001,
        }
    }
    grid = VoxelGrid(grid_dims=grid_dims)
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
        grid_dims=grid_dims,
        lambda_dv=0.1,
        time_step=60.0,
        max_depth=max_depth,
    )

    return initial_state, mdp


def test_sequential_mcts(state, mdp, iters=100):
    """Test sequential CPU MCTS"""
    print("\n" + "="*100)
    print("Sequential CPU MCTS (1 thread)")
    print("="*100)

    mcts = MCTS(model=mdp, iters=iters, max_depth=mdp.max_depth, c=1.4, gamma=1.0)

    print(f"Running {iters} iterations...")
    start_time = time.time()
    action, value, stats = mcts.get_best_root_action(state, 0, ".", return_stats=True)
    elapsed = time.time() - start_time

    print(f"Time: {elapsed:.2f}s")
    print(f"Per iteration: {(elapsed/iters)*1000:.2f}ms")
    print(f"Best action: {action}")
    print(f"Predicted value: {value:.4f}")

    timing = {
        "total": elapsed,
        "iterations": iters,
        "per_iteration": elapsed / iters,
    }

    return timing, action, value, {"device": "CPU"}


def test_parallel_mcts(state, mdp, total_iters=100, num_processes=4):
    """Test parallel CPU MCTS"""
    print("\n" + "="*100)
    print(f"Parallel CPU MCTS ({num_processes} processes)")
    print("="*100)

    mcts = ParallelMCTS(
        model=mdp,
        iters=total_iters,
        max_depth=mdp.max_depth,
        c=1.4,
        gamma=1.0,
        num_processes=num_processes
    )

    print(f"Running {total_iters} total iterations ({total_iters//num_processes} per process)...")
    start_time = time.time()
    action, value, stats = mcts.get_best_root_action(state, 0, ".", return_stats=True)
    elapsed = time.time() - start_time

    print(f"Time: {elapsed:.2f}s")
    print(f"Per iteration: {(elapsed/total_iters)*1000:.2f}ms")
    print(f"Best action: {action}")
    print(f"Predicted value: {value:.4f}")

    timing = {
        "total": elapsed,
        "iterations": total_iters,
        "per_iteration": elapsed / total_iters,
    }

    return timing, action, value, {"device": "CPU", "num_processes": num_processes}


def test_gpu_mcts(state, mdp, iters=100, batch_size=16):
    """Test GPU-accelerated MCTS"""
    print("\n" + "="*100)
    print(f"GPU-Accelerated MCTS (batch_size={batch_size})")
    print("="*100)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available! GPU MCTS will use CPU fallback.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(0)}")

    mcts = MCTSPU(
        model=mdp,
        iters=iters,
        max_depth=mdp.max_depth,
        c=1.4,
        gamma=1.0,
        roll_policy="random",
        batch_size=batch_size,
        device=device
    )

    print(f"Running {iters} iterations...")
    start_time = time.time()
    action, value, stats = mcts.get_best_root_action(state, 0, ".", return_stats=True)
    elapsed = time.time() - start_time

    print(f"Time: {elapsed:.2f}s")
    print(f"Per iteration: {(elapsed/iters)*1000:.2f}ms")
    print(f"Best action: {action}")
    print(f"Predicted value: {value:.4f}")
    print(f"Device: {stats['device']}")
    print(f"GPU-accelerated: {stats['use_gpu']}")

    timing = {
        "total": elapsed,
        "iterations": iters,
        "per_iteration": elapsed / iters,
    }

    gpu_stats = {
        "device": device,
        "batch_size": batch_size,
        "use_gpu": stats["use_gpu"],
    }

    return timing, action, value, gpu_stats


def test_parallel_gpu_mcts(state, mdp, iters=100, batch_size=16, num_workers=None):
    """Test parallel GPU-accelerated MCTS with shared tree"""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print("\n" + "="*100)
    print(f"Parallel GPU-Accelerated MCTS ({num_workers} workers, batch_size={batch_size})")
    print("="*100)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available! Parallel GPU MCTS will use CPU fallback.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(0)}")

    mcts = ParallelMCTSGPU(
        model=mdp,
        iters=iters,
        max_depth=mdp.max_depth,
        c=1.4,
        gamma=1.0,
        roll_policy="random",
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        virtual_loss=1.0
    )

    print(f"Running {iters} iterations with {num_workers} parallel workers...")
    start_time = time.time()
    action, value, stats = mcts.get_best_root_action(state, 0, ".", return_stats=True)
    elapsed = time.time() - start_time

    print(f"Time: {elapsed:.2f}s")
    print(f"Per iteration: {(elapsed/iters)*1000:.2f}ms")
    print(f"Best action: {action}")
    print(f"Predicted value: {value:.4f}")
    print(f"Device: {stats['device']}")
    print(f"GPU-accelerated: {stats['use_gpu']}")
    print(f"Workers: {stats['num_workers']}")

    timing = {
        "total": elapsed,
        "iterations": iters,
        "per_iteration": elapsed / iters,
    }

    gpu_stats = {
        "device": device,
        "batch_size": batch_size,
        "use_gpu": stats["use_gpu"],
        "num_workers": num_workers,
    }

    return timing, action, value, gpu_stats


def run_small_benchmark():
    """Small benchmark for quick testing"""
    print("\n" + "="*100)
    print("SMALL BENCHMARK (Quick Test)")
    print("="*100)

    num_cpus = max(1, cpu_count() - 1)  # Use all but one CPU

    config = {
        "iterations": 50,
        "grid_dims": (15, 15, 15),
        "max_depth": 2,
        "camera_res": 16,
        "num_cpu_processes": num_cpus,
        "gpu_batch_size": 8,
    }

    print(f"Configuration:")
    print(f"  Iterations: {config['iterations']}")
    print(f"  Grid: {config['grid_dims']}")
    print(f"  Max depth: {config['max_depth']}")
    print(f"  Camera: {config['camera_res']}x{config['camera_res']} ({config['camera_res']**2} rays)")
    print(f"  CPU processes: {config['num_cpu_processes']} (max available)")

    state, mdp = setup_test(
        grid_dims=config["grid_dims"],
        max_depth=config["max_depth"],
        camera_res=config["camera_res"]
    )

    results = BenchmarkResults()

    # Sequential baseline
    timing, action, value, stats = test_sequential_mcts(state, mdp, iters=config["iterations"])
    results.add_result("Sequential CPU MCTS", config, timing, stats)

    # Parallel CPU
    timing, action, value, stats = test_parallel_mcts(
        state, mdp,
        total_iters=config["iterations"],
        num_processes=config["num_cpu_processes"]
    )
    results.add_result(f"Parallel CPU MCTS ({config['num_cpu_processes']} processes)", config, timing, stats)

    # GPU
    timing, action, value, stats = test_gpu_mcts(
        state, mdp,
        iters=config["iterations"],
        batch_size=config["gpu_batch_size"]
    )
    results.add_result("GPU MCTS", config, timing, stats)

    results.print_summary()
    return results


def run_medium_benchmark():
    """Medium benchmark for realistic testing (matches config.json)"""
    print("\n" + "="*100)
    print("MEDIUM BENCHMARK (Realistic Test - Matches config.json)")
    print("="*100)

    num_cpus = max(1, cpu_count() - 1)  # Use all but one CPU

    config = {
        "iterations": 100,  # mcts_iters from config
        "grid_dims": (20, 20, 20),  # From scenario_full_mcts.py line 95
        "max_depth": 5,  # max_horizon from config
        "camera_res": 64,  # sensor_res from config
        "num_cpu_processes": num_cpus,
        "gpu_batch_size": 16,
    }

    print(f"Configuration:")
    print(f"  Iterations: {config['iterations']}")
    print(f"  Grid: {config['grid_dims']}")
    print(f"  Max depth: {config['max_depth']}")
    print(f"  Camera: {config['camera_res']}x{config['camera_res']} ({config['camera_res']**2} rays)")
    print(f"  CPU processes: {config['num_cpu_processes']} (max available)")

    state, mdp = setup_test(
        grid_dims=config["grid_dims"],
        max_depth=config["max_depth"],
        camera_res=config["camera_res"]
    )

    results = BenchmarkResults()

    # Sequential baseline
    timing, action, value, stats = test_sequential_mcts(state, mdp, iters=config["iterations"])
    results.add_result("Sequential CPU MCTS", config, timing, stats)

    # Parallel CPU
    timing, action, value, stats = test_parallel_mcts(
        state, mdp,
        total_iters=config["iterations"],
        num_processes=config["num_cpu_processes"]
    )
    results.add_result(f"Parallel CPU MCTS ({config['num_cpu_processes']} processes)", config, timing, stats)

    # GPU MCTS with optimized kernels
    timing, action, value, stats = test_gpu_mcts(
        state, mdp,
        iters=config["iterations"],
        batch_size=config["gpu_batch_size"]
    )
    results.add_result("GPU MCTS (serial)", config, timing, stats)

    # Parallel GPU MCTS with virtual loss and implicit stream management
    timing, action, value, stats = test_parallel_gpu_mcts(
        state, mdp,
        iters=config["iterations"],
        batch_size=config["gpu_batch_size"]
    )
    results.add_result(f"Parallel GPU MCTS ({stats['num_workers']} workers)", config, timing, stats)

    results.print_summary()
    results.save_json("output/benchmark_results.json")
    return results


def run_scaling_benchmark():
    """Scaling benchmark with increasing iterations"""
    print("\n" + "="*100)
    print("SCALING BENCHMARK (Vary Iterations)")
    print("="*100)

    num_cpus = max(1, cpu_count() - 1)  # Use all but one CPU

    base_config = {
        "grid_dims": (20, 20, 20),
        "max_depth": 3,
        "camera_res": 32,
        "num_cpu_processes": num_cpus,
        "gpu_batch_size": 16,
    }

    state, mdp = setup_test(
        grid_dims=base_config["grid_dims"],
        max_depth=base_config["max_depth"],
        camera_res=base_config["camera_res"]
    )

    results = BenchmarkResults()

    for iters in [50, 100, 200]:
        print(f"\n--- Running with {iters} iterations ---")

        config = {**base_config, "iterations": iters}

        # Sequential
        timing, _, _, stats = test_sequential_mcts(state, mdp, iters=iters)
        results.add_result(f"Sequential CPU MCTS (iter={iters})", config, timing, stats)

        # GPU
        timing, _, _, stats = test_gpu_mcts(state, mdp, iters=iters, batch_size=base_config["gpu_batch_size"])
        results.add_result(f"GPU MCTS (iter={iters})", config, timing, stats)

    results.print_summary()
    results.save_json("output/benchmark_scaling_results.json")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU MCTS Benchmark")
    parser.add_argument("--mode", default="small",
                       choices=["small", "medium", "scaling"],
                       help="Benchmark mode: small (quick), medium (realistic), or scaling (vary iterations)")
    parser.add_argument("--output", default="output/benchmark_results.json",
                       help="Output JSON file")

    args = parser.parse_args()

    print("\n" + "="*100)
    print("GPU-ACCELERATED MCTS BENCHMARK")
    print("="*100)
    print(f"Benchmark Mode: {args.mode}")
    print(f"Output: {args.output}")

    try:
        if args.mode == "small":
            results = run_small_benchmark()
        elif args.mode == "medium":
            results = run_medium_benchmark()
        elif args.mode == "scaling":
            results = run_scaling_benchmark()

        print("\n✅ Benchmark completed successfully!")

    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
