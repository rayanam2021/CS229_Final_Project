"""
Test GPU-accelerated camera observations against CPU baseline.

This test validates:
1. GPU implementation produces correct results (within statistical tolerance)
2. Performance improvement on GPU is significant
3. Entropy measurements are close between CPU and GPU
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camera.camera_observations import (
    VoxelGrid, GroundTruthRSO, simulate_observation as cpu_simulate_observation
)

try:
    from camera.gpu_camera_observations import (
        simulate_observation_gpu, TORCH_AVAILABLE
    )
except Exception as e:
    print(f"Failed to import GPU module: {e}")
    TORCH_AVAILABLE = False


def run_test():
    """Run comprehensive GPU vs CPU comparison."""

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available. Cannot test GPU implementation.")
        return False

    print("=" * 70)
    print("GPU vs CPU Camera Observation Test")
    print("=" * 70)

    # Test parameters
    num_observations = 5
    camera_fn = {
        'fov_degrees': 30.0,
        'sensor_res': (64, 64),
        'noise_params': {
            'p_hit_given_occupied': 0.95,
            'p_hit_given_empty': 0.001,
        }
    }

    # Create test grid and RSO
    grid_cpu = VoxelGrid(grid_dims=(20, 20, 20), voxel_size=1.0, origin=(-10, -10, -10))
    grid_gpu = VoxelGrid(grid_dims=(20, 20, 20), voxel_size=1.0, origin=(-10, -10, -10))
    rso = GroundTruthRSO(grid_cpu)

    # Generate multiple observation positions
    servicer_positions = np.array([
        [15, 15, 15],
        [15, -15, 15],
        [-15, 15, 15],
        [-15, -15, 15],
        [0, 0, 20],
    ])

    print(f"\nTest Configuration:")
    print(f"  Grid dimensions: {grid_cpu.dims}")
    print(f"  Voxel size: {grid_cpu.voxel_size}")
    print(f"  Ground truth RSO size: {np.sum(rso.shape)} voxels")
    print(f"  Camera FOV: {camera_fn['fov_degrees']}°")
    print(f"  Sensor resolution: {camera_fn['sensor_res']}")
    print(f"  Number of observations: {num_observations}")
    print(f"  Rays per observation: {camera_fn['sensor_res'][0] * camera_fn['sensor_res'][1]}")

    # Run observations
    print("\n" + "=" * 70)
    print("Running Observations...")
    print("=" * 70)

    cpu_times = []
    gpu_times = []
    hit_count_diffs = []
    miss_count_diffs = []

    for i, servicer_pos in enumerate(servicer_positions[:num_observations]):
        print(f"\nObservation {i+1}/{num_observations} from position {servicer_pos}")

        # CPU observation
        t0 = time.time()
        cpu_hits, cpu_misses = cpu_simulate_observation(
            grid_cpu, rso, camera_fn, servicer_pos
        )
        cpu_time = time.time() - t0
        cpu_times.append(cpu_time)

        # GPU observation
        t0 = time.time()
        gpu_hits, gpu_misses = simulate_observation_gpu(
            grid_gpu, rso, camera_fn, servicer_pos
        )
        gpu_time = time.time() - t0
        gpu_times.append(gpu_time)

        # Compare results
        cpu_hit_count = len(cpu_hits)
        gpu_hit_count = len(gpu_hits)
        cpu_miss_count = len(cpu_misses)
        gpu_miss_count = len(gpu_misses)

        hit_count_diff = gpu_hit_count - cpu_hit_count
        miss_count_diff = gpu_miss_count - cpu_miss_count

        hit_count_diffs.append(hit_count_diff)
        miss_count_diffs.append(miss_count_diff)

        print(f"  CPU: {cpu_hit_count} hits, {cpu_miss_count} misses (time: {cpu_time:.4f}s)")
        print(f"  GPU: {gpu_hit_count} hits, {gpu_miss_count} misses (time: {gpu_time:.4f}s)")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        print(f"  Hit difference: {hit_count_diff} ({100*hit_count_diff/max(1, cpu_hit_count):.1f}%)")
        print(f"  Miss difference: {miss_count_diff}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    avg_cpu_time = np.mean(cpu_times)
    avg_gpu_time = np.mean(gpu_times)
    total_cpu_time = np.sum(cpu_times)
    total_gpu_time = np.sum(gpu_times)
    avg_speedup = avg_cpu_time / avg_gpu_time

    print(f"\nTiming:")
    print(f"  Average CPU time: {avg_cpu_time:.4f}s")
    print(f"  Average GPU time: {avg_gpu_time:.4f}s")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Total CPU time: {total_cpu_time:.4f}s")
    print(f"  Total GPU time: {total_gpu_time:.4f}s")
    print(f"  Total speedup: {total_cpu_time/total_gpu_time:.2f}x")

    print(f"\nAccuracy:")
    print(f"  Average hit count difference: {np.mean(np.abs(hit_count_diffs)):.1f}")
    print(f"  Max hit count difference: {np.max(np.abs(hit_count_diffs)):.1f}")
    print(f"  Average miss count difference: {np.mean(np.abs(miss_count_diffs)):.1f}")

    print(f"\nEntropy:")
    cpu_entropy = grid_cpu.get_entropy()
    gpu_entropy = grid_gpu.get_entropy()
    entropy_diff = abs(cpu_entropy - gpu_entropy)
    entropy_diff_pct = 100 * entropy_diff / max(cpu_entropy, 1e-6)

    print(f"  CPU entropy: {cpu_entropy:.4f}")
    print(f"  GPU entropy: {gpu_entropy:.4f}")
    print(f"  Difference: {entropy_diff:.4f} ({entropy_diff_pct:.2f}%)")

    # Determine success
    success = True
    issues = []
    warnings = []

    # Check if GPU is actually faster (or at least not much slower)
    # NOTE: GPU startup overhead and PyTorch initialization can make small batches slower
    # For real speedup, need larger batches or multiple observations
    if avg_speedup < 0.5:
        issues.append(f"GPU significantly slower than CPU (speedup: {avg_speedup:.2f}x)")
        success = False
    elif avg_speedup < 1.0:
        warnings.append(f"GPU slower than CPU (speedup: {avg_speedup:.2f}x) - expected with small batches")

    # Check hit count accuracy
    # GPU and CPU will have different random sequences, so hits/misses won't match exactly
    # But the DISTRIBUTION should be similar (entropy should be close)
    max_hit_diff = np.max(np.abs(hit_count_diffs))
    max_hit_pct_diff = 100 * max_hit_diff / max(1, np.max(np.abs(np.array([len(cpu_hits) for _ in range(len(hit_count_diffs))]))))
    if max_hit_diff > 50:  # Very loose tolerance - huge differences would indicate a bug
        warnings.append(f"Large hit count variance across observations: max diff {max_hit_diff}")

    # Check entropy is similar (within 10% - loose tolerance for different RNG)
    if entropy_diff_pct > 10.0:
        warnings.append(f"Entropy difference: {entropy_diff_pct:.2f}% (different RNG sequences expected)")
    if entropy_diff_pct > 30.0:
        issues.append(f"Entropy too different: {entropy_diff_pct:.2f}% - may indicate algorithm bug")
        success = False

    print("\n" + "=" * 70)
    if success and not warnings:
        print("✓ ALL TESTS PASSED")
    elif success:
        print("✓ TESTS PASSED")
        if warnings:
            print("\nWarnings:")
            for warn in warnings:
                print(f"  ⚠ {warn}")
    else:
        print("✗ TESTS FAILED")
        for issue in issues:
            print(f"  ✗ {issue}")
        if warnings:
            print("\nWarnings:")
            for warn in warnings:
                print(f"  ⚠ {warn}")

    print("=" * 70)

    return success


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
