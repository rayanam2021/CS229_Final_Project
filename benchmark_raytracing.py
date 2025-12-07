#!/usr/bin/env python
"""
Comprehensive ray-tracing benchmark comparing all implementations.
"""
import torch
import numpy as np
import time
import sys

print("="*80)
print("COMPREHENSIVE RAY-TRACING BENCHMARK")
print("="*80)
print()

# Configuration
n_rays = 4096  # 64x64
grid_dims = (20, 20, 20)
cam_pos_np = np.array([15.0, 0.0, 0.0])

print(f"Configuration:")
print(f"  Rays: {n_rays} ({int(np.sqrt(n_rays))}x{int(np.sqrt(n_rays))})")
print(f"  Grid: {grid_dims}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print()

# ============================================================================
# TEST 1: CUDA Kernel (Fully Optimized - GPU Tensors)
# ============================================================================
print("[1/4] CUDA Kernel (Fully Optimized - GPU Tensors)")
print("-"*80)

from camera.cuda.cuda_wrapper import trace_rays_cuda

ray_origins = torch.randn(n_rays, 3, device='cuda') * 5.0
ray_dirs = torch.randn(n_rays, 3, device='cuda')
ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=1, keepdim=True)
grid_shape = torch.zeros(grid_dims, dtype=torch.bool, device='cuda')
grid_shape[5:15, 5:15, 5:15] = True
grid_min = torch.tensor([-10.0, -10.0, -10.0], device='cuda')

# Warmup
_ = trace_rays_cuda(ray_origins, ray_dirs, grid_shape, grid_min, 1.0, 0.9, 0.1, return_tensors=True)
torch.cuda.synchronize()

# Benchmark
runs = 100
start = time.time()
for _ in range(runs):
    hits, misses = trace_rays_cuda(ray_origins, ray_dirs, grid_shape, grid_min, 1.0, 0.9, 0.1, return_tensors=True)
    torch.cuda.synchronize()
cuda_opt_time = (time.time() - start) / runs * 1000

print(f"  Time: {cuda_opt_time:.2f}ms")
print(f"  Throughput: {n_rays / (cuda_opt_time/1000):.0f} rays/sec")
print(f"  Output: {hits.shape[0]} hits, {misses.shape[0]} misses (GPU tensors)")

# ============================================================================
# TEST 2: CUDA Kernel (Legacy - Returns Python Lists)
# ============================================================================
print()
print("[2/4] CUDA Kernel (Legacy Wrapper - Returns Lists)")
print("-"*80)

# Warmup
_ = trace_rays_cuda(ray_origins, ray_dirs, grid_shape, grid_min, 1.0, 0.9, 0.1, return_tensors=False)
torch.cuda.synchronize()

# Benchmark (fewer runs due to slowness)
runs = 10
start = time.time()
for _ in range(runs):
    hits, misses = trace_rays_cuda(ray_origins, ray_dirs, grid_shape, grid_min, 1.0, 0.9, 0.1, return_tensors=False)
    torch.cuda.synchronize()
cuda_legacy_time = (time.time() - start) / runs * 1000

print(f"  Time: {cuda_legacy_time:.2f}ms")
print(f"  Throughput: {n_rays / (cuda_legacy_time/1000):.0f} rays/sec")
print(f"  Output: {len(hits)} hits, {len(misses)} misses (Python lists)")

# ============================================================================
# TEST 3: CPU Original Implementation
# ============================================================================
print()
print("[3/4] CPU Original (Sequential Ray Tracing)")
print("-"*80)

try:
    # Import from original file
    sys.path.insert(0, '/home/saveasmtz/Documents/AA228/CS229_Final_Project/camera')
    import camera_observations_orig as orig

    # Setup
    grid_cpu = orig.VoxelGrid(grid_dims, use_torch=False, device='cpu')
    rso_cpu = orig.GroundTruthRSO(grid_dims)
    rso_cpu.shape = np.ones(grid_dims, dtype=bool)
    rso_cpu.shape[5:15, 5:15, 5:15] = True

    camera_fn = {
        'fov_degrees': 60,
        'sensor_res': [64, 64],
        'noise_params': {'p_hit_given_occupied': 0.9, 'p_hit_given_empty': 0.1}
    }

    # Warmup
    _ = orig.simulate_observation(grid_cpu, rso_cpu, camera_fn, cam_pos_np)

    # Benchmark (very few runs due to slowness)
    runs = 3
    print(f"  Running {runs} iterations (CPU is slow)...")
    start = time.time()
    for i in range(runs):
        hits, misses = orig.simulate_observation(grid_cpu, rso_cpu, camera_fn, cam_pos_np)
        print(f"    Iteration {i+1}/{runs} complete...")
    cpu_time = (time.time() - start) / runs * 1000

    print(f"  Time: {cpu_time:.2f}ms")
    print(f"  Throughput: {n_rays / (cpu_time/1000):.0f} rays/sec")
    print(f"  Output: {len(hits)} hits, {len(misses)} misses")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    cpu_time = None

# ============================================================================
# TEST 4: Current GPU Implementation
# ============================================================================
print()
print("[4/4] Current Implementation (with CUDA enabled)")
print("-"*80)

try:
    from camera.camera_observations import VoxelGrid, GroundTruthRSO, simulate_observation

    # Setup
    grid = VoxelGrid(grid_dims, use_torch=True, device='cuda')
    rso = GroundTruthRSO(grid_dims)
    rso.shape = grid_shape

    camera_fn = {
        'fov_degrees': 60,
        'sensor_res': [64, 64],
        'noise_params': {'p_hit_given_occupied': 0.9, 'p_hit_given_empty': 0.1}
    }

    # Warmup
    _ = simulate_observation(grid, rso, camera_fn, cam_pos_np)
    torch.cuda.synchronize()

    # Benchmark
    runs = 50
    start = time.time()
    for _ in range(runs):
        hits, misses = simulate_observation(grid, rso, camera_fn, cam_pos_np)
        torch.cuda.synchronize()
    current_time = (time.time() - start) / runs * 1000

    print(f"  Time: {current_time:.2f}ms")
    print(f"  Throughput: {n_rays / (current_time/1000):.0f} rays/sec")
    if isinstance(hits, torch.Tensor):
        print(f"  Output: {hits.shape[0]} hits, {misses.shape[0]} misses (GPU tensors)")
    else:
        print(f"  Output: {len(hits)} hits, {len(misses)} misses")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    current_time = None

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print()
print(f"{'Implementation':<45} {'Time (ms)':<12} {'Speedup':<15}")
print("-"*80)

if cpu_time:
    print(f"{'CPU Original (Sequential)':<45} {cpu_time:>8.2f}     {'1.0x (baseline)':<15}")
if current_time:
    speedup = cpu_time / current_time if cpu_time else 0
    print(f"{'Current (CUDA Optimized)':<45} {current_time:>8.2f}     {speedup:>10.1f}x")
print(f"{'CUDA Legacy Wrapper':<45} {cuda_legacy_time:>8.2f}     {cpu_time/cuda_legacy_time if cpu_time else 0:>10.1f}x")
print(f"{'CUDA Fully Optimized':<45} {cuda_opt_time:>8.2f}     {cpu_time/cuda_opt_time if cpu_time else 0:>10.0f}x")

print()
print("="*80)
print("RESEARCH IMPACT: Episode Time Estimates")
print("="*80)
print()
print("Pure MCTS (750,000 observations per episode):")
print("-"*80)
if cpu_time:
    print(f"  CPU Original:       {(cpu_time/1000)*750000/3600:>8.1f} hours  ({(cpu_time/1000)*750000/3600/24:.1f} days)")
if current_time:
    print(f"  Current (CUDA):     {(current_time/1000)*750000/3600:>8.2f} hours")
print(f"  CUDA Legacy:        {(cuda_legacy_time/1000)*750000/3600:>8.1f} hours")
print(f"  CUDA Optimized:     {(cuda_opt_time/1000)*750000/3600:>8.2f} hours")

print()
print("AlphaZero (25,000 observations per episode):")
print("-"*80)
if cpu_time:
    print(f"  CPU Original:       {(cpu_time/1000)*25000/3600:>8.2f} hours")
if current_time:
    print(f"  Current (CUDA):     {(current_time/1000)*25000/60:>8.2f} minutes")
print(f"  CUDA Legacy:        {(cuda_legacy_time/1000)*25000/60:>8.1f} minutes")
print(f"  CUDA Optimized:     {(cuda_opt_time/1000)*25000/60:>8.2f} minutes")

print()
print("="*80)
print("Benchmark Complete!")
print("="*80)
