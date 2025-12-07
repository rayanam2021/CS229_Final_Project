#!/usr/bin/env python
"""
Test that CUDA implementation produces statistically similar results to CPU original.
"""
import torch
import numpy as np
import sys

print("="*80)
print("CORRECTNESS TEST: CUDA vs CPU Original")
print("="*80)
print()

# Configuration
grid_dims = (20, 20, 20)
cam_pos_np = np.array([15.0, 0.0, 0.0])
num_trials = 10

camera_fn = {
    'fov_degrees': 60,
    'sensor_res': [32, 32],  # Smaller for faster CPU testing
    'noise_params': {'p_hit_given_occupied': 0.9, 'p_hit_given_empty': 0.1}
}

print(f"Configuration:")
print(f"  Grid: {grid_dims}")
print(f"  Sensor: {camera_fn['sensor_res']}")
print(f"  Trials: {num_trials}")
print()

# ============================================================================
# Setup CPU Original
# ============================================================================
print("Setting up CPU original implementation...")
sys.path.insert(0, '/home/saveasmtz/Documents/AA228/CS229_Final_Project/camera')
import camera_observations_orig as orig

grid_cpu = orig.VoxelGrid(grid_dims, use_torch=False, device='cpu')
rso_cpu = orig.GroundTruthRSO(grid_cpu)  # Pass grid, not tuple
rso_cpu.shape = np.zeros(grid_dims, dtype=bool)
rso_cpu.shape[5:15, 5:15, 5:15] = True  # Cube in center

print(f"  Initial CPU entropy: {orig.calculate_entropy(grid_cpu.belief):.4f}")

# ============================================================================
# Setup CUDA Implementation
# ============================================================================
print()
print("Setting up CUDA implementation...")
from camera.camera_observations import VoxelGrid, GroundTruthRSO, simulate_observation

grid_cuda = VoxelGrid(grid_dims, use_torch=True, device='cuda')
rso_cuda = GroundTruthRSO(grid_cuda)  # Pass grid, not tuple
rso_cuda.shape = torch.zeros(grid_dims, dtype=torch.bool, device='cuda')
rso_cuda.shape[5:15, 5:15, 5:15] = True  # Same cube

print(f"  Initial CUDA entropy: {orig.calculate_entropy(grid_cuda.belief.cpu().numpy()):.4f}")

# ============================================================================
# Run trials and collect statistics
# ============================================================================
print()
print("="*80)
print("Running comparison trials...")
print("="*80)
print()

cpu_entropies = []
cuda_entropies = []
cpu_hit_counts = []
cuda_hit_counts = []
cpu_miss_counts = []
cuda_miss_counts = []

for trial in range(num_trials):
    print(f"Trial {trial+1}/{num_trials}:")

    # CPU observation
    hits_cpu, misses_cpu = orig.simulate_observation(grid_cpu, rso_cpu, camera_fn, cam_pos_np)
    cpu_entropy = orig.calculate_entropy(grid_cpu.belief)
    cpu_entropies.append(cpu_entropy)
    cpu_hit_counts.append(len(hits_cpu))
    cpu_miss_counts.append(len(misses_cpu))

    print(f"  CPU:  {len(hits_cpu)} hits, {len(misses_cpu)} misses, entropy={cpu_entropy:.4f}")

    # CUDA observation
    hits_cuda, misses_cuda = simulate_observation(grid_cuda, rso_cuda, camera_fn, cam_pos_np)
    torch.cuda.synchronize()

    # Convert CUDA belief to numpy for entropy calculation
    if isinstance(grid_cuda.belief, torch.Tensor):
        belief_cuda_np = grid_cuda.belief.cpu().numpy()
    else:
        belief_cuda_np = grid_cuda.belief

    cuda_entropy = orig.calculate_entropy(belief_cuda_np)
    cuda_entropies.append(cuda_entropy)

    # Handle tensor or list outputs
    if isinstance(hits_cuda, torch.Tensor):
        n_hits_cuda = hits_cuda.shape[0]
        n_misses_cuda = misses_cuda.shape[0]
    else:
        n_hits_cuda = len(hits_cuda)
        n_misses_cuda = len(misses_cuda)

    cuda_hit_counts.append(n_hits_cuda)
    cuda_miss_counts.append(n_misses_cuda)

    print(f"  CUDA: {n_hits_cuda} hits, {n_misses_cuda} misses, entropy={cuda_entropy:.4f}")
    print()

# ============================================================================
# Statistical Analysis
# ============================================================================
print("="*80)
print("STATISTICAL COMPARISON")
print("="*80)
print()

cpu_entropies = np.array(cpu_entropies)
cuda_entropies = np.array(cuda_entropies)
cpu_hit_counts = np.array(cpu_hit_counts)
cuda_hit_counts = np.array(cuda_hit_counts)
cpu_miss_counts = np.array(cpu_miss_counts)
cuda_miss_counts = np.array(cuda_miss_counts)

print("Entropy Statistics:")
print("-"*80)
print(f"  CPU:   mean={cpu_entropies.mean():.4f}, std={cpu_entropies.std():.4f}, range=[{cpu_entropies.min():.4f}, {cpu_entropies.max():.4f}]")
print(f"  CUDA:  mean={cuda_entropies.mean():.4f}, std={cuda_entropies.std():.4f}, range=[{cuda_entropies.min():.4f}, {cuda_entropies.max():.4f}]")
print(f"  Difference: {abs(cpu_entropies.mean() - cuda_entropies.mean()):.4f} ({abs(cpu_entropies.mean() - cuda_entropies.mean())/cpu_entropies.mean()*100:.2f}%)")

print()
print("Hit Count Statistics:")
print("-"*80)
print(f"  CPU:   mean={cpu_hit_counts.mean():.1f}, std={cpu_hit_counts.std():.1f}, range=[{cpu_hit_counts.min()}, {cpu_hit_counts.max()}]")
print(f"  CUDA:  mean={cuda_hit_counts.mean():.1f}, std={cuda_hit_counts.std():.1f}, range=[{cuda_hit_counts.min()}, {cuda_hit_counts.max()}]")
print(f"  Difference: {abs(cpu_hit_counts.mean() - cuda_hit_counts.mean()):.1f} ({abs(cpu_hit_counts.mean() - cuda_hit_counts.mean())/cpu_hit_counts.mean()*100:.2f}%)")

print()
print("Miss Count Statistics:")
print("-"*80)
print(f"  CPU:   mean={cpu_miss_counts.mean():.1f}, std={cpu_miss_counts.std():.1f}, range=[{cpu_miss_counts.min()}, {cpu_miss_counts.max()}]")
print(f"  CUDA:  mean={cuda_miss_counts.mean():.1f}, std={cuda_miss_counts.std():.1f}, range=[{cuda_miss_counts.min()}, {cuda_miss_counts.max()}]")
print(f"  Difference: {abs(cpu_miss_counts.mean() - cuda_miss_counts.mean()):.1f} ({abs(cpu_miss_counts.mean() - cuda_miss_counts.mean())/cpu_miss_counts.mean()*100:.2f}%)")

# ============================================================================
# Belief State Comparison
# ============================================================================
print()
print("="*80)
print("BELIEF STATE COMPARISON (after all observations)")
print("="*80)
print()

belief_cpu = grid_cpu.belief
belief_cuda = grid_cuda.belief.cpu().numpy() if isinstance(grid_cuda.belief, torch.Tensor) else grid_cuda.belief

# Calculate statistics
mse = np.mean((belief_cpu - belief_cuda)**2)
mae = np.mean(np.abs(belief_cpu - belief_cuda))
max_diff = np.max(np.abs(belief_cpu - belief_cuda))
correlation = np.corrcoef(belief_cpu.flatten(), belief_cuda.flatten())[0, 1]

print("Voxel-wise Belief Comparison:")
print("-"*80)
print(f"  Mean Squared Error (MSE):     {mse:.6f}")
print(f"  Mean Absolute Error (MAE):    {mae:.6f}")
print(f"  Max Absolute Difference:      {max_diff:.6f}")
print(f"  Correlation:                  {correlation:.6f}")

# Check if beliefs are statistically similar
similar = mse < 0.01 and correlation > 0.99

print()
if similar:
    print("✅ PASS: CUDA and CPU implementations produce statistically similar results!")
else:
    print("⚠️  WARNING: Significant differences detected between CUDA and CPU.")
    print("   This could be due to:")
    print("   - Different random number generation")
    print("   - Numerical precision differences")
    print("   - Implementation bugs")

print()
print("="*80)
print("Correctness Test Complete!")
print("="*80)
