"""
Python wrapper for CUDA ray tracing kernel.
"""
import torch
import numpy as np
import os
import sys

# Try to import the compiled CUDA extension
try:
    import ray_tracing_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA ray tracing extension not compiled. Run: cd camera/cuda && python setup.py install")


def trace_rays_cuda(ray_origins, ray_dirs, rso_shape, grid_origin, voxel_size,
                     p_hit_occ, p_hit_emp, seed=None, return_tensors=True):
    """
    CUDA-accelerated ray tracing.

    Args:
        ray_origins: (N, 3) torch.Tensor (float32, CUDA)
        ray_dirs: (N, 3) torch.Tensor (float32, CUDA)
        rso_shape: (grid_x, grid_y, grid_z) torch.Tensor (bool, CUDA)
        grid_origin: (3,) torch.Tensor (float32, CUDA)
        voxel_size: float
        p_hit_occ: float (probability of hit given occupied)
        p_hit_emp: float (probability of hit given empty)
        seed: int (random seed)
        return_tensors: bool (if True, return GPU tensors; if False, return lists)

    Returns:
        If return_tensors=True:
            hits: (M, 3) torch.Tensor (int32, CUDA) - hit coordinates
            misses: (K, 3) torch.Tensor (int32, CUDA) - miss coordinates
        If return_tensors=False:
            hits: list of tuples (x, y, z)
            misses: list of tuples (x, y, z)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA extension not available")

    if seed is None:
        seed = np.random.randint(0, 2**32)

    # Call CUDA kernel
    result = ray_tracing_cuda.trace_rays(
        ray_origins.contiguous(),
        ray_dirs.contiguous(),
        rso_shape.contiguous(),
        grid_origin.contiguous(),
        float(voxel_size),
        float(p_hit_occ),
        float(p_hit_emp),
        int(seed)
    )

    # Parse results
    num_rays = ray_origins.shape[0]

    hit_coords_flat = result[0].view(num_rays, 3)
    hit_count = result[1]
    miss_coords_flat = result[2].view(num_rays * 120, 3)  # MAX_STEPS = 120
    miss_counts = result[3]

    if return_tensors:
        # GPU-optimized path: extract valid coordinates using fully vectorized operations
        # Extract hits (keep on GPU)
        hit_mask = hit_count > 0
        hits_tensor = hit_coords_flat[hit_mask]  # (M, 3) where M is number of hits

        # Extract misses (fully vectorized, no Python loops or .item() calls)
        # Create a mask for all valid miss entries
        # Each ray can have up to MAX_STEPS misses, stored sequentially
        MAX_STEPS = 120

        # Expand miss_counts to create a mask: [ray0_count, ray0_count, ..., ray1_count, ray1_count, ...]
        # Shape: (num_rays * MAX_STEPS,)
        ray_indices = torch.arange(num_rays, device=miss_counts.device).repeat_interleave(MAX_STEPS)
        step_indices = torch.arange(MAX_STEPS, device=miss_counts.device).repeat(num_rays)

        # Create mask: step_indices < miss_counts[ray_indices]
        miss_mask = step_indices < miss_counts[ray_indices]

        # Extract valid misses using the mask
        misses_tensor = miss_coords_flat[miss_mask]  # (K, 3) where K is total valid misses

        return hits_tensor, misses_tensor
    else:
        # Legacy path: return lists (slow due to CPU-GPU transfers)
        # Extract hits
        hits = []
        for i in range(num_rays):
            if hit_count[i] > 0:
                hit_coord = hit_coords_flat[i].cpu().numpy()
                hits.append(tuple(hit_coord))

        # Extract misses
        misses = []
        offset = 0
        for i in range(num_rays):
            count = miss_counts[i].item()
            for j in range(count):
                miss_coord = miss_coords_flat[offset + j].cpu().numpy()
                misses.append(tuple(miss_coord))
            offset += 120  # MAX_STEPS

        return hits, misses
