"""
Test and benchmark CUDA ray-tracing kernel vs PyTorch implementation.
"""

import sys
import numpy as np
import torch
import time

sys.path.insert(0, '/home/saveasmtz/Documents/CS229_Final_Project')

from camera.gpu_camera import voxel_traversal_dda_batched


def benchmark_dda(num_rays=4096, grid_dims=(20, 20, 20), voxel_size=1.0, device="cuda:0"):
    """Benchmark DDA ray traversal."""
    print(f"\n{'='*70}")
    print(f"Benchmarking DDA Ray Traversal")
    print(f"{'='*70}")
    print(f"Rays: {num_rays}")
    print(f"Grid dims: {grid_dims}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    # Setup
    origin = np.array([-10, -10, -10], dtype=np.float32)
    max_bound = origin + np.array(grid_dims) * voxel_size

    # Random rays
    rays_o = torch.randn(num_rays, 3, dtype=torch.float32, device=device) * 5
    rays_d = torch.randn(num_rays, 3, dtype=torch.float32, device=device)
    rays_d = rays_d / (torch.norm(rays_d, dim=1, keepdim=True) + 1e-9)

    # Warm up
    print("\nWarming up GPU...")
    for _ in range(2):
        _ = voxel_traversal_dda_batched(
            rays_o, rays_d, origin, max_bound, grid_dims, voxel_size,
            use_cuda_kernel=False  # Warmup with PyTorch
        )
    torch.cuda.synchronize(device)

    # Benchmark PyTorch version
    print("\n[PyTorch Version]")
    times_pytorch = []
    for trial in range(5):
        torch.cuda.synchronize(device)
        t0 = time.time()
        voxels_torch, steps_torch = voxel_traversal_dda_batched(
            rays_o, rays_d, origin, max_bound, grid_dims, voxel_size,
            use_cuda_kernel=False
        )
        torch.cuda.synchronize(device)
        t1 = time.time()
        times_pytorch.append(t1 - t0)

    avg_pytorch = np.mean(times_pytorch)
    print(f"  Time: {avg_pytorch*1000:.2f} ms (σ={np.std(times_pytorch)*1000:.2f} ms)")

    # Benchmark CUDA kernel version
    print("\n[CUDA Kernel Version]")
    times_cuda = []
    try:
        for trial in range(5):
            torch.cuda.synchronize(device)
            t0 = time.time()
            voxels_cuda, steps_cuda = voxel_traversal_dda_batched(
                rays_o, rays_d, origin, max_bound, grid_dims, voxel_size,
                use_cuda_kernel=True
            )
            torch.cuda.synchronize(device)
            t1 = time.time()
            times_cuda.append(t1 - t0)

        avg_cuda = np.mean(times_cuda)
        print(f"  Time: {avg_cuda*1000:.2f} ms (σ={np.std(times_cuda)*1000:.2f} ms)")

        # Compare results
        print("\n[Correctness Check]")
        voxels_diff = torch.abs(voxels_cuda - voxels_torch)
        steps_diff = torch.abs(steps_cuda - steps_torch)
        print(f"  Max voxel difference: {voxels_diff.max().item()}")
        print(f"  Max steps difference: {steps_diff.max().item()}")

        # Speedup
        speedup = avg_pytorch / avg_cuda
        print(f"\n[Results]")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  PyTorch: {avg_pytorch*1000:.2f} ms")
        print(f"  CUDA:    {avg_cuda*1000:.2f} ms")

        if speedup < 1.0:
            print(f"  ⚠️  CUDA is slower - PyTorch may be better for this workload")
        else:
            print(f"  ✓ CUDA kernel is faster")

    except Exception as e:
        print(f"  ✗ CUDA kernel failed: {e}")
        print(f"  (This is expected if Numba CUDA is not available)")


def test_correctness():
    """Test that CUDA kernel produces correct results."""
    print(f"\n{'='*70}")
    print(f"Correctness Test")
    print(f"{'='*70}")

    device = "cuda:0"
    grid_dims = (10, 10, 10)
    voxel_size = 1.0
    origin = np.array([-5, -5, -5], dtype=np.float32)
    max_bound = origin + np.array(grid_dims) * voxel_size

    # Simple test: single ray pointing into grid
    num_rays = 1
    rays_o = torch.tensor([[-8.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    rays_d = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    voxels_torch, steps_torch = voxel_traversal_dda_batched(
        rays_o, rays_d, origin, max_bound, grid_dims, voxel_size,
        use_cuda_kernel=False
    )

    try:
        voxels_cuda, steps_cuda = voxel_traversal_dda_batched(
            rays_o, rays_d, origin, max_bound, grid_dims, voxel_size,
            use_cuda_kernel=True
        )

        match = torch.allclose(voxels_torch.float(), voxels_cuda.float(), atol=0) and \
                torch.allclose(steps_torch.float(), steps_cuda.float(), atol=0)

        if match:
            print(f"✓ Results match between PyTorch and CUDA")
        else:
            print(f"✗ Results differ:")
            print(f"  PyTorch steps: {steps_torch[0].item()}")
            print(f"  CUDA steps:    {steps_cuda[0].item()}")
            print(f"  Voxels (first 5):")
            print(f"    PyTorch: {voxels_torch[0, :5, :].cpu().numpy()}")
            print(f"    CUDA:    {voxels_cuda[0, :5, :].cpu().numpy()}")
    except Exception as e:
        print(f"✗ CUDA kernel test failed: {e}")


if __name__ == "__main__":
    print("CUDA Ray-Tracing Kernel Tests")
    print("=" * 70)

    # Test correctness
    test_correctness()

    # Small benchmark
    benchmark_dda(num_rays=1024, grid_dims=(20, 20, 20), device="cuda:0")

    # Larger benchmark
    benchmark_dda(num_rays=4096, grid_dims=(20, 20, 20), device="cuda:0")

    print(f"\n{'='*70}")
    print("Tests complete!")
