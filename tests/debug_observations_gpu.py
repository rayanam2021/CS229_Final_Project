"""
tests/debug_observations_gpu_entropy.py

A GPU observation test where entropy MUST decrease.
Camera is positioned close to the voxel grid, pointed straight at the target.
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from camera.camera_observations import (
    VoxelGrid,
    GroundTruthRSO,
    simulate_observation,
    calculate_entropy,
)


def test_entropy_decreasing_gpu(num_steps=8):
    print("=== GPU Observation Test (Entropy Should Decrease) ===")

    if not TORCH_AVAILABLE:
        print("PyTorch is not installed → skipping GPU test.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using torch device: {device}")

    # Make the grid large enough to see something
    dims = (60, 60, 60)      # 60 m × 60 m × 60 m cube (reasonable size)
    voxel_size = 1.0         # meters

    grid_gpu = VoxelGrid(
        grid_dims=dims,
        voxel_size=voxel_size,
        origin=(-30, -30, -30),   # center grid around (0,0,0)
        use_torch=True,
        device=device
    )
    rso_gpu = GroundTruthRSO(grid_gpu)

    # Camera settings — wide FOV, realistic resolution
    camera_fn = {
        "fov_degrees": 60.0,
        "sensor_res": (64, 64),
        "noise_params": {
            "p_hit_given_occupied": 0.95,
            "p_hit_given_empty": 0.01,
        },
    }

    # Camera flies around the grid at 40 m distance (close)
    radius = 40.0
    angles = np.linspace(0.0, 2.0 * np.pi, num_steps, endpoint=False)

    # Start entropy
    e_prev = calculate_entropy(grid_gpu.belief)
    print(f"Initial entropy: {e_prev:.6f}")

    for i, theta in enumerate(angles):
        cam_pos = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            10.0           # slight elevation to avoid grazing plane
        ])

        simulate_observation(grid_gpu, rso_gpu, camera_fn, cam_pos)
        e_curr = calculate_entropy(grid_gpu.belief)

        # Check nan
        has_nan = (
            torch.isnan(grid_gpu.belief).any().item()
            if isinstance(grid_gpu.belief, torch.Tensor)
            else np.isnan(grid_gpu.belief).any()
        )

        print(f"Step {i+1}/{num_steps}: entropy={e_curr:.6f}, Δentropy={e_prev - e_curr:.6f}, NaNs={has_nan}")

        if has_nan:
            print("ERROR: NaNs detected — check clamping!")
            break

        if e_curr < e_prev:
            print("Entropy decreased!")
        else:
            print("WARNING: Entropy did not decrease (possible geometry/miss)")

        e_prev = e_curr

    print("\nTest completed.")


if __name__ == "__main__":
    np.random.seed(0)
    test_entropy_decreasing_gpu()
