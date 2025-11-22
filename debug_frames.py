#!/usr/bin/env python3
"""
Debug script for create_visualization_frames in scenario_full_mcts.py

Usage:
    python debug_create_visualization_frames.py
"""

import os
import numpy as np
import imageio.v2 as imageio

import matplotlib
matplotlib.use("Agg")  # headless backend for saving figures

import simulation.scenario_full_mcts as sim  # make sure this is importable (same folder or on PYTHONPATH)


def main():
    # --- Output folder ---
    out_folder = "debug_vis"
    os.makedirs(out_folder, exist_ok=True)

    # --- Minimal grid + RSO (matches your main sim) ---
    grid = sim.VoxelGrid(grid_dims=(20, 20, 20))
    rso = sim.GroundTruthRSO(grid)

    # --- Camera config (same structure as in scenario_full_mcts) ---
    camera_fn = {
        "fov_degrees": 10.0,
        "sensor_res": (64, 64),
        "noise_params": {
            "p_hit_given_occupied": 0.95,
            "p_hit_given_empty": 0.001,
        },
    }

    # --- Synthetic trajectory: N positions on a circle ---
    num_steps = 20
    time_step = 10.0
    radius = 500.0  # meters (arbitrary for visualization)
    z_height = 100.0

    camera_positions = []
    view_directions = []

    for k in range(num_steps):
        theta = 2.0 * np.pi * k / num_steps
        pos = np.array([radius * np.cos(theta),
                        radius * np.sin(theta),
                        z_height])
        camera_positions.append(pos)
        view_directions.append(-pos / np.linalg.norm(pos))  # look at origin

    print("========== DEBUG: create_visualization_frames ==========")
    print(f"len(camera_positions) = {len(camera_positions)}")
    print(f"Example camera_position[0] = {camera_positions[0]}")
    print(f"Example view_direction[0] = {view_directions[0]}")

    # --- Call the function under test ---
    frames = sim.create_visualization_frames(
        out_folder=out_folder,
        grid_initial=grid,
        rso=rso,
        camera_fn=camera_fn,
        camera_positions=camera_positions,
        view_directions=view_directions,
        num_steps=num_steps,
        time_step=time_step,
    )

    # Sanity: check that consecutive frames differ in pixels
    if len(frames) > 1:
        diff01 = np.abs(frames[0].astype(int) - frames[1].astype(int)).mean()
        diff0last = np.abs(frames[0].astype(int) - frames[-1].astype(int)).mean()
        print(f"Mean pixel diff frame0-frame1  = {diff01}")
        print(f"Mean pixel diff frame0-framelast = {diff0last}")

    print(f"len(frames) = {len(frames)}")

    if not frames:
        print("❌ No frames were returned!")
        return

    # Print a bit more info about the frames
    for i, fr in enumerate(frames[:3]):
        print(f"frame[{i}] shape = {fr.shape}, dtype = {fr.dtype}")

    # --- Save a few individual frames to inspect visually ---
    indices_to_save = [0, len(frames) // 2, len(frames) - 1]
    indices_to_save = sorted(set(idx for idx in indices_to_save if 0 <= idx < len(frames)))

    for idx in indices_to_save:
        png_path = os.path.join(out_folder, f"debug_frame_{idx:03d}.png")
        imageio.imwrite(png_path, frames[idx])
        print(f"Saved frame {idx} to {png_path}")

    # --- Save a debug video with all frames ---
    video_path = os.path.join(out_folder, "debug_video.mp4")
    imageio.mimsave(
        video_path,
        frames,
        format="MP4",
        fps=5,
        codec="libx264",
        macro_block_size=1,
    )
    print(f"Saved debug video to {video_path}")
    print("========================================================")


if __name__ == "__main__":
    main()
