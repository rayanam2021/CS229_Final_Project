"""
Orbital Camera Simulation with Full MCTS Tree Search (Horizon=3, Parallel Evaluation).

This version uses a 3-level MCTS tree with all 13 actions at each level,
evaluating 13^3 = 2,197 distinct paths using parallel processing
and selecting the optimal first action.
"""

import os
import csv
import imageio
import time as tm
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from roe.dynamics import apply_impulsive_dv
from roe.propagation import propagateGeomROE, rtn_to_roe
from mcts.mcts_controller import MCTSController
from camera.camera_observations import VoxelGrid, GroundTruthRSO, calculate_entropy, simulate_observation, plot_scenario, update_plot


# --- FUNCTION FOR OFFLINE VISUALIZATION ---
def create_visualization_frames(out_folder, grid_initial, rso, camera_fn, camera_positions, view_directions, num_steps, time_step):
    """
    Creates all visualization frames offline for video export after the simulation.
    A new, temporary VoxelGrid is used and updated for visualization purposes only, 
    to properly show the state progression in each frame.
    
    This function also calculates the required axis limits to encompass the RSO, the grid, 
    and the entire servicer path, ensuring appropriate scaling.
    """
    print("\n🎬 Generating visualization frames for video export...")
    
    camera_positions = np.asarray(camera_positions)
    view_directions = np.asarray(view_directions)

    # 1. Initialize a new grid for visualization purposes only
    vis_grid = VoxelGrid(grid_dims=grid_initial.dims, voxel_size=grid_initial.voxel_size, origin=grid_initial.origin)
    
    # 2. Determine plot limits to encompass the entire path and the grid
    all_x = np.array([pos[0] for pos in camera_positions] + [vis_grid.origin[0], vis_grid.max_bound[0]])
    all_y = np.array([pos[1] for pos in camera_positions] + [vis_grid.origin[1], vis_grid.max_bound[1]])
    all_z = np.array([pos[2] for pos in camera_positions] + [vis_grid.origin[2], vis_grid.max_bound[2]])
    
    max_range_dim = np.array([np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)]).max() / 2.0
    if max_range_dim == 0: max_range_dim = 15.0 # Default fallback
    
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5
    
    plot_extent = max_range_dim + max_range_dim * 0.2 + 5.0 
    
    x_lim = (mid_x - plot_extent, mid_x + plot_extent)
    y_lim = (mid_y - plot_extent, mid_y + plot_extent)
    z_lim = (mid_z - plot_extent, mid_z + plot_extent)

    # 3. Create initial plot
    init_pos = camera_positions[0]
    init_view = view_directions[0]
    fig, ax, artists = plot_scenario(vis_grid, rso, init_pos, init_view, camera_fn['fov_degrees'], camera_fn['sensor_res'])
    
    # Manually set limits
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_box_aspect([1,1,1]) # Set equal aspect ratio for 3D plot
    
    frames = []
    
    # 4. Loop through each frame, update belief (simulated observation), and capture
    for frame in range(len(camera_positions)):
        cam_hist = camera_positions[:frame + 1]
        view_hist = view_directions[:frame + 1]
        current_camera_pos = cam_hist[-1]

        # update vis grid
        simulate_observation(vis_grid, rso, camera_fn, current_camera_pos)

        # update plot
        update_plot(
            frame,
            vis_grid,
            rso,
            cam_hist,
            view_hist,
            camera_fn['fov_degrees'],
            camera_fn['sensor_res'],
            camera_fn['noise_params'],
            ax,
            artists,
            np.array([0.0, 0.0, 0.0]),
        )

        # optional: make frame index visible in the title
        ax.set_title(f"RSO Characterization - Frame {frame+1}/{len(camera_positions)}")

        # --- KEY CHANGE: copy the image data ---
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        rgba_buffer = np.asarray(renderer.buffer_rgba())
        frame_image = rgba_buffer[:, :, :3].copy()   # <-- COPY HERE
        frames.append(frame_image)

    plt.close(fig) # Close the figure after generating all frames

    print("np.may_share_memory(frames[0], frames[1]):",
      np.may_share_memory(frames[0], frames[1]))
    
    return frames


def run_orbital_camera_sim_full_mcts(target_radius, gamma_r, r_min_rollout, r_max_rollout,
                                     horizon=5, num_steps=20, time_step=10.0,
                                     mcts_iters=1, mcts_c=1.4, mcts_gamma=0.99, lambda_dv=0.01,
                                     alpha_dv=1, beta_tan=1,
                                     verbose=False, visualize=True, out_folder=None,
                                     use_torch_grid=False, grid_device=None):
    """
    High-level simulation combining:
    - Full MCTS tree search (configurable horizon)
    - Parallel evaluation of child nodes
    - Orbit propagation
    - Entropy-based reward
    - Replay buffer recording
    
    Args:
        horizon: MCTS tree depth (default 2)
        num_steps: Number of timesteps (default 20). 
                   NOTE: Running with < 10 steps may cause video saving to fail.
        time_step: Time step duration in seconds (default 30.0)
        verbose: Print detailed planning info (default False)
        visualize: Show live animation (default True)
    """

    print("Starting Orbital Camera Simulation with Full MCTS Tree Search...")
    print(f"   Horizon: {horizon} levels")
    print(f"   Actions per level: 13")
    print(f"   Total paths: 13^{horizon} = {13**horizon}")
    print(f"   Time step: {time_step} seconds")
    print(f"   Number of steps: {num_steps}")
    if(use_torch_grid):
        print(f"Using torch and {grid_device}")
    else:
        print(f"Using cpu")

    if num_steps < 10:
        print("   WARNING: Running with num_steps < 10 may result in a non-playable video file.")
    print()

    # --- Setup simulation ---
    mu_earth = 398600.4418  # km^3/s^2
    a_chief = 7000.0        # km
    e_chief = 0.001
    i_chief = np.deg2rad(98.0)
    omega_chief = 0.0
    n_chief = np.sqrt(mu_earth / (a_chief ** 3))

    # --- Define camera ---
    camera_fn = {
        'fov_degrees': 10.0,
        'sensor_res': (64, 64),
        'noise_params': {"p_hit_given_occupied": 0.95, "p_hit_given_empty": 0.001}
    }

    # --- Initialize belief and ground truth ---
    grid = VoxelGrid(
        grid_dims=(20, 20, 20),
        use_torch=use_torch_grid,
        device=grid_device,   # None = choose "cuda" if available
    )
    rso = GroundTruthRSO(grid)
    
    # --- Create full MCTS controller ---
    controller = MCTSController(
        mu_earth, a_chief, e_chief, i_chief, omega_chief, n_chief,
        time_step=time_step,
        horizon=horizon,
        lambda_dv=lambda_dv,
        mcts_iters=mcts_iters,
        mcts_c=mcts_c,
        gamma=mcts_gamma,
        alpha_dv=alpha_dv,
        beta_tan=beta_tan,
        target_radius=target_radius,
        gamma_r=gamma_r,
        r_min_rollout=r_min_rollout,
        r_max_rollout=r_max_rollout
    )

    # --- Initialize state ---
    initial_state = np.array([0.0, 0.0, 0.0, 0.0002, 0.0, 0.0])
    state = initial_state + np.random.uniform(-0.0001, 0.0001, size=6)

    # --- Print initial RTN position for diagnostics ---
    tspan0 = np.array([0.0])
    rho_rtn0, rhodot_rtn0 = propagateGeomROE(state, a_chief, e_chief, i_chief, omega_chief, n_chief, tspan0)
    pos_rtn0 = rho_rtn0[:, 0]
    vel_rtn0 = rhodot_rtn0[:, 0]
    print(f"Initial RTN position (km): {np.round(pos_rtn0, 3)}")
    print(f"Initial RTN velocity (km/s): {np.round(vel_rtn0, 3)}")
    
    time = 0.0
    tspan = np.array([time_step])
    initial_entropy = grid.get_entropy()
    entropy_history = [initial_entropy]

    # For visualization, record camera position / direction lists
    camera_positions = []
    view_directions = []
    
    print(f"Initial State (ROEs): {np.round(state, 5)}")
    print(f"Initial Entropy: {initial_entropy:.4f}")
    print()

    start_time = tm.time()

    # --- Main simulation loop ---
    for step in range(num_steps):
        print(f"\n{'='*70}")
        print(f"Step {step+1}/{num_steps} (Time: {time:.1f}s)")
        print(f"{'='*70}")

        # 1. MCTS decision (build full tree and select best action)
        print("Building full MCTS tree ...")
        action, predicted_value, stats = controller.select_action(state, time, tspan, grid, rso, camera_fn, step, verbose=verbose, out_folder=out_folder)
        
        print(f"Best path found:")
        print(f"   First action (to execute): {np.round(action, 4)} m/s")
        print(f"   Root value: {predicted_value:.6f}")
        # print(f"   Full path length: {len(best_path)} actions")
        # if len(best_path) > 1:
        #     print(f"   Full path (first 3): {[np.round(a, 4) for a in best_path[:min(3, len(best_path))]]}")

        # 2. Execute ONLY the first action (one step of the plan)
        print(f"\nExecuting first action of optimal path...")
        # next_state_impulse is the state *immediately* after the burn
        next_state_impulse = apply_impulsive_dv(state, action, a_chief, n_chief, tspan0)        
        
        # 3. Propagate to next timestep
        rho_rtn_next, rhodot_rtn_next = propagateGeomROE(next_state_impulse, a_chief, e_chief, i_chief, omega_chief, n_chief, tspan) # Use the propagated state
        pos_next = rho_rtn_next[:, 0] * 1000  # Convert to meters
        vel_next = rhodot_rtn_next[:, 0] * 1000  # Convert to meters/s
        print(f"   New position (RTN m): {np.round(pos_next, 3)}")
        print(f"   New velocity (RTN m/s): {np.round(vel_next, 3)}")
        next_state_propagated = np.array(rtn_to_roe(rho_rtn_next[:, 0], rhodot_rtn_next[:, 0], a_chief, n_chief, tspan0))

        # Save camera position and view direction for visualization
        camera_positions.append(pos_next)
        view_directions.append(-pos_next / np.linalg.norm(pos_next))

        # 4. Take actual observation and update REAL grid
        print(f"Taking observation and updating belief...")
        entropy_before = grid.get_entropy()
        # This is the *only* place the grid's belief is permanently updated.
        simulate_observation(grid, rso, camera_fn, pos_next) 
        entropy_after = grid.get_entropy()
        entropy_history.append(entropy_after)
        
        entropy_reduction = entropy_before - entropy_after
        dv_cost = np.linalg.norm(action)
        actual_reward = entropy_reduction - controller.lambda_dv * dv_cost
        
        print(f"   Entropy: {entropy_before:.4f} → {entropy_after:.4f}")
        print(f"   Entropy reduction: {entropy_reduction:.6f}")
        print(f"   ΔV cost: {dv_cost:.6f}")
        print(f"   Reward: {actual_reward:.6f}")

        # 5. Record transition
        # We record the state *before* the action (state) and the
        # state *after* both the action and propagation (next_state_propagated)
        info_gain = entropy_reduction

        controller.record_transition(
            t=time,
            state=state,
            action=action,
            reward=actual_reward,
            next_state=next_state_propagated,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            info_gain=info_gain,
            dv_cost=dv_cost,
            step_idx=step,
            root_stats=stats,
            predicted_value=stats.get("predicted_value", None),
        )

        # 6. Move to next step
        state = next_state_propagated # Use the fully propagated state
        time += time_step

    end_time = tm.time()
    print(f"⏱ Episode runtime: {end_time - start_time:.2f} seconds")

    # --- Save results ---
    print(f"\n{'='*70}")
    print("Simulation complete! Saving results...")
    print(f"{'='*70}")
    controller.save_replay_buffer(base_dir=out_folder)

    # --- Visualization (Offline Video Generation) ---
    if visualize and len(camera_positions) > 0:
        try:
            # Convert to numpy arrays
            camera_positions = np.array(camera_positions)
            view_directions = np.array(view_directions)
            
            # Generate all frames
            frames = create_visualization_frames(
                out_folder, grid, rso, camera_fn, camera_positions, view_directions, num_steps, time_step
            )

            # Save Video
            video_path = os.path.join(out_folder, "final_visualization.mp4")
            imageio.mimsave(video_path, frames, format='MP4', fps=5, codec='libx264', macro_block_size=1)
            print(f"Saved video animation to {video_path}")
            
            # Save final PNG
            vis_png_path = os.path.join(out_folder, "final_visualization.png")
            imageio.imwrite(vis_png_path, frames[-1])
            print(f"Saved final visualization to {vis_png_path}")

        except ImportError:
            print("imageio and imageio[ffmpeg] are required for video export.")
            print("Please install with: pip install imageio imageio[ffmpeg]")
        except Exception as e:
            print(f"Visualization generation failed: {e}")

    # Save entropy progression plot and CSV
    try:
        plt.figure()
        plt.plot(np.arange(len(entropy_history)), entropy_history, marker='o')
        plt.xlabel('Step')
        plt.ylabel('Total Entropy')
        plt.title('Entropy Progression')
        png_path = os.path.join(out_folder, 'entropy_progression.png')
        plt.savefig(png_path)
        plt.close() # Close plot figure
        # Save CSV
        csv_path = os.path.join(out_folder, 'entropy_history.csv')
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['step', 'entropy'])
            for i, e in enumerate(entropy_history):
                writer.writerow([i, float(e)])
        print(f"Saved entropy progression to {png_path} and {csv_path}")
    except Exception as e:
        print(f"Failed to save entropy progression: {e}")
    
    final_entropy = grid.get_entropy()
    entropy_reduction_total = initial_entropy - final_entropy
    entropy_reduction_pct = (entropy_reduction_total / initial_entropy) * 100
    
    print(f"Final Entropy: {final_entropy:.4f} (Initial: {initial_entropy:.4f})")
    print(f"Total Entropy Reduction: {entropy_reduction_total:.4f} ({entropy_reduction_pct:.1f}%)")
    print(f"Total Transitions Recorded: {len(controller.replay_buffer)}")
    print()