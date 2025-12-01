"""
Orbital Camera Simulation with Full MCTS Tree Search.
Visualization loop updated to fix Video Buffer Overwrite issue.
"""

import matplotlib
# Force non-interactive backend to prevent display errors
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import imageio
from camera.camera_observations import VoxelGrid, GroundTruthRSO, calculate_entropy, simulate_observation, plot_scenario, update_plot
from mcts.mcts_controller import MCTSController
from roe.propagation import propagateGeomROE, rtn_to_roe
from roe.dynamics import apply_impulsive_dv

def create_visualization_frames(out_folder, grid_initial, rso, camera_fn, camera_positions, view_directions, burn_indices):
    """
    Creates visualization frames with growing path and dynamic scaling.
    """
    print("\n🎬 Generating visualization frames...")
    
    vis_grid = VoxelGrid(grid_dims=grid_initial.dims, voxel_size=grid_initial.voxel_size, origin=grid_initial.origin)
    
    # Calculate Plot Limits
    all_x = camera_positions[:,0]; all_y = camera_positions[:,1]; all_z = camera_positions[:,2]
    
    # Add grid bounds to ensure plot isn't too zoomed in
    all_x = np.concatenate([all_x, [vis_grid.origin[0], vis_grid.max_bound[0]]])
    all_y = np.concatenate([all_y, [vis_grid.origin[1], vis_grid.max_bound[1]]])
    all_z = np.concatenate([all_z, [vis_grid.origin[2], vis_grid.max_bound[2]]])

    max_range = np.array([np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)]).max() / 2.0
    if max_range == 0: max_range = 15.0
    
    mid_x = (np.max(all_x) + np.min(all_x)) * 0.5
    mid_y = (np.max(all_y) + np.min(all_y)) * 0.5
    mid_z = (np.max(all_z) + np.min(all_z)) * 0.5
    
    # Extent defines the zoom level
    extent = max_range * 1.1
    
    # Initialize Plot with Start Position
    fig, ax, artists = plot_scenario(vis_grid, rso, camera_positions[0], view_directions[0], 
                                     camera_fn['fov_degrees'], camera_fn['sensor_res'])
    
    ax.set_xlim(mid_x - extent, mid_x + extent)
    ax.set_ylim(mid_y - extent, mid_y + extent)
    ax.set_zlim(mid_z - extent, mid_z + extent)
    ax.set_box_aspect([1,1,1])
    
    frames = []
    
    for frame in range(len(camera_positions)):
        # Simulate observation for visualization (accumulates belief on vis_grid)
        current_camera_pos = camera_positions[frame]
        simulate_observation(vis_grid, rso, camera_fn, current_camera_pos) 
        
        # Update plot artists
        update_plot(frame, vis_grid, rso, camera_positions, view_directions, 
                    camera_fn['fov_degrees'], camera_fn['sensor_res'], 
                    camera_fn['noise_params'], ax, artists, 
                    np.array([0.0, 0.0, 0.0]), 
                    burn_indices=burn_indices)
        
        # Draw canvas
        fig.canvas.draw()
        
        # Extract image buffer
        try:
            buf = fig.canvas.buffer_rgba()
        except AttributeError:
            buf = fig.canvas.renderer.buffer_rgba()
            
        w, h = fig.canvas.get_width_height()
        
        # Convert buffer to numpy array
        # CRITICAL FIX: .copy() ensures we store the PIXELS, not a pointer to the buffer
        image = np.frombuffer(buf, dtype=np.uint8).copy()
        
        # Reshape and slice off alpha channel
        image = image.reshape((h, w, 4))[:, :, :3]
        
        frames.append(image)
        
    plt.close(fig)
    return frames


def run_orbital_camera_sim_full_mcts(sim_config, orbit_params, camera_params, initial_state_roe, out_folder):

    # Unpack Config
    horizon = sim_config.get('max_horizon', 5)
    num_steps = sim_config.get('num_steps', 20)
    time_step = sim_config.get('time_step', 30.0)
    verbose = sim_config.get('verbose', False)
    visualize = sim_config.get('visualize', True)
    use_torch_grid = sim_config.get('use_torch_grid', False)
    grid_device = sim_config.get('grid_device', 'cpu')
    alpha_dv = sim_config.get('alpha_dv', 1.0)
    beta_tan = sim_config.get('beta_tan', 1.0)
    target_radius = sim_config.get('target_radius', 50.0)
    gamma_r = sim_config.get('gamma_r', 0.99)
    r_min_rollout = sim_config.get('r_min_rollout', 0.0)
    r_max_rollout = sim_config.get('r_max_rollout', 100.0)
    lambda_dv = sim_config.get('lambda_dv', 1.0)
    mcts_iters = sim_config.get('mcts_iters', 3000)
    mcts_c = sim_config.get('mcts_c', 1.4)
    gamma = sim_config.get('gamma', 0.99)

    print("Starting Orbital Camera Simulation...")
    print(f"   Time step: {time_step} seconds")
    print(f"   Number of steps: {num_steps}")
    if use_torch_grid:
        print(f"Using torch and {grid_device}")
    else:
        print(f"Using cpu")

    start_time = time.time()

    # Orbit & Dynamics Setup
    mu_earth = orbit_params['mu_earth']
    a_chief = orbit_params['a_chief_km']
    e_chief = orbit_params['e_chief']
    i_chief = np.deg2rad(orbit_params['i_chief_deg'])
    omega_chief = np.deg2rad(orbit_params['omega_chief_deg'])
    n_chief = np.sqrt(mu_earth / (a_chief ** 3))

    grid = VoxelGrid(grid_dims=(20, 20, 20))
    rso = GroundTruthRSO(grid)
    
    controller = MCTSController(mu_earth, a_chief, e_chief, i_chief, omega_chief, n_chief,
                                time_step=time_step, horizon=horizon, alpha_dv=alpha_dv, beta_tan=beta_tan,
                                target_radius=target_radius, gamma_r=gamma_r, r_min_rollout=r_min_rollout,
                                r_max_rollout=r_max_rollout, lambda_dv=lambda_dv, branching_factor=13,
                                num_workers=None, mcts_iters=mcts_iters, mcts_c=mcts_c, gamma=gamma)

    state = initial_state_roe
    t = 0.0

    initial_entropy = grid.get_entropy()
    entropy_history = [initial_entropy]

    # Visualization Data
    # Initialize with t=0 state
    rho_start, _ = propagateGeomROE(state, a_chief, e_chief, i_chief, omega_chief, n_chief, np.array([t]), t0=t)
    pos_start = rho_start[:, 0] * 1000 
    
    camera_positions = [pos_start]
    view_directions = [-pos_start / np.linalg.norm(pos_start)]
    burn_indices = [] 

    print(f"Initial State (ROEs): {np.round(state, 5)}")
    
    # --- Main simulation loop ---
    for step in range(num_steps):
        print(f"\n{'='*70}\nStep {step+1}/{num_steps} (Time: {t:.1f}s)\n{'='*70}")

        # 1. Plan Action
        action, predicted_value, stats = controller.select_action(state, t, np.array([time_step]), grid, rso, camera_params, verbose=verbose, out_folder=out_folder)
        print(f"Best Action: {np.round(action, 4)} m/s")

        # 2. Record Burn
        if np.linalg.norm(action) > 1e-6:
            burn_indices.append(len(camera_positions) - 1)

        # 3. Apply Impulse
        next_state_impulse = apply_impulsive_dv(
            state, action, a_chief, n_chief, np.array([t]),
            e=e_chief, i=i_chief, omega=omega_chief
        )

        # 4. Propagate
        t_next = t + time_step
        rho_rtn_next, rhodot_rtn_next = propagateGeomROE(
            next_state_impulse, a_chief, e_chief, i_chief, omega_chief, n_chief,
            np.array([t_next]), t0=t
        )

        # 5. Map Back to ROE
        next_state_propagated = np.array(rtn_to_roe(
            rho_rtn_next[:, 0], rhodot_rtn_next[:, 0],
            a_chief, n_chief, np.array([t_next])
        ))

        # 6. Record New Position
        pos_next = rho_rtn_next[:, 0] * 1000
        camera_positions.append(pos_next)
        view_directions.append(-pos_next / np.linalg.norm(pos_next))

        # 7. Observation
        entropy_before = grid.get_entropy()
        simulate_observation(grid, rso, camera_params, pos_next)
        entropy_after = grid.get_entropy()
        entropy_history.append(entropy_after)

        # 8. Logging
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
            t=t,
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

        # 9. Advance Clock
        state = next_state_propagated
        t += time_step

    end_time = time.time()
    print(f"⏱ Episode runtime: {end_time - start_time:.2f} seconds")

    # --- Save results ---
    controller.save_replay_buffer(base_dir=out_folder)

    # --- Visualization ---
    if visualize and len(camera_positions) > 0:
        # Convert lists to numpy arrays for processing
        cam_pos_arr = np.array(camera_positions)
        view_dir_arr = np.array(view_directions)
        
        frames = create_visualization_frames(
            out_folder, grid, rso, camera_params, 
            cam_pos_arr, view_dir_arr, 
            burn_indices
        )
        
        if len(frames) > 0:
            print(f"Captured {len(frames)} frames. Frame shape: {frames[0].shape}")
            video_path = os.path.join(out_folder, "final_visualization.mp4")
            try:
                imageio.mimsave(video_path, frames, format='MP4', fps=5, codec='libx264', macro_block_size=1)
                print(f"Saved video: {video_path}")
            except Exception as e:
                print(f"Failed to save MP4: {e}. Attempting GIF fallback...")
                gif_path = os.path.join(out_folder, "final_visualization.gif")
                imageio.mimsave(gif_path, frames, format='GIF', fps=5)
                print(f"Saved GIF: {gif_path}")
            
            imageio.imwrite(os.path.join(out_folder, "final_frame.png"), frames[-1])
        else:
            print("No frames captured. Skipping video generation.")

    # Plot Entropy
    plt.figure()
    plt.plot(entropy_history, marker='o')
    plt.savefig(os.path.join(out_folder, 'entropy_progression.png'))
    plt.close()