"""
Orbital Camera Simulation with Full MCTS Tree Search.
Updates:
1. Burn tracking for visualization.
2. Corrected physics loop (calculate pos at time T, burn, propagate to T+dt).
"""

import numpy as np
from camera.camera_observations import VoxelGrid, GroundTruthRSO, simulate_observation, plot_scenario, update_plot
import matplotlib.pyplot as plt
import os
import json
from mcts.mcts_controller import MCTSController
from roe.propagation import propagateGeomROE, rtn_to_roe
from roe.dynamics import apply_impulsive_dv
import imageio
import csv

def create_visualization_frames(out_folder, grid_initial, rso, camera_fn, camera_positions, view_directions, burn_indices):
    """
    Creates visualization frames. 
    Accepts 'burn_indices' to mark maneuvers on the plot.
    """
    print("\n🎬 Generating visualization frames...")
    
    vis_grid = VoxelGrid(grid_dims=grid_initial.dims, voxel_size=grid_initial.voxel_size, origin=grid_initial.origin)
    
    # Calculate Plot Limits
    all_x = camera_positions[:,0]; all_y = camera_positions[:,1]; all_z = camera_positions[:,2]
    max_range = np.array([np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)]).max() / 2.0
    if max_range == 0: max_range = 15.0
    mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
    extent = max_range * 1.5 + 10.0 # Add buffer
    
    fig, ax, artists = plot_scenario(vis_grid, rso, camera_positions[0], view_directions[0], 
                                     camera_fn['fov_degrees'], camera_fn['sensor_res'])
    
    ax.set_xlim(mid_x - extent, mid_x + extent)
    ax.set_ylim(mid_y - extent, mid_y + extent)
    ax.set_zlim(mid_z - extent, mid_z + extent)
    ax.set_box_aspect([1,1,1])
    
    frames = []
    
    for frame in range(len(camera_positions)):
        current_camera_pos = camera_positions[frame]
        simulate_observation(vis_grid, rso, camera_fn, current_camera_pos) 
        
        # Pass burn_indices to update_plot
        update_plot(frame, vis_grid, rso, camera_positions, view_directions, 
                    camera_fn['fov_degrees'], camera_fn['sensor_res'], 
                    camera_fn['noise_params'], ax, artists, 
                    np.array([0.0, 0.0, 0.0]), # Target CoM
                    burn_indices=burn_indices) # <--- NEW
        
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.get_renderer().buffer_rgba())[:, :, :3])
        
    plt.close(fig)
    return frames


def run_orbital_camera_sim_full_mcts(sim_config, orbit_params, camera_params, initial_state_roe, out_folder):
    
    # Unpack Config
    horizon = sim_config.get('max_horizon', 5)
    num_steps = sim_config.get('num_steps', 20)
    time_step = sim_config.get('time_step', 30.0)
    verbose = sim_config.get('verbose', False)
    visualize = sim_config.get('visualize', True)

    print("Starting Orbital Camera Simulation...")
    print(f"   Time step: {time_step} seconds")
    print(f"   Number of steps: {num_steps}")

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
                                time_step=time_step, horizon=horizon, branching_factor=13, num_workers=None)

    state = initial_state_roe
    time = 0.0
    
    initial_entropy = grid.get_entropy()
    entropy_history = [initial_entropy]
    
    # Visualization Data
    camera_positions = []
    view_directions = []
    burn_indices = [] # Track which frame indices correspond to burns

    # Initial State Record
    # We record the state at t=0 before any loops
    rho_start, _ = propagateGeomROE(state, a_chief, e_chief, i_chief, omega_chief, n_chief, np.array([time]), t0=time)
    pos_start = rho_start[:, 0] * 1000 
    camera_positions.append(pos_start)
    view_directions.append(-pos_start / np.linalg.norm(pos_start))

    print(f"Initial State (ROEs): {np.round(state, 5)}")
    
    # --- Main simulation loop ---
    for step in range(num_steps):
        print(f"\n{'='*70}\nStep {step+1}/{num_steps} (Time: {time:.1f}s)\n{'='*70}")

        # 1. Plan Action
        action, predicted_value, _ = controller.select_action(state, time, np.array([time_step]), grid, rso, camera_params, verbose=verbose, out_folder=out_folder)
        
        print(f"Best Action: {np.round(action, 4)} m/s")

        # 2. Record Burn (Visualization)
        # If action is non-zero, mark this frame index as a burn
        if np.linalg.norm(action) > 1e-6:
            burn_indices.append(len(camera_positions) - 1) # The CURRENT position is where burn happens

        # 3. Apply Impulse (Physics)
        # Apply at current absolute time
        next_state_impulse = apply_impulsive_dv(
            state, action, a_chief, n_chief, np.array([time]), 
            e=e_chief, i=i_chief, omega=omega_chief
        )        
        
        # 4. Propagate (Natural Motion) to Next Step
        # Target time is t + dt
        t_next = time + time_step
        
        rho_rtn_next, rhodot_rtn_next = propagateGeomROE(
            next_state_impulse, 
            a_chief, e_chief, i_chief, omega_chief, n_chief, 
            np.array([t_next]), 
            t0=time # Propagation starts from 'time'
        )
        
        # 5. Map Back to ROE (State Update)
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
        
        print(f"Entropy: {entropy_before:.4f} -> {entropy_after:.4f}")
        
        controller.record_transition(time, state, action, actual_reward, next_state_propagated)

        # 9. Advance Clock
        state = next_state_propagated 
        time += time_step

    # --- Save results ---
    controller.save_replay_buffer(base_dir=out_folder)

    # --- Visualization ---
    if visualize and len(camera_positions) > 0:
        try:
            frames = create_visualization_frames(
                out_folder, grid, rso, camera_params, 
                np.array(camera_positions), np.array(view_directions), 
                burn_indices # Pass burn indices here
            )
            video_path = os.path.join(out_folder, "final_visualization.mp4")
            imageio.mimsave(video_path, frames, format='MP4', fps=5, codec='libx264')
            print(f"Saved video: {video_path}")
            
            imageio.imwrite(os.path.join(out_folder, "final_frame.png"), frames[-1])
        except Exception as e:
            print(f"Visualization failed: {e}")

    # Plot Entropy
    plt.figure()
    plt.plot(entropy_history, marker='o')
    plt.savefig(os.path.join(out_folder, 'entropy_progression.png'))
    plt.close()