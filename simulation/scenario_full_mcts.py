"""
Orbital Camera Simulation with Full MCTS Tree Search.
"""

import matplotlib
matplotlib.use('Agg') 

import numpy as np
from camera.camera_observations import VoxelGrid, GroundTruthRSO, simulate_observation, plot_scenario, update_plot
import matplotlib.pyplot as plt
import os
import json
import time
from mcts.mcts_controller import MCTSController
from mcts.mcts_controller_gpu import MCTSControllerGPU
from roe.propagation import propagateGeomROE, rtn_to_roe
from roe.dynamics import apply_impulsive_dv
import imageio
import csv


class EpisodeEntropyNormalizer:
    """
    Normalize rewards by the initial entropy of the episode.

    This ensures that rewards are relative to how uncertain you started,
    making the learning signal more consistent across different scenarios.
    """
    def __init__(self):
        self.initial_entropy = None

    def start_episode(self, initial_entropy):
        """Call at the beginning of each episode with the initial entropy."""
        self.initial_entropy = initial_entropy

    def normalize(self, reward):
        """Normalize reward by initial entropy."""
        if self.initial_entropy is None or self.initial_entropy == 0:
            return reward
        return reward / self.initial_entropy


def create_visualization_frames(out_folder, grid_initial, rso, camera_fn, camera_positions, view_directions, burn_indices):
    print("\n🎬 Generating visualization frames...")
    vis_grid = VoxelGrid(grid_dims=grid_initial.dims, voxel_size=grid_initial.voxel_size, origin=grid_initial.origin)
    
    all_x = camera_positions[:,0]; all_y = camera_positions[:,1]; all_z = camera_positions[:,2]
    all_x = np.concatenate([all_x, [vis_grid.origin[0], vis_grid.max_bound[0]]])
    all_y = np.concatenate([all_y, [vis_grid.origin[1], vis_grid.max_bound[1]]])
    all_z = np.concatenate([all_z, [vis_grid.origin[2], vis_grid.max_bound[2]]])

    max_range = np.array([np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)]).max() / 2.0
    if max_range == 0: max_range = 15.0
    mid_x, mid_y, mid_z = (np.max(all_x)+np.min(all_x))*0.5, (np.max(all_y)+np.min(all_y))*0.5, (np.max(all_z)+np.min(all_z))*0.5
    extent = max_range * 1.1
    
    fig, ax, artists = plot_scenario(vis_grid, rso, camera_positions[0], view_directions[0], camera_fn['fov_degrees'], camera_fn['sensor_res'])
    ax.set_xlim(mid_x - extent, mid_x + extent)
    ax.set_ylim(mid_y - extent, mid_y + extent)
    ax.set_zlim(mid_z - extent, mid_z + extent)
    ax.set_box_aspect([1,1,1])
    
    frames = []
    for frame in range(len(camera_positions)):
        simulate_observation(vis_grid, rso, camera_fn, camera_positions[frame]) 
        # Removed plot_extent argument
        update_plot(frame, vis_grid, rso, camera_positions, view_directions, camera_fn['fov_degrees'], camera_fn['sensor_res'], camera_fn['noise_params'], ax, artists, np.array([0.0, 0.0, 0.0]), burn_indices)
        fig.canvas.draw()
        
        try: buf = fig.canvas.buffer_rgba()
        except: buf = fig.canvas.renderer.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype=np.uint8).copy().reshape((h, w, 4))[:, :, :3]
        frames.append(image)
        
    plt.close(fig)
    return frames

def run_orbital_camera_sim_full_mcts(sim_config, orbit_params, camera_params, control_params, initial_state_roe, out_folder):
    horizon = sim_config.get('max_horizon', 5)
    num_steps = sim_config.get('num_steps', 20)
    time_step = sim_config.get('time_step', 30.0)
    verbose = sim_config.get('verbose', False)
    visualize = sim_config.get('visualize', True)
    use_torch_grid = sim_config.get('use_torch_grid', False)
    grid_device = sim_config.get('grid_device', 'cpu')
    alpha_dv = sim_config.get('alpha_dv', 1.0)
    beta_tan = sim_config.get('beta_tan', 1.0)
    rollout_policy = sim_config.get('desired_rollout_policy', "random")
    lambda_dv = sim_config.get('lambda_dv', 1.0)
    mcts_iters = sim_config.get('mcts_iters', 3000)
    mcts_c = sim_config.get('mcts_c', 1.4)
    gamma = sim_config.get('gamma', 0.99)
    # Use parallel CPU MCTS by default (6.15x speedup)
    num_processes = sim_config.get('num_processes', None)

    print("Starting Orbital Camera Simulation...")
    print(f"   Time step: {time_step} seconds")
    print(f"   Number of steps: {num_steps}")
    print(f"Using Parallel CPU MCTS (6.15x speedup)")
    print(f"  - Processes: {num_processes or 'auto (cpu_count - 1)'}")

    mu_earth = orbit_params['mu_earth']
    a_chief = orbit_params['a_chief_km']
    e_chief = orbit_params['e_chief']
    i_chief = np.deg2rad(orbit_params['i_chief_deg'])
    omega_chief = np.deg2rad(orbit_params['omega_chief_deg'])
    n_chief = np.sqrt(mu_earth / (a_chief ** 3))

    grid = VoxelGrid(grid_dims=(20, 20, 20))
    rso = GroundTruthRSO(grid)

    # Create parallel CPU MCTS controller
    print("\n💻 Creating Parallel CPU MCTS controller...")
    controller = MCTSController(
        mu_earth, a_chief, e_chief, i_chief, omega_chief, n_chief,
        time_step=time_step, horizon=horizon, alpha_dv=alpha_dv, beta_tan=beta_tan,
        rollout_policy=rollout_policy, lambda_dv=lambda_dv, branching_factor=13,
        num_workers=None, mcts_iters=mcts_iters, mcts_c=mcts_c, gamma=gamma,
        use_parallel_mcts=True,
        num_processes=num_processes,
        verbose=verbose
    )

    state = initial_state_roe
    time_sim = 0.0

    initial_entropy = grid.get_entropy()
    entropy_history = [initial_entropy]
    entropy_reductions = []  # Track entropy reductions for early stopping

    # Initialize reward normalizer for this episode
    reward_normalizer = EpisodeEntropyNormalizer()
    reward_normalizer.start_episode(initial_entropy)

    rho_start, _ = propagateGeomROE(state, a_chief, e_chief, i_chief, omega_chief, n_chief, np.array([time_sim]), t0=time_sim)
    pos_start = rho_start[:, 0] * 1000
    camera_positions = [pos_start]
    view_directions = [-pos_start / np.linalg.norm(pos_start)]
    burn_indices = []

    print(f"Using torch and cuda")
    print(f"Initial entropy: {initial_entropy:.4f}")
    roe_str = np.array2string(state, formatter={'float_kind':lambda x: "%.1e" % x})
    print(f"Initial State (ROEs): {roe_str}")

    start_time = time.time()
    early_stopped = False

    for step in range(num_steps):
        print(f"\n{'='*70}")
        print(f"Step {step+1}/{num_steps} (Time: {time_sim:.1f}s)")
        print(f"{'='*70}")

        action, predicted_value, stats = controller.select_action(state, time_sim, np.array([time_step]), grid, rso, camera_params, verbose=verbose, out_folder=out_folder)
        
        action_str = np.array2string(action, formatter={'float_kind':lambda x: "%.2f" % x})
        print(f"Best Action: {action_str} m/s")

        if np.linalg.norm(action) > 1e-6:
            burn_indices.append(len(camera_positions) - 1)

        t_burn = np.array([time_sim])
        next_state_impulse = apply_impulsive_dv(state, action, a_chief, n_chief, t_burn, e=e_chief, i=i_chief, omega=omega_chief)        
        
        t_next = time_sim + time_step
        rho_rtn_next, rhodot_rtn_next = propagateGeomROE(next_state_impulse, a_chief, e_chief, i_chief, omega_chief, n_chief, np.array([t_next]), t0=time_sim)
        
        next_state_propagated = np.array(rtn_to_roe(rho_rtn_next[:, 0], rhodot_rtn_next[:, 0], a_chief, n_chief, np.array([t_next])))

        pos_next = rho_rtn_next[:, 0] * 1000 
        camera_positions.append(pos_next)
        view_directions.append(-pos_next / np.linalg.norm(pos_next))
        
        entropy_before = grid.get_entropy()
        simulate_observation(grid, rso, camera_params, pos_next) 
        entropy_after = grid.get_entropy()
        entropy_history.append(entropy_after)
        
        entropy_reduction = entropy_before - entropy_after
        entropy_reductions.append(entropy_reduction)
        dv_cost = np.linalg.norm(action)
        actual_reward = entropy_reduction - controller.lambda_dv * dv_cost

        # Normalize reward by initial entropy
        normalized_reward = reward_normalizer.normalize(actual_reward)

        print(f"   Entropy: {entropy_before:.4f} → {entropy_after:.4f}")
        print(f"   Entropy reduction: {entropy_reduction:.6f}")
        print(f"   ΔV cost: {dv_cost:.6f}")
        print(f"   Reward: {actual_reward:.6f}")
        print(f"   Normalized reward: {normalized_reward:.6f}")

        controller.record_transition(time_sim, state, action, normalized_reward, next_state_propagated,
                                     entropy_before=entropy_before, entropy_after=entropy_after,
                                     info_gain=entropy_reduction, dv_cost=dv_cost,
                                     step_idx=step, root_stats=stats, predicted_value=predicted_value)

        state = next_state_propagated
        time_sim += time_step

        # Early stopping: check after 30 steps if entropy reduction is stagnating
        if step >= 29 and len(entropy_reductions) >= 30:  # At least 30 steps completed
            # Calculate average reduction over last 3-5 steps (or all available if less than 5)
            window_size = min(5, len(entropy_reductions))
            recent_reductions = entropy_reductions[-window_size:]
            avg_reduction = np.mean(recent_reductions)

            # Check if average reduction is less than 1% of initial entropy per step
            early_stop_threshold = 0.01 * initial_entropy

            if avg_reduction < early_stop_threshold:
                print(f"\n{'='*70}")
                print(f"⏹ EARLY STOPPING TRIGGERED")
                print(f"{'='*70}")
                print(f"Step {step+1}: Average entropy reduction over last {window_size} steps: {avg_reduction:.6f}")
                print(f"Threshold (1% of initial entropy): {early_stop_threshold:.6f}")
                print(f"Entropy has stagnated - skipping remaining {num_steps - step - 1} steps")
                print(f"{'='*70}")
                early_stopped = True
                break

    end_time = time.time()
    print(f"⏱ Episode runtime: {end_time - start_time:.2f} seconds")
    controller.save_replay_buffer(base_dir=out_folder)

    # Save early stopping metadata
    metadata = {
        "early_stopped": early_stopped,
        "total_steps_completed": len(entropy_reductions),
        "num_steps_requested": num_steps,
    }
    if early_stopped and len(entropy_reductions) >= 10:
        window_size = min(5, len(entropy_reductions))
        recent_reductions = entropy_reductions[-window_size:]
        metadata["final_avg_reduction"] = float(np.mean(recent_reductions))
        metadata["early_stop_threshold"] = float(0.01 * initial_entropy)

    metadata_path = os.path.join(out_folder, "early_stopping_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if visualize and len(camera_positions) > 0:
        try:
            frames = create_visualization_frames(
                out_folder, grid, rso, camera_params, 
                np.array(camera_positions), np.array(view_directions), 
                burn_indices 
            )
            video_path = os.path.join(out_folder, "final_visualization.mp4")
            imageio.mimsave(video_path, frames, format='MP4', fps=5, codec='libx264', macro_block_size=1)
            print(f"Saved video: {video_path}")
            imageio.imwrite(os.path.join(out_folder, "final_frame.png"), frames[-1])
        except Exception as e:
            print(f"Visualization failed: {e}")

    plt.figure()
    plt.plot(entropy_history, marker='o')
    plt.savefig(os.path.join(out_folder, 'entropy_progression.png'))
    plt.close()