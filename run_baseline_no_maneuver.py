import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

# Import project modules
from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from camera.camera_observations import VoxelGrid, GroundTruthRSO, plot_scenario, update_plot, simulate_observation
from roe.propagation import propagateGeomROE

def load_config(path="config.json"):
    if not os.path.exists(path):
        # Create a dummy config if missing
        return {
            'orbit': {
                'mu_earth': 398600.4418, 'a_chief_km': 7000.0, 'e_chief': 0.001, 
                'i_chief_deg': 98.0, 'omega_chief_deg': 30.0
            },
            'camera': {
                'fov_degrees': 15.0, 'sensor_res': [64, 64],
                'noise_params': {'p_hit_given_occupied': 0.95, 'p_hit_given_empty': 0.001}
            },
            'initial_roe_meters': {
                'da': 0.0, 'dl': 200.0, 'dex': 100.0, 'dey': 0.0, 'dix': 50.0, 'diy': 0.0
            }
        }
    with open(path, "r") as f:
        return json.load(f)

def create_visualization_frames(out_folder, grid_initial, rso, camera_fn, camera_positions, view_directions):
    print("Generating video frames...")
    # Create a fresh grid for visualization so we can replay the belief updates
    vis_grid = VoxelGrid(grid_dims=grid_initial.dims, voxel_size=grid_initial.voxel_size, origin=grid_initial.origin)
    
    # Setup plot
    fig, ax, artists = plot_scenario(vis_grid, rso, camera_positions[0], view_directions[0], 
                                     camera_fn['fov_degrees'], camera_fn['sensor_res'])
    
    # Fix limits to keep the view stable
    all_pos = np.vstack([camera_positions, [vis_grid.origin], [vis_grid.max_bound]])
    mid = np.mean(all_pos, axis=0)
    max_range = np.max(np.ptp(all_pos, axis=0)) / 2.0
    extent = max_range * 1.2 + 10.0
    
    ax.set_xlim(mid[0]-extent, mid[0]+extent)
    ax.set_ylim(mid[1]-extent, mid[1]+extent)
    ax.set_zlim(mid[2]-extent, mid[2]+extent)
    ax.set_box_aspect([1,1,1])

    frames = []
    burn_indices = [] # No burns in baseline
    
    # Render loop
    for frame in range(len(camera_positions)):
        # Re-simulate observation for visualization
        simulate_observation(vis_grid, rso, camera_fn, camera_positions[frame])
        
        # Update artists
        update_plot(frame, vis_grid, rso, camera_positions, view_directions, 
                    camera_fn['fov_degrees'], camera_fn['sensor_res'], 
                    camera_fn['noise_params'], ax, artists, np.array([0.0, 0.0, 0.0]), 
                    burn_indices)
        
        # Capture frame
        fig.canvas.draw()
        try: 
            buf = fig.canvas.buffer_rgba()
        except: 
            buf = fig.canvas.renderer.buffer_rgba()
            
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).copy().reshape((h, w, 4))[:, :, :3]
        frames.append(img)
        
        if frame % 10 == 0:
            print(f"Rendered frame {frame}/{len(camera_positions)}")
            
    plt.close(fig)
    return frames

def run_baseline():
    print("--- Starting BASELINE Simulation (No Maneuvers) ---")
    
    # 1. Configuration (Manually Override for Visibility)
    config = load_config()
    op = config['orbit']
    cp = config['camera']
    
    # OVERRIDE: Use longer steps to see the ellipse clearly
    TIME_STEP = 120.0   # 2 minutes
    NUM_STEPS = 50      # 100 minutes total (approx 1 full orbit)
    
    print(f"Time Step: {TIME_STEP} s")
    print(f"Duration:  {NUM_STEPS} steps ({NUM_STEPS*TIME_STEP/60:.1f} mins)")

    # 2. Setup Environment
    grid = VoxelGrid(grid_dims=(20, 20, 20))
    rso = GroundTruthRSO(grid)
    
    # Initialize MDP
    mdp = OrbitalMCTSModel(
        a_chief=op['a_chief_km'], 
        e_chief=op['e_chief'], 
        i_chief=np.deg2rad(op['i_chief_deg']), 
        omega_chief=np.deg2rad(op['omega_chief_deg']), 
        n_chief=np.sqrt(op['mu_earth']/op['a_chief_km']**3),
        rso=rso, 
        camera_fn=cp, 
        grid_dims=grid.dims, 
        lambda_dv=0.0,
        time_step=TIME_STEP, 
        max_depth=5
    )

    # 3. Initial State
    rm = config['initial_roe_meters']
    am = op['a_chief_km'] * 1000.0
    base_roe = np.array([rm['da'], rm['dl'], rm['dex'], rm['dey'], rm['dix'], rm['diy']], dtype=float) / am
    
    state = OrbitalState(roe=base_roe, grid=grid, time=0.0)
    
    # 4. Storage for Plotting/Video
    entropy_history = [grid.get_entropy()]
    trajectory_points = []
    
    # Data for animation
    camera_positions = []
    view_directions = []
    
    # Get initial position
    from roe.propagation import map_roe_to_rtn
    r_start, _ = map_roe_to_rtn(base_roe, mdp.a_chief, mdp.n_chief, f=0.0, omega=mdp.omega_chief)
    pos_start = r_start * 1000.0
    
    # --- CRITICAL FIX: Initial Observation at t=0 ---
    simulate_observation(grid, rso, cp, pos_start)
    # Update entropy history with the entropy AFTER the first look
    entropy_history[0] = grid.get_entropy()
    # ------------------------------------------------
    
    trajectory_points.append(pos_start)
    camera_positions.append(pos_start)
    view_directions.append(-pos_start / np.linalg.norm(pos_start))

    print(f"Initial Entropy: {entropy_history[0]:.4f}")

    # 5. Simulation Loop
    for step in range(NUM_STEPS):
        # FORCE NO ACTION
        action = np.array([0.0, 0.0, 0.0])
        
        # Step Physics
        next_state, reward = mdp.step(state, action)
        
        # Record Data
        ent = next_state.grid.get_entropy()
        entropy_history.append(ent)
        
        # Get Position for Plotting (Recalculate strictly for viz)
        t_next = next_state.time
        f_next = mdp.n_chief * t_next
        r_next, _ = map_roe_to_rtn(next_state.roe, mdp.a_chief, mdp.n_chief, f=f_next, omega=mdp.omega_chief)
        pos_meters = r_next * 1000.0
        
        trajectory_points.append(pos_meters)
        camera_positions.append(pos_meters)
        view_directions.append(-pos_meters / np.linalg.norm(pos_meters))
        
        print(f"Step {step+1:02d}: Time={t_next:.1f}s | Entropy={ent:.4f}")
        
        state = next_state

    # 6. Visualization Results
    output_dir = "output_baseline"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Entropy
    plt.figure(figsize=(10, 5))
    plt.plot(entropy_history, marker='o', label='Passive Entropy')
    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.title("Baseline: Passive Information Gain (Drift Only)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "baseline_entropy.png"))
    plt.close()

    # Plot 2: 3D Trajectory
    traj = np.array(trajectory_points)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Path
    ax.plot(traj[:,0], traj[:,1], traj[:,2], label='Passive Trajectory', color='blue', linewidth=2)
    ax.scatter(traj[0,0], traj[0,1], traj[0,2], color='green', marker='o', s=50, label='Start')
    ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], color='red', marker='x', s=50, label='End')
    ax.scatter(0, 0, 0, color='black', marker='*', s=100, label='Target (0,0,0)')
    
    # Equal Aspect Ratio Hack
    max_range = np.array([traj[:,0].max()-traj[:,0].min(), 
                          traj[:,1].max()-traj[:,1].min(), 
                          traj[:,2].max()-traj[:,2].min()]).max() / 2.0
    mid_x = (traj[:,0].max()+traj[:,0].min()) * 0.5
    mid_y = (traj[:,1].max()+traj[:,1].min()) * 0.5
    mid_z = (traj[:,2].max()+traj[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('Radial (m)')
    ax.set_ylabel('Along-Track (m)')
    ax.set_zlabel('Cross-Track (m)')
    ax.set_title(f"Baseline Trajectory ({NUM_STEPS} steps x {TIME_STEP}s)")
    ax.legend()
    plt.savefig(os.path.join(output_dir, "baseline_trajectory.png"))
    plt.close(fig)

    # 3. Generate Animation
    frames = create_visualization_frames(
        output_dir, grid, rso, cp,
        np.array(camera_positions), np.array(view_directions)
    )
    
    if frames:
        video_path = os.path.join(output_dir, "baseline_video.mp4")
        imageio.mimsave(video_path, frames, fps=5, macro_block_size=1)
        imageio.imwrite(os.path.join(output_dir, "baseline_final_frame.png"), frames[-1])
        print(f"Video saved to: {video_path}")
    
    print(f"\nAll baseline artifacts saved to: {output_dir}/")

if __name__ == "__main__":
    run_baseline()