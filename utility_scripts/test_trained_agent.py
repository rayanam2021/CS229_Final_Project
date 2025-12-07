import os
import torch
import json
import numpy as np
import pandas as pd
import imageio
import matplotlib
matplotlib.use('Agg') # Safe for headless/parallel runs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import project modules
from learning.policy_value_network import PolicyValueNetwork
from mcts.mcts_alphazero_controller import MCTSAlphaZeroCPU
from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from camera.camera_observations import VoxelGrid, GroundTruthRSO, simulate_observation, plot_scenario, update_plot
from roe.propagation import map_roe_to_rtn

def load_config(path="config.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

def create_visualization_frames(out_folder, grid_initial, rso, camera_fn, camera_positions, view_directions, burn_indices):
    print("Generating video frames...")
    vis_grid = VoxelGrid(grid_dims=grid_initial.dims, voxel_size=grid_initial.voxel_size, origin=grid_initial.origin)
    all_pos = np.vstack([camera_positions, [vis_grid.origin], [vis_grid.max_bound]])
    mid = np.mean(all_pos, axis=0)
    max_range = np.max(np.ptp(all_pos, axis=0)) / 2.0
    extent = max_range * 1.2 + 10.0

    fig, ax, artists = plot_scenario(vis_grid, rso, camera_positions[0], view_directions[0], 
                                     camera_fn['fov_degrees'], camera_fn['sensor_res'])
    ax.set_xlim(mid[0]-extent, mid[0]+extent)
    ax.set_ylim(mid[1]-extent, mid[1]+extent)
    ax.set_zlim(mid[2]-extent, mid[2]+extent)
    ax.set_box_aspect([1,1,1])

    frames = []
    for frame in range(len(camera_positions)):
        simulate_observation(vis_grid, rso, camera_fn, camera_positions[frame])
        update_plot(frame, vis_grid, rso, camera_positions, view_directions, 
                    camera_fn['fov_degrees'], camera_fn['sensor_res'], 
                    camera_fn['noise_params'], ax, artists, np.array([0.0, 0.0, 0.0]), 
                    burn_indices)
        fig.canvas.draw()
        try: buf = fig.canvas.buffer_rgba()
        except: buf = fig.canvas.renderer.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).copy().reshape((h, w, 4))[:, :, :3]
        frames.append(img)
        
        if frame % 10 == 0:
            print(f"Rendered frame {frame}/{len(camera_positions)}")
            
    plt.close(fig)
    return frames

def run_simulation(config, checkpoint_path, output_dir):
    print(f"--- Loading Agent from {checkpoint_path} ---")
    
    # 1. Setup Environment
    op = config['orbit']
    cp = config['camera']
    ctrl_lambda = config.get('control', {}).get('lambda_dv', 0.01)
    
    # Initialize fresh grid and RSO
    grid_dims = (20, 20, 20)
    grid = VoxelGrid(grid_dims=grid_dims)
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
        lambda_dv=ctrl_lambda,
        time_step=config['simulation']['time_step'], 
        max_depth=config['simulation']['max_horizon']
    )

    # 2. Load Neural Network
    network = PolicyValueNetwork(grid_dims=grid.dims, num_actions=13, hidden_dim=128)
    
    if not os.path.exists(checkpoint_path):
        print(f"CRITICAL ERROR: Checkpoint not found at {checkpoint_path}")
        return

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    network.load_state_dict(ckpt['network_state'])
    network.eval()
    print(f"Network loaded (Epoch {ckpt.get('epoch', 'Unknown')})")

    # 3. Configure MCTS Agent
    mcts_agent = MCTSAlphaZeroCPU(
        model=mdp, 
        network=network, 
        c_puct=1.4, 
        n_iters=50,  
        gamma=0.99
    )

    # 4. Set Initial State (UNPERTURBED)
    rm = config['initial_roe_meters']
    am = op['a_chief_km'] * 1000.0
    base_roe = np.array([rm['da'], rm['dl'], rm['dex'], rm['dey'], rm['dix'], rm['diy']], dtype=float) / am
    
    # --- CRITICAL: Initial Observation at t=0 ---
    r_init, _ = map_roe_to_rtn(base_roe, mdp.a_chief, mdp.n_chief, f=0.0, omega=mdp.omega_chief)
    pos_init = r_init * 1000.0
    simulate_observation(grid, rso, cp, pos_init)
    
    initial_ent = grid.get_entropy()
    mdp.initial_entropy = initial_ent  
    
    state = OrbitalState(roe=base_roe, grid=grid, time=0.0)
    
    print(f"Initial ROE (m): {np.array2string(base_roe * am, precision=1, separator=', ')}")
    print(f"Initial Entropy: {initial_ent:.4f}")
    
    # 5. Run Simulation Loop
    steps = config['simulation']['num_steps']
    
    # Data storage for artifacts
    entropy_history = [initial_ent]
    trajectory_data = [] # List of dicts for CSV
    camera_positions = [pos_init]
    view_directions = [-pos_init/np.linalg.norm(pos_init)]
    burn_indices = []
    
    print(f"Starting Simulation ({steps} steps)...")

    for step in range(steps):
        # Run MCTS
        pi, value, _ = mcts_agent.search(state)
        
        # Select best action
        best_idx = np.argmax(pi)
        action = mdp.get_all_actions()[best_idx]
        
        # Record burn
        if np.linalg.norm(action) > 1e-6:
            burn_indices.append(len(camera_positions) - 1)

        # Apply Action
        next_state, reward = mdp.step(state, action)
        
        # --- Visualization Data Collection ---
        # Calculate Position for plotting (MDP calculates it internally but doesn't return it)
        t_next = next_state.time
        f_next = mdp.n_chief * t_next
        r_next, _ = map_roe_to_rtn(next_state.roe, mdp.a_chief, mdp.n_chief, f=f_next, omega=mdp.omega_chief)
        pos_next = r_next * 1000.0
        
        camera_positions.append(pos_next)
        view_directions.append(-pos_next/np.linalg.norm(pos_next))
        # -------------------------------------

        # Metrics
        ent = next_state.grid.get_entropy()
        entropy_history.append(ent)
        
        # Log to Console
        act_str = np.array2string(action, precision=2, separator=', ', suppress_small=True)
        print(f"Step {step+1}: Act={act_str} m/s | Ent={ent:.4f} | Rew={reward:.4f}")
        
        # Log to Data Structure
        trajectory_data.append({
            'step': step + 1,
            'time': t_next,
            'action': action.tolist(),
            'reward': reward,
            'entropy': ent,
            'state': state.roe.tolist(),
            'next_state': next_state.roe.tolist(),
            'position': pos_next.tolist()
        })

        state = next_state

    # 6. Save All Artifacts
    os.makedirs(output_dir, exist_ok=True)
    
    # A. Save CSV
    df = pd.DataFrame(trajectory_data)
    df.to_csv(os.path.join(output_dir, "test_episode_data.csv"), index=False)
    
    # B. Save Entropy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(entropy_history, marker='o', linestyle='-', label='MCTS Agent')
    plt.title(f"Test Flight Entropy Reduction\nModel: {os.path.basename(checkpoint_path)}")
    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "test_entropy.png"))
    plt.close()
    
    # C. Save 3D Trajectory Plot (Comparable to Baseline)
    traj_arr = np.array(camera_positions)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_arr[:,0], traj_arr[:,1], traj_arr[:,2], label='Agent Trajectory', color='blue')
    ax.scatter(traj_arr[0,0], traj_arr[0,1], traj_arr[0,2], color='green', marker='o', label='Start')
    ax.scatter(traj_arr[-1,0], traj_arr[-1,1], traj_arr[-1,2], color='red', marker='x', label='End')
    ax.scatter(0, 0, 0, color='black', marker='*', s=100, label='Target')
    
    # Burns
    if burn_indices:
        burn_pos = traj_arr[burn_indices]
        ax.scatter(burn_pos[:,0], burn_pos[:,1], burn_pos[:,2], color='orange', marker='^', s=50, label='Maneuver', zorder=10)

    # Equal Aspect Ratio
    max_range = np.array([traj_arr[:,0].max()-traj_arr[:,0].min(), 
                          traj_arr[:,1].max()-traj_arr[:,1].min(), 
                          traj_arr[:,2].max()-traj_arr[:,2].min()]).max() / 2.0
    mid_x = (traj_arr[:,0].max()+traj_arr[:,0].min()) * 0.5
    mid_y = (traj_arr[:,1].max()+traj_arr[:,1].min()) * 0.5
    mid_z = (traj_arr[:,2].max()+traj_arr[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('R [m]'); ax.set_ylabel('T [m]'); ax.set_zlabel('N [m]')
    ax.set_title("Agent Trajectory")
    ax.legend()
    plt.savefig(os.path.join(output_dir, "test_trajectory.png"))
    plt.close()

    # D. Save Video
    frames = create_visualization_frames(
        output_dir, VoxelGrid((20,20,20)), GroundTruthRSO(VoxelGrid((20,20,20))), cp,
        np.array(camera_positions), np.array(view_directions), burn_indices
    )
    if frames:
        video_path = os.path.join(output_dir, "test_video.mp4")
        imageio.mimsave(video_path, frames, fps=5, macro_block_size=1)
        imageio.imwrite(os.path.join(output_dir, "test_final_frame.png"), frames[-1])
        print(f"Video saved to: {video_path}")

    print(f"\nAll test artifacts saved to: {output_dir}/")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    RUN_FOLDER = "output_training/run_2025-12-02_23-37-56" 
    CHECKPOINT_FILE = "checkpoint_ep_4.pt" 
    # ---------------------

    cfg = load_config()
    
    # Auto-detect latest run logic
    if not os.path.exists(RUN_FOLDER):
        base_dir = cfg['simulation'].get('output_dir', 'output_training')
        if os.path.exists(base_dir):
            runs = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith('run_')], key=os.path.getmtime)
            if runs:
                RUN_FOLDER = runs[-1]
                print(f"Auto-detected latest run: {RUN_FOLDER}")

    ckpt_path = os.path.join(RUN_FOLDER, "checkpoints", CHECKPOINT_FILE)
    output_path = os.path.join(RUN_FOLDER, "test_results")
    
    # Checkpoint Fallback
    if not os.path.exists(ckpt_path):
        ckpt_dir = os.path.join(RUN_FOLDER, "checkpoints")
        if os.path.exists(ckpt_dir):
            pts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
            if pts:
                # Sort numerically if possible to get 'checkpoint_ep_100' instead of '10'
                pts.sort(key=lambda f: int(''.join(filter(str.isdigit, f))) if any(c.isdigit() for c in f) else 0)
                ckpt_path = os.path.join(ckpt_dir, pts[-1])
                print(f"Checkpoint {CHECKPOINT_FILE} not found. Using latest: {pts[-1]}")

    run_simulation(cfg, ckpt_path, output_path)