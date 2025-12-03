import numpy as np
import torch
import os
import json
import logging
import sys
import time
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from datetime import datetime

# Import local modules
from learning.training import SelfPlayTrainer
from learning.policy_value_network import PolicyValueNetwork
from mcts.mcts_alphazero_controller import MCTSAlphaZeroCPU
from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from camera.camera_observations import VoxelGrid, GroundTruthRSO, simulate_observation, plot_scenario, update_plot

def create_visualization_frames(out_folder, grid_initial, rso, camera_fn, camera_positions, view_directions, burn_indices):
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
    plt.close(fig)
    return frames

class AlphaZeroTrainer:
    def __init__(self, config_path="config.json"):
        if not os.path.exists(config_path): raise FileNotFoundError
        with open(config_path, 'r') as f: self.config = json.load(f)
        
        self.base_out_dir = self.config['simulation'].get('output_dir', 'output_training')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(self.base_out_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        self._setup_logging()
        self.setup_environment()
        
        self.network = PolicyValueNetwork(grid_dims=self.grid.dims, num_actions=13, hidden_dim=128)
        
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.trainer = SelfPlayTrainer(
            network=self.network, 
            learning_rate=self.config['training'].get('learning_rate', 0.001),
            weight_decay=1e-4, 
            device='cpu',
            checkpoint_dir=ckpt_dir,
            max_buffer_size=self.config['training'].get('buffer_size', 10000)
        )

    def _setup_logging(self):
        self.logger = logging.getLogger("AlphaZero")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(self.run_dir, "training.log"))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

    def log(self, msg): self.logger.info(msg)

    def setup_environment(self):
        op, cp, rm = self.config['orbit'], self.config['camera'], self.config['initial_roe_meters']
        ctrl = self.config.get('control', {'lambda_dv': 0.01})
        am = op['a_chief_km'] * 1000.0
        self.base_roe = np.array([rm['da'], rm['dl'], rm['dex'], rm['dey'], rm['dix'], rm['diy']], dtype=float) / am
        self.grid = VoxelGrid(grid_dims=(20,20,20))
        self.rso = GroundTruthRSO(self.grid)
        self.mdp = OrbitalMCTSModel(op['a_chief_km'], op['e_chief'], np.deg2rad(op['i_chief_deg']), 
                                    np.deg2rad(op['omega_chief_deg']), np.sqrt(op['mu_earth']/op['a_chief_km']**3), 
                                    self.rso, cp, self.grid.dims, ctrl['lambda_dv'], 
                                    self.config['simulation']['time_step'], self.config['simulation']['max_horizon'])

    def get_perturbed_state(self):
        b, am = self.config['monte_carlo']['perturbation_bounds'], self.config['orbit']['a_chief_km'] * 1000.0
        p = np.array([np.random.uniform(-b[k], b[k]) for k in ['da','dl','dex','dey','dix','diy']])
        return OrbitalState(self.base_roe + p/am, VoxelGrid(self.grid.dims), 0.0)

    def plot_history(self):
        history = self.trainer.training_history
        if not history.get('total_loss'):
            self.log("No loss history to plot.")
            return

        epochs = range(1, len(history['total_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['total_loss'], label='Total Loss', color='blue')
        plt.plot(epochs, history['value_loss'], label='Value Loss', color='orange', linestyle='--')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Convergence'); plt.legend(); plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['policy_loss'], color='green', label='Policy Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Policy Loss'); plt.legend(); plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.run_dir, "loss_history.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        self.log(f"Training loss plot saved to {save_path}")

    def run_training_loop(self):
        cfg = self.config
        num_episodes, steps = cfg['monte_carlo']['num_episodes'], cfg['simulation']['num_steps']
        tr_cfg = cfg['training']
        
        with open(os.path.join(self.run_dir, "run_config.json"), "w") as f: json.dump(cfg, f, indent=4)
        self.log(f"STARTING TRAINING: {num_episodes} Episodes")
        
        for episode in range(num_episodes):
            ep_start_time = time.time()
            ep_dir = os.path.join(self.run_dir, f"episode_{episode+1:02d}")
            os.makedirs(ep_dir, exist_ok=True)
            self.log(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            state = self.get_perturbed_state()
            roe_str = np.array2string(state.roe * self.config['orbit']['a_chief_km'] * 1000.0, precision=1, separator=', ')
            self.log(f"Initial ROE (m): {roe_str}")
            
            # --- CRITICAL UPDATE: Set Initial Entropy ---
            # This allows the MDP to normalize rewards relative to the starting condition
            initial_ent = state.grid.get_entropy()
            self.mdp.initial_entropy = initial_ent
            self.log(f"Initial Entropy: {initial_ent:.4f}")
            # --------------------------------------------
            
            trajectory = []
            camera_positions = []
            view_directions = []
            burn_indices = []
            entropy_history = [initial_ent]
            
            # Initial Viz Pos
            from roe.propagation import propagateGeomROE
            rho_start, _ = propagateGeomROE(state.roe, self.mdp.a_chief, self.mdp.e_chief, self.mdp.i_chief, self.mdp.omega_chief, self.mdp.n_chief, np.array([0.0]), t0=0.0)
            pos_start = rho_start[:,0]*1000
            camera_positions.append(pos_start)
            view_directions.append(-pos_start/np.linalg.norm(pos_start))
            
            mcts = MCTSAlphaZeroCPU(self.mdp, self.network, n_iters=tr_cfg['mcts_iters'], c_puct=tr_cfg['c_puct'], gamma=tr_cfg['gamma'])
            sim_time = 0.0

            for step in range(steps):
                pi, _, root_node = mcts.search(state)
                
                # Export tree occasionally
                if step == 0 or step % 5 == 0:
                    try:
                        mcts.export_tree_to_dot(root_node, episode+1, step+1, os.path.join(ep_dir, "trees"))
                    except Exception as e:
                        pass
                
                action_idx = np.random.choice(len(pi), p=pi)
                action = self.mdp.get_all_actions()[action_idx]
                
                if np.linalg.norm(action) > 1e-6: burn_indices.append(len(camera_positions)-1)
                
                next_state, reward = self.mdp.step(state, action)
                
                # Viz Data Propagation
                t_burn = np.array([state.time])
                from roe.dynamics import apply_impulsive_dv
                imp_state = apply_impulsive_dv(state.roe, action, self.mdp.a_chief, self.mdp.n_chief, t_burn, self.mdp.e_chief, self.mdp.i_chief, self.mdp.omega_chief)
                t_next = np.array([state.time + self.mdp.time_step])
                rho_n, _ = propagateGeomROE(imp_state, self.mdp.a_chief, self.mdp.e_chief, self.mdp.i_chief, self.mdp.omega_chief, self.mdp.n_chief, t_next, t0=state.time)
                pos_next = rho_n[:,0]*1000
                
                camera_positions.append(pos_next)
                view_directions.append(-pos_next/np.linalg.norm(pos_next))
                
                ent = next_state.grid.get_entropy()
                entropy_history.append(ent)
                
                trajectory.append({
                    'roe': state.roe, 
                    'belief': state.grid.belief.copy(), 
                    'pi': pi, 
                    'reward': reward, 
                    'action': action, 
                    'time': sim_time,
                    'next_roe': next_state.roe
                })
                
                act_str = np.array2string(action, precision=3, separator=', ', suppress_small=True)
                self.log(f"Step {step+1}: Act={act_str} m/s | Ent={ent:.4f} | Rew={reward:.4f}")
                state = next_state
                sim_time += self.mdp.time_step

            # --- CSV LOGGING ---
            data_rows = []
            for i, t in enumerate(trajectory):
                # entropy_history[0] is initial, [i+1] is after step i
                row = {
                    'time': t['time'],
                    'step': i + 1,
                    'action': t['action'].tolist(),
                    'reward': t['reward'],
                    'entropy': entropy_history[i+1],
                    'state': t['roe'].tolist(),
                    'next_state': t['next_roe'].tolist()
                }
                data_rows.append(row)

            df = pd.DataFrame(data_rows)
            df.to_csv(os.path.join(ep_dir, "episode_data.csv"), index=False)
            
            plt.figure(); plt.plot(entropy_history, marker='o'); plt.savefig(os.path.join(ep_dir, "entropy.png")); plt.close()

            if cfg['simulation'].get('visualize', True):
                frames = create_visualization_frames(ep_dir, self.grid, self.rso, self.config['camera'], 
                                                    np.array(camera_positions), np.array(view_directions), burn_indices)
                if frames:
                    imageio.mimsave(os.path.join(ep_dir, "video.mp4"), frames, fps=5, macro_block_size=1)
                    imageio.imwrite(os.path.join(ep_dir, "final_frame.png"), frames[-1])

            # --- Training Update ---
            R = 0
            for t in reversed(range(len(trajectory))):
                R = trajectory[t]['reward'] + tr_cfg['gamma'] * R
                
                self.trainer.add_to_replay_buffer(
                    trajectory[t]['roe'], 
                    trajectory[t]['belief'], 
                    trajectory[t]['pi'], 
                    float(R),
                    trajectory[t]['action'],
                    trajectory[t]['reward'],
                    trajectory[t]['next_roe'],
                    trajectory[t]['time']
                )
            
            if len(self.trainer.replay_buffer) >= tr_cfg['batch_size']:
                self.log("Training Network...")
                for _ in range(tr_cfg['epochs_per_cycle']):
                    l = self.trainer.train_epoch(5, tr_cfg['batch_size'])
                self.log(f"Loss: P={l['policy_loss']:.4f} V={l['value_loss']:.4f}")
            
            self.trainer.save_checkpoint(episode + 1)
            self.log(f"Episode Time: {time.time()-ep_start_time:.1f}s")

        self.plot_history()
        self.log("Complete.")