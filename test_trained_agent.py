import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
from learning.policy_value_network import PolicyValueNetwork
from mcts.mcts_alphazero_controller import MCTSAlphaZeroCPU
from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from camera.camera_observations import VoxelGrid, GroundTruthRSO

def load_config(path="config.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

def run_simulation(config, checkpoint_path, output_dir):
    print(f"--- Loading Agent from {checkpoint_path} ---")
    
    # 1. Setup Environment
    op = config['orbit']
    cp = config['camera']
    ctrl_lambda = config.get('control', {}).get('lambda_dv', 0.01)
    
    # Initialize fresh grid and RSO
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
        n_iters=50,  # Higher iterations = stronger play
        gamma=0.99
    )

    # 4. Set Initial State
    rm = config['initial_roe_meters']
    am = op['a_chief_km'] * 1000.0
    base_roe = np.array([rm['da'], rm['dl'], rm['dex'], rm['dey'], rm['dix'], rm['diy']], dtype=float) / am
    
    state = OrbitalState(roe=base_roe, grid=grid, time=0.0)
    
    # 5. Run Simulation Loop
    steps = config['simulation']['num_steps']
    entropy_history = [grid.get_entropy()]
    print(f"Starting Simulation ({steps} steps)...")
    print(f"Initial Entropy: {entropy_history[0]:.4f}")

    for step in range(steps):
        # Run MCTS
        pi, value, _ = mcts_agent.search(state)
        
        # Select best action (Deterministic/Greedy for testing)
        best_idx = np.argmax(pi)
        action = mdp.get_all_actions()[best_idx]
        
        # Apply Action
        next_state, reward = mdp.step(state, action)
        
        # Metrics
        act_mag = np.linalg.norm(action)
        ent = next_state.grid.get_entropy()
        entropy_history.append(ent)
        
        print(f"Step {step+1:02d}: Action |dv|={act_mag:.3f} m/s | Value Est={value:.3f} | Entropy={ent:.4f}")
        
        state = next_state

    # 6. Plot Simulation Results
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(entropy_history, marker='o', linestyle='-')
    plt.title(f"Test Flight Entropy Reduction\nModel: {os.path.basename(checkpoint_path)}")
    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.grid(True)
    
    res_path = os.path.join(output_dir, "test_flight_entropy.png")
    plt.savefig(res_path)
    print(f"\nSimulation Complete. Result graph saved to: {res_path}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update this to point to the specific run you want to test
    # If using Windows, make sure to use forward slashes or raw strings for paths
    RUN_FOLDER = "output_training/run_2025-12-01_22-43-34" 
    CHECKPOINT_FILE = "best.pt"
    # ---------------------

    cfg = load_config()
    
    # Auto-detect latest run if the specific folder doesn't exist
    if not os.path.exists(RUN_FOLDER):
        base_dir = cfg['simulation'].get('output_dir', 'output_training')
        if os.path.exists(base_dir):
            runs = sorted([d for d in os.listdir(base_dir) if d.startswith('run_')])
            if runs:
                RUN_FOLDER = os.path.join(base_dir, runs[-1])
                print(f"Auto-detected latest run: {RUN_FOLDER}")

    ckpt_path = os.path.join(RUN_FOLDER, "checkpoints", CHECKPOINT_FILE)
    output_path = os.path.join(RUN_FOLDER, "test_results")
    
    run_simulation(cfg, ckpt_path, output_path)