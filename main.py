import os
import json
import numpy as np
from datetime import datetime
from simulation.scenario_full_mcts import run_orbital_camera_sim_full_mcts

def get_dimensionless_roe(roe_meters, a_chief_km):
    """
    Converts ROE components defined in meters into the dimensionless 
    Quasi-Nonsingular ROE state vector required for propagation.
    """
    a_chief_m = a_chief_km * 1000.0
    
    # Extract keys safely
    da = roe_meters.get('da', 0.0)
    dl = roe_meters.get('dl', 0.0)
    dex = roe_meters.get('dex', 0.0)
    dey = roe_meters.get('dey', 0.0)
    dix = roe_meters.get('dix', 0.0)
    diy = roe_meters.get('diy', 0.0)
    
    vec_meters = np.array([da, dl, dex, dey, dix, diy], dtype=float)
    return vec_meters / a_chief_m

def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    # 1. Load Configuration
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found. Please create it first.")
        exit(1)
        
    config = load_config(config_path)
    
    # 2. Extract sections
    sim_conf = config['simulation']
    orbit_conf = config['orbit']
    cam_conf = config['camera']
    roe_meters = config['initial_roe_meters']
    
    # --- NEW: Extract Control Config ---
    # This is required for the MCTS to know the fuel cost (lambda_dv)
    ctrl_conf = config.get('control', {'lambda_dv': 0.0})
    
    # 3. Process Initial State (Meters -> Dimensionless)
    a_chief_km = orbit_conf['a_chief_km']
    initial_roe_dimless = get_dimensionless_roe(roe_meters, a_chief_km)
    
    print("="*50)
    print(f"Initializing Simulation from {config_path}")
    print(f"Orbit: a={a_chief_km} km, i={orbit_conf['i_chief_deg']} deg")
    print(f"Initial ROE (Meters): {list(roe_meters.values())}")
    print(f"Initial ROE (Dimless): {np.round(initial_roe_dimless, 6)}")
    print("="*50)

    # 4. Setup Output Directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_out = sim_conf.get("output_dir", "output")
    out_folder = os.path.join(base_out, timestamp)
    os.makedirs(out_folder, exist_ok=True)
    
    # Save a copy of the config used for this run (Done BEFORE run in case of crash)
    with open(os.path.join(out_folder, "run_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 5. Run Simulation
    run_orbital_camera_sim_full_mcts(
        sim_config=sim_conf,
        orbit_params=orbit_conf,
        camera_params=cam_conf,
        control_params=ctrl_conf,     # <--- PASSED HERE
        initial_state_roe=initial_roe_dimless,
        out_folder=out_folder
    )