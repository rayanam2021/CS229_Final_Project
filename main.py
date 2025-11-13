from simulation.scenario_full_mcts import run_orbital_camera_sim_full_mcts
import os
from datetime import datetime

if __name__ == "__main__":
    # You can set parameters here, or use defaults from scenario_full_mcts.py
    
    MAX_HORIZON = 2
    NUM_STEPS = 20
    TIME_STEP = 30.0  # seconds
    VERBOSE = True
    VISUALIZE = True
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUT_FOLDER = os.path.join("output", timestamp)
    os.makedirs(OUT_FOLDER, exist_ok=True)

    run_orbital_camera_sim_full_mcts(horizon=MAX_HORIZON, num_steps=NUM_STEPS, time_step=TIME_STEP, verbose=VERBOSE, visualize=VISUALIZE, out_folder=OUT_FOLDER)