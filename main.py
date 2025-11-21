import json
import os
import numpy as np

from simulation.scenario_full_mcts import run_orbital_camera_sim_full_mcts
from datetime import datetime

def load_config(path="config.json"):
    with open(path, "r") as f:
        cfg = json.load(f)
        cfg["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")     # just to know when analyzing results

        return cfg

if __name__ == "__main__":
    # You can set parameters here, or use defaults from scenario_full_mcts.py
    cfg = load_config("config.json")

    if "seed" in cfg:
        np.random.seed(cfg["seed"])
    
    OUT_FOLDER = os.path.join("output", cfg.get("run_id"))#, cfg.get("timestamp"))
    os.makedirs(OUT_FOLDER, exist_ok=True)

    run_orbital_camera_sim_full_mcts(
        num_steps    = cfg.get("num_steps", 20),
        time_step    = cfg.get("time_step", 30.0),
        horizon      = cfg.get("horizon", 10),
        mcts_iters   = cfg.get("mcts_iters", 100),
        mcts_c       = cfg.get("mcts_c", 1.4),  # exploration constant
        mcts_gamma   = cfg.get("gamma", 1.4),
        verbose      = cfg.get("verbose", True),
        visualize    = cfg.get("visualize", True),
        lambda_dv    = cfg.get("lambda_dv", True),
        out_folder   = OUT_FOLDER
    )

    with open(os.path.join(OUT_FOLDER, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
