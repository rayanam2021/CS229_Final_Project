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
        num_steps    = cfg.get("num_steps"),
        time_step    = cfg.get("time_step"),
        horizon      = cfg.get("horizon"),
        mcts_iters   = cfg.get("mcts_iters"),
        mcts_c       = cfg.get("mcts_c"),  # exploration constant
        mcts_gamma   = cfg.get("gamma"),
        alpha_dv     = cfg.get("alpha_dv"),
        beta_tan     = cfg.get("beta_tan"),
        target_radius = cfg.get("target_radius"),
        gamma_r     = cfg.get("gamma_r"),
        r_min_rollout = cfg.get("r_min_rollout"),
        r_max_rollout = cfg.get("r_max_rollout"),
        verbose      = cfg.get("verbose", True),
        visualize    = cfg.get("visualize", True),
        lambda_dv    = cfg.get("lambda_dv", True),
        out_folder   = OUT_FOLDER
    )

    with open(os.path.join(OUT_FOLDER, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
