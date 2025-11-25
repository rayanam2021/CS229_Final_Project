import os
import json
import numpy as np

from simulation.scenario_full_mcts import run_orbital_camera_sim_full_mcts
from datetime import datetime

# Created to test sweeps for many params and find sweet spot

if __name__ == "__main__":
    # You can set parameters here, or use defaults from scenario_full_mcts.py
    # c_values = [0.7, 1.4, 2.0]
    iterations = [500, 1000, 1500]
    seeds = [0, 1, 2]

    for iter in iterations:
        for seed in seeds:
            # out_dir = f"output/experiments/c_sweep/c_{c}_seed_{seed}"
            out_dir = f"output/experiments/iteration_sweep/i_{iter}_seed_{seed}"
            os.makedirs(out_dir, exist_ok=True)

            config = {
                "run_id": f"c_{c}_seed_{seed}",
                "mcts_iters": 1000,
                "mcts_c": 1.4,
                "gamma": 0.95,
                "horizon": 15,
                "lambda_dv": 0.01,
                "num_steps": 20,
                "time_step": 30.0,

                "alpha_dv":10,
                "beta_tan":0.5,
                "seed": seed,
            }
            with open(os.path.join(out_dir, f"config_{c}_{seed}.json"), "w") as f:
                json.dump(config, f, indent=2)

            np.random.seed(seed)
            
            run_orbital_camera_sim_full_mcts(
                num_steps    = config["num_steps"],
                time_step    = config["time_step"],
                horizon      = config["horizon"],
                mcts_iters   = config["mcts_iters"],
                mcts_c       = config["mcts_c"],  # exploration constant
                mcts_gamma   = config["gamma"],
                lambda_dv    = config["lambda_dv"],
                alpha_dv     = config["alpha_dv"],
                beta_tan     = config["beta_tan"],
                verbose      = True,
                visualize    = True,
                out_folder   = out_dir
            )
