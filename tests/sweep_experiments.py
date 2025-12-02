import os
import json
import numpy as np
import pandas as pd
import time

from simulation.scenario_full_mcts import run_orbital_camera_sim_full_mcts
from datetime import datetime

# Sweep over MCTS parameters with random rollout policy

if __name__ == "__main__":
    # Sweep parameters for MCTS tree search
    mcts_iters_values = [500, 1000, 2000]     # Number of simulations per planning step
    mcts_c_values = [0.7, 1.4, 2.0]           # UCB exploration constant
    gamma_values = [0.90, 0.95, 0.99]         # Discount factor for future rewards
    max_horizon_values = [10, 20, 40]         # Tree search depth (steps)
    lambda_dv_values = [0.001, 0.01, 0.1]     # Fuel cost weight in reward
    seeds = [0, 1, 2]

    # Initialize results file with header
    summary_path = "output/experiments/mcts_sweep_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    results = []
    sweep_start_time = time.time()
    total_runs = len(mcts_iters_values) * len(mcts_c_values) * len(gamma_values) * len(max_horizon_values) * len(lambda_dv_values) * len(seeds)
    run_count = 0

    for mcts_iters in mcts_iters_values:
        for mcts_c in mcts_c_values:
            for gamma in gamma_values:
                for max_horizon in max_horizon_values:
                    for lambda_dv in lambda_dv_values:
                        for seed in seeds:
                            run_count += 1
                            out_dir = f"output/experiments/mcts_sweep/iters_{mcts_iters}_c_{mcts_c}_gamma_{gamma}_h_{max_horizon}_lam_{lambda_dv}_seed_{seed}"
                            os.makedirs(out_dir, exist_ok=True)

                            config = {
                                "run_id": f"iters_{mcts_iters}_c_{mcts_c}_gamma_{gamma}_h_{max_horizon}_lam_{lambda_dv}_seed_{seed}",
                                "mcts_iters": mcts_iters,
                                "mcts_c": mcts_c,
                                "gamma": gamma,
                                "max_horizon": max_horizon,
                                "lambda_dv": lambda_dv,
                                "num_steps": 20,
                                "time_step": 30.0,
                                "seed": seed,
                                "desired_rollout_policy": "random",  # Use random policy (default)
                                "alpha_dv": 60.0,                     # Not used with random policy
                                "beta_tan": 0.1,                      # Not used with random policy
                            }

                            with open(os.path.join(out_dir, f"config.json"), "w") as f:
                                json.dump(config, f, indent=2)

                            print(f"\n{'='*70}")
                            print(f"Run {run_count}/{total_runs}")
                            print(f"mcts_iters={mcts_iters}, mcts_c={mcts_c}, gamma={gamma}, horizon={max_horizon}, lambda_dv={lambda_dv}, seed={seed}")
                            print(f"Output: {out_dir}")
                            print(f"{'='*70}")

                            np.random.seed(seed)
                            run_start_time = time.time()

                            try:
                                run_orbital_camera_sim_full_mcts(
                                    sim_config=config,
                                    orbit_params={
                                        "mu_earth": 398600.4418,
                                        "a_chief_km": 7000.0,
                                        "e_chief": 0.001,
                                        "i_chief_deg": 98.0,
                                        "omega_chief_deg": 30.0,
                                    },
                                    camera_params={
                                        "fov_degrees": 10.0,
                                        "sensor_res": [64, 64],
                                        "noise_params": {
                                            "p_hit_given_occupied": 0.95,
                                            "p_hit_given_empty": 0.001,
                                        }
                                    },
                                    initial_state_roe=np.array([0.0, 200.0, 100.0, 0.0, 50.0, 0.0]) / 7000000.0,  # meters -> dimensionless
                                    out_folder=out_dir
                                )

                                # Read results from replay buffer
                                replay_path = os.path.join(out_dir, "replay_buffer.csv")
                                if os.path.exists(replay_path):
                                    df = pd.read_csv(replay_path)
                                    final_entropy = df["entropy_after"].iloc[-1] if len(df) > 0 else np.nan
                                    total_dv_cost = df["dv_cost"].sum() if "dv_cost" in df.columns else np.nan
                                    info_gain_per_dv = (df["info_gain"].sum() / total_dv_cost) if total_dv_cost > 0 else np.nan

                                    result = {
                                        "mcts_iters": mcts_iters,
                                        "mcts_c": mcts_c,
                                        "gamma": gamma,
                                        "max_horizon": max_horizon,
                                        "lambda_dv": lambda_dv,
                                        "seed": seed,
                                        "final_entropy": final_entropy,
                                        "total_dv_cost": total_dv_cost,
                                        "total_info_gain": df["info_gain"].sum() if "info_gain" in df.columns else np.nan,
                                        "info_gain_per_dv": info_gain_per_dv,
                                    }
                                else:
                                    result = {
                                        "mcts_iters": mcts_iters,
                                        "mcts_c": mcts_c,
                                        "gamma": gamma,
                                        "max_horizon": max_horizon,
                                        "lambda_dv": lambda_dv,
                                        "seed": seed,
                                        "error": "replay_buffer.csv not found",
                                    }
                            except Exception as e:
                                print(f"ERROR in run: {e}")
                                result = {
                                    "mcts_iters": mcts_iters,
                                    "mcts_c": mcts_c,
                                    "gamma": gamma,
                                    "max_horizon": max_horizon,
                                    "lambda_dv": lambda_dv,
                                    "seed": seed,
                                    "error": str(e),
                                }

                            # Calculate running time for this simulation
                            run_elapsed_time = time.time() - run_start_time
                            result["elapsed_time_seconds"] = run_elapsed_time

                            # Append result to list and save immediately
                            results.append(result)
                            results_df = pd.DataFrame(results)
                            results_df.to_csv(summary_path, index=False)
                            print(f"Result saved to {summary_path}")
                            print(f"Elapsed time: {run_elapsed_time:.2f} seconds ({run_elapsed_time/60:.2f} minutes)")

    # Calculate and save total sweep time
    total_sweep_time = time.time() - sweep_start_time
    hours = int(total_sweep_time // 3600)
    minutes = int((total_sweep_time % 3600) // 60)
    seconds = int(total_sweep_time % 60)

    print(f"\n\n{'='*70}")
    print(f"SWEEP COMPLETE!")
    print(f"Total sweep time: {hours}h {minutes}m {seconds}s ({total_sweep_time:.2f} seconds)")
    print(f"Average time per run: {total_sweep_time/len(results):.2f} seconds")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}")
    print(results_df)

    # Also save timing summary to a text log file
    log_path = os.path.join(os.path.dirname(summary_path), "sweep_timing.log")
    with open(log_path, "w") as f:
        f.write(f"MCTS Comprehensive Sweep Timing Report\n")
        f.write(f"{'='*70}\n")
        f.write(f"Sweep end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total sweep time: {hours}h {minutes}m {seconds}s ({total_sweep_time:.2f} seconds)\n")
        f.write(f"Number of runs: {len(results)}\n")
        f.write(f"Average time per run: {total_sweep_time/len(results):.2f} seconds\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Parameter grid:\n")
        f.write(f"  mcts_iters: {mcts_iters_values}\n")
        f.write(f"  mcts_c: {mcts_c_values}\n")
        f.write(f"  gamma: {gamma_values}\n")
        f.write(f"  max_horizon: {max_horizon_values}\n")
        f.write(f"  lambda_dv: {lambda_dv_values}\n")
        f.write(f"  seeds: {seeds}\n")
        f.write(f"Total combinations: {total_runs}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Per-run timings:\n")
        for idx, row in results_df.iterrows():
            if "elapsed_time_seconds" in row:
                f.write(f"  iters={row['mcts_iters']}, c={row['mcts_c']}, gamma={row['gamma']}, h={row['max_horizon']}, lam={row['lambda_dv']}, seed={row['seed']}: {row['elapsed_time_seconds']:.2f}s\n")

    print(f"Timing log saved to: {log_path}")
