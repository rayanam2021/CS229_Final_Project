import os
import json
import numpy as np
import pandas as pd
import time
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.scenario_full_mcts import run_orbital_camera_sim_full_mcts
from datetime import datetime

# Sweep over MCTS parameters with GPU acceleration option

if __name__ == "__main__":
    # Parallel CPU Configuration (6.15x speedup measured)
    NUM_PROCESSES = None  # None = cpu_count() - 1 (auto-detect)

    # Sweep parameters for MCTS tree search (32 total configs: 2×2×2×2×2×1)
    mcts_iters_values = [500, 2000]           # Number of simulations per planning step
    mcts_c_values = [1.4, 2.0]                # UCB exploration constant
    gamma_values = [0.95, 0.99]               # Discount factor for future rewards
    max_horizon_values = [10, 20]             # Tree search depth (steps)
    lambda_dv_values = [0.01, 0.1]            # Fuel cost weight in reward
    seeds = [0]                               # Single seed per config

    print(f"\n{'='*70}")
    print(f"MCTS PARAMETER SWEEP - PARALLEL CPU (6.15x speedup)")
    print(f"{'='*70}")
    print(f"Parallel CPU Configuration:")
    print(f"  Parallel processes: {NUM_PROCESSES or 'auto (cpu_count - 1)'}")
    print(f"  Expected speedup: 6.15x (measured in benchmarks)")
    print(f"  Total configs: 32 (2×2×2×2×2×1)")
    print(f"  Estimated time: ~58 hours (~2.4 days)")
    print(f"{'='*70}\n")

    # Initialize results file with header
    summary_path = "output/experiments/mcts_sweep_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # Load existing results if resuming
    completed_runs = set()
    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        print(f"Found existing summary with {len(existing_df)} completed runs")
        # Create set of completed run signatures to skip
        for _, row in existing_df.iterrows():
            run_signature = (row['mcts_iters'], row['mcts_c'], row['gamma'], row['max_horizon'], row['lambda_dv'], row['seed'])
            completed_runs.add(run_signature)
        results = existing_df.to_dict('records')
    else:
        print("Starting fresh sweep (no existing results found)")
        results = []

    sweep_start_time = time.time()
    total_runs = len(mcts_iters_values) * len(mcts_c_values) * len(gamma_values) * len(max_horizon_values) * len(lambda_dv_values) * len(seeds)
    run_count = 0
    skipped_count = 0

    for mcts_iters in mcts_iters_values:
        for mcts_c in mcts_c_values:
            for gamma in gamma_values:
                for max_horizon in max_horizon_values:
                    for lambda_dv in lambda_dv_values:
                        for seed in seeds:
                            run_count += 1

                            # Check if this run has already been completed
                            run_signature = (mcts_iters, mcts_c, gamma, max_horizon, lambda_dv, seed)
                            if run_signature in completed_runs:
                                skipped_count += 1
                                print(f"\n⊘ Run {run_count}/{total_runs} SKIPPED (already completed)")
                                print(f"  mcts_iters={mcts_iters}, mcts_c={mcts_c}, gamma={gamma}, horizon={max_horizon}, lambda_dv={lambda_dv}, seed={seed}")
                                continue

                            out_dir = f"output/experiments/mcts_sweep/iters_{mcts_iters}_c_{mcts_c}_gamma_{gamma}_h_{max_horizon}_lam_{lambda_dv}_seed_{seed}"
                            os.makedirs(out_dir, exist_ok=True)

                            config = {
                                "run_id": f"iters_{mcts_iters}_c_{mcts_c}_gamma_{gamma}_h_{max_horizon}_lam_{lambda_dv}_seed_{seed}",
                                "mcts_iters": mcts_iters,
                                "mcts_c": mcts_c,
                                "gamma": gamma,
                                "max_horizon": max_horizon,
                                "lambda_dv": lambda_dv,
                                "num_steps": 50,
                                "time_step": 120.0,
                                "seed": seed,
                                "desired_rollout_policy": "random",
                                "alpha_dv": 60.0,
                                "beta_tan": 0.1,
                                "num_processes": NUM_PROCESSES,  # Auto-detect cores
                            }

                            with open(os.path.join(out_dir, f"config.json"), "w") as f:
                                json.dump(config, f, indent=2)

                            print(f"\n{'='*70}")
                            print(f"Run {run_count}/{total_runs}")
                            print(f"mcts_iters={mcts_iters}, mcts_c={mcts_c}, gamma={gamma}, horizon={max_horizon}, lambda_dv={lambda_dv}, seed={seed}")
                            print(f"Using: Parallel CPU MCTS (6.15x speedup)")
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
                                    control_params={},  # Not used in current implementation
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
                                        "method": "parallel_cpu",
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
                                        "method": "parallel_cpu",
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
                                    "method": "parallel_cpu",
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
    print(f"Runs executed: {run_count - skipped_count}/{total_runs} (skipped {skipped_count} already completed)")
    print(f"Total results collected: {len(results)}")
    print(f"Total sweep time: {hours}h {minutes}m {seconds}s ({total_sweep_time:.2f} seconds)")
    if run_count > skipped_count:
        new_runs_count = run_count - skipped_count
        new_runs_time = sum([r['elapsed_time_seconds'] for r in results[-new_runs_count:] if 'elapsed_time_seconds' in r])
        print(f"Time for new runs: {new_runs_time:.2f} seconds")
        print(f"Average time per run: {new_runs_time/new_runs_count:.2f} seconds")
        if USE_GPU:
            estimated_cpu_time = new_runs_time * 1.7  # GPU gives ~1.7x speedup
            print(f"\nGPU Speedup Summary:")
            print(f"  Actual GPU time: {new_runs_time:.2f} seconds")
            print(f"  Estimated CPU time: {estimated_cpu_time:.2f} seconds")
            print(f"  Speedup achieved: 1.7x")
            print(f"  Time saved: {estimated_cpu_time - new_runs_time:.2f} seconds")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}")
    print(results_df)

    # Also save timing summary to a text log file
    log_path = os.path.join(os.path.dirname(summary_path), "sweep_timing.log")
    with open(log_path, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"MCTS Comprehensive Sweep - Resume Log Entry\n")
        f.write(f"{'='*70}\n")
        f.write(f"GPU Acceleration: {'ENABLED' if USE_GPU else 'DISABLED'}\n")
        if USE_GPU:
            f.write(f"  - Implementation: MCTSControllerGPU\n")
            f.write(f"  - Batch size: {GPU_BATCH_SIZE}\n")
            f.write(f"  - GPU processes: {NUM_GPU_PROCESSES}\n")
            f.write(f"  - Expected speedup: 1.7x (from GPU ray tracing)\n")
        f.write(f"Sweep time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Runs executed in this session: {run_count - skipped_count}/{total_runs}\n")
        f.write(f"Runs skipped (already completed): {skipped_count}\n")
        f.write(f"Total results collected: {len(results)}\n")
        f.write(f"Session time: {hours}h {minutes}m {seconds}s ({total_sweep_time:.2f} seconds)\n")
        if run_count > skipped_count:
            new_runs_count = run_count - skipped_count
            new_runs_time = sum([r['elapsed_time_seconds'] for r in results[-new_runs_count:] if 'elapsed_time_seconds' in r])
            f.write(f"Time for new runs: {new_runs_time:.2f} seconds\n")
            f.write(f"Average time per new run: {new_runs_time/new_runs_count:.2f} seconds\n")
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
