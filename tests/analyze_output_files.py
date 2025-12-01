import os
import json
import pandas as pd
import numpy as np

def find_config_file(run_dir):
    """
    Look for a JSON file that starts with 'config' in run_dir.
    e.g. config_0.7_0.json
    """
    for fname in os.listdir(run_dir):
        if fname.startswith("config") and fname.endswith(".json"):
            return os.path.join(run_dir, fname)
    return None

def load_run(run_dir):
    config_path = find_config_file(run_dir)
    ent_path    = os.path.join(run_dir, "entropy_history.csv")
    rep_path    = os.path.join(run_dir, "replay_buffer.csv")

    if config_path is None or \
       not os.path.isfile(ent_path) or \
       not os.path.isfile(rep_path):
        return None

    with open(config_path, "r") as f:
        config = json.load(f)

    ent = pd.read_csv(ent_path)
    rep = pd.read_csv(rep_path)

    # Entropy metrics (your file has only "entropy")
    H0 = float(ent.loc[0, "entropy"])
    Hf = float(ent.loc[ent.index[-1], "entropy"])
    dH = H0 - Hf
    dH_pct = 100 * dH / H0

    # ---- NEW: handle dv / cost columns robustly ----
    lambda_dv = config.get("lambda_dv", None)

    if "total_dv" in rep.columns:
        # If you ever add a cumulative column later
        total_dv = float(rep.loc[rep.index[-1], "total_dv"])
    elif "dv_cost" in rep.columns:
        # Per-step λ * ||Δv||; sum to get total cost
        total_dv_cost = float(rep["dv_cost"].sum())
        if lambda_dv not in (None, 0):
            total_dv = total_dv_cost / lambda_dv  # actual Δv magnitude
        else:
            total_dv = total_dv_cost              # fallback: just report cost
    else:
        total_dv = None  # we’ll still keep other metrics

    run_metrics = {
        "run_dir": run_dir,
        "run_id": config.get("run_id", None),
        "seed": config.get("seed", None),
        "c": config.get("mcts_c", None),
        "iters": config.get("mcts_iters", None),
        "gamma": config.get("gamma", None),
        "horizon": config.get("horizon", None),
        "lambda_dv": lambda_dv,
        "H0": H0,
        "Hf": Hf,
        "ΔH": dH,
        "ΔH%": dH_pct,
        "total_dv": total_dv,
    }

    return run_metrics


def scan_experiments(root_dir):
    """
    Recursively scans `root_dir` for experiment folders containing:
      - config_*.json
      - entropy_history.csv
      - replay_buffer.csv
    Returns a pandas DataFrame with all runs.
    """
    all_runs = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # We only care about directories that have both CSVs
        if "entropy_history.csv" in filenames and "replay_buffer.csv" in filenames:
            metrics = load_run(dirpath)
            if metrics is not None:
                all_runs.append(metrics)

    if len(all_runs) == 0:
        print("No experiment runs found in:", root_dir)
        return None

    df = pd.DataFrame(all_runs)
    return df


def summarize(df, group_cols=["c"], metrics=["ΔH%", "total_dv"]):
    """
    Group by some config fields (e.g., c or iters) and compute mean/std.
    """
    summary = df.groupby(group_cols)[metrics].agg(["mean", "std"])
    return summary


if __name__ == "__main__":
    # Adjust this path to your setup; from your screenshot it’s:
    ROOT = os.path.join(os.getcwd(), "output/experiments/c_sweep")

    df = scan_experiments(ROOT)
    if df is not None:
        print("\n=== All Runs ===")
        print(df)

        print("\n=== Summary by c ===")
        print(summarize(df, group_cols=["c"], metrics=["ΔH%", "total_dv"]))
