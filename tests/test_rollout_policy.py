import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from camera.camera_observations import VoxelGrid, GroundTruthRSO

# To test new heuristic rollout policy:
# - Prefer smaller |dv|
# - Prefer tangential/normal actions (parallax)
# - Prefer radius near target_radius
# - Reject actions that push radius outside [r_min_rollout, r_max_rollout]

def build_test_model():
    # --- 1. Make a simple grid + RSO + camera fn ---
    grid_dims = (32, 32, 32)
    grid = VoxelGrid(grid_dims)

    # Example RSO: a simple cube from your existing code
    rso = GroundTruthRSO(grid)

    camera_fn = {
        "fov_degrees": 60.0,
        "sensor_res": (32, 32),
        "noise_params": {
            "hit_false_neg": 0.05,
            "empty_false_pos": 0.001,
        },
    }

    # Orbital parameters: use whatever you use in scenario_full_mcts
    a_chief = 7000e3
    e_chief = 0.0
    i_chief = 0.0
    omega_chief = 0.0
    n_chief = np.sqrt(3.986004418e14 / a_chief**3)
    time_step = 30.0

    model = OrbitalMCTSModel(
        a_chief=a_chief,
        e_chief=e_chief,
        i_chief=i_chief,
        omega_chief=omega_chief,
        n_chief=n_chief,
        time_step=time_step,
        grid_dims=grid_dims,
        rso=rso,
        camera_fn=camera_fn,
        lambda_dv=0.01,
        max_depth=15
    )

    # --- 2. Build a test state (some relative position & empty grid) ---
    roe = np.array([0.0, 0.001, 0.0, 0.0, 0.0, 0.0])  # example ROE
    state = OrbitalState(roe=roe, grid=grid)

    return model, state

def main(num_samples=10000, alpha_dv=10.0, beta_tan=0.5):
    model, state = build_test_model()

    action_counts = Counter()
    norms = []
    axis_counts = Counter()  # 'R', 'T', 'N', 'zero'

    for _ in range(num_samples):
        a = model.rollout_policy(state, alpha_dv=alpha_dv, beta_tan=beta_tan)
        actions = model.actions(state)
        # find index in the fixed action list
        idx = next(i for i, cand in enumerate(actions) if np.allclose(cand, a))

        action_counts[idx] += 1

        dv_norm = np.linalg.norm(a)
        norms.append(dv_norm)

        if dv_norm == 0.0:
            axis_counts["zero"] += 1
        else:
            dominant_axis = int(np.argmax(np.abs(a)))
            if dominant_axis == 0:
                axis_counts["R"] += 1
            elif dominant_axis == 1:
                axis_counts["T"] += 1
            else:
                axis_counts["N"] += 1

    # --- Print stats ---
    print(f"Total samples: {num_samples}")
    print("Action index counts:", action_counts)
    print("Axis category counts:", axis_counts)
    print("Mean |dv|:", np.mean(norms))
    print("Fraction zero dv:", axis_counts["zero"] / num_samples)

    # --- Bar plot of action frequencies ---
    indices = sorted(action_counts.keys())
    freqs = [action_counts[i] for i in indices]

    plt.figure()
    plt.bar(indices, freqs)
    plt.xlabel("Action index")
    plt.ylabel("Count")
    plt.title(f"Rollout action distribution (alpha={alpha_dv}, beta={beta_tan})")
    plt.show()

if __name__ == "__main__":
    main()
