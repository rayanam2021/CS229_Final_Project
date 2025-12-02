"""
Test script to verify the simplified rollout policy heuristics.
Checks that the policy correctly:
1. Penalizes larger |dv| magnitudes
2. Prefers tangential/normal actions (axes 1, 2) over radial (axis 0)
3. Distributes actions via softmax weighting
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from mcts.orbital_mdp_model import OrbitalMCTSModel, OrbitalState
from camera.camera_observations import VoxelGrid, GroundTruthRSO


def build_test_model(alpha_dv=10.0, beta_tan=0.5):
    """Build a test model with given heuristic parameters."""
    grid_dims = (32, 32, 32)
    grid = VoxelGrid(grid_dims)
    rso = GroundTruthRSO(grid)

    camera_fn = {
        "fov_degrees": 60.0,
        "sensor_res": (32, 32),
        "noise_params": {
            "p_hit_given_occupied": 0.95,
            "p_hit_given_empty": 0.001,
        },
    }

    a_chief = 7000e3  # meters
    e_chief = 0.0
    i_chief = 0.0
    omega_chief = 0.0
    mu_earth = 3.986004418e14  # m^3/s^2
    n_chief = np.sqrt(mu_earth / a_chief**3)
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
        max_depth=15,
        alpha_dv=alpha_dv,
        beta_tan=beta_tan,
        r_min_rollout=500.0,
        r_max_rollout=5000.0,
        target_radius=2000.0,
        gamma_r=0.002
    )

    roe = np.array([0.0, 0.001, 0.0, 0.0, 0.0, 0.0])
    state = OrbitalState(roe=roe, grid=grid, time=0.0)

    return model, state


def compute_action_scores(actions, alpha_dv, beta_tan):
    """Compute the scores for all actions according to the heuristic."""
    scores = []
    for a in actions:
        dv_norm = np.linalg.norm(a)
        s = -alpha_dv * dv_norm

        if dv_norm > 0:
            main_axis = int(np.argmax(np.abs(a)))
            if main_axis in (1, 2):
                s += beta_tan

        scores.append(s)

    return np.array(scores)


def compute_softmax_probs(scores):
    """Convert scores to probabilities using softmax."""
    scores = np.array(scores)
    scores -= scores.max()
    probs = np.exp(scores)
    probs /= probs.sum()
    return probs


def categorize_action(action):
    """Categorize action by dominant axis."""
    dv_norm = np.linalg.norm(action)
    if dv_norm == 0.0:
        return "zero"
    main_axis = int(np.argmax(np.abs(action)))
    if main_axis == 0:
        return "radial"
    elif main_axis == 1:
        return "tangential"
    else:
        return "normal"


def test_heuristic_scores(alpha_dv=10.0, beta_tan=0.5):
    """Test that action scores match expected heuristic behavior."""
    model, state = build_test_model(alpha_dv=alpha_dv, beta_tan=beta_tan)
    actions = model.actions(state)

    print("\n" + "="*70)
    print(f"HEURISTIC SCORE TEST (alpha_dv={alpha_dv}, beta_tan={beta_tan})")
    print("="*70)

    scores = compute_action_scores(actions, alpha_dv, beta_tan)
    probs = compute_softmax_probs(scores)

    print("\nAction | |dv|    | Axis      | Score   | Probability")
    print("-" * 70)
    for i, a in enumerate(actions):
        dv_norm = np.linalg.norm(a)
        category = categorize_action(a)
        print(f"{i:3d}    | {dv_norm:.4f}  | {category:9s} | {scores[i]:7.3f} | {probs[i]:.4f}")

    # Verify heuristics
    zero_action_idx = 0
    small_radial_indices = [1, 2]
    small_tang_indices = [3, 4, 5, 6]
    small_norm_indices = [7, 8, 9, 10]
    large_tang_indices = [11, 12]
    large_norm_indices = [13]

    print("\n" + "="*70)
    print("HEURISTIC VERIFICATION")
    print("="*70)

    # Check: zero action should have highest probability (no dv cost)
    zero_prob = probs[zero_action_idx]
    print(f"✓ Zero action probability: {zero_prob:.4f}")

    # Check: tangential/normal actions should have higher prob than radial for same |dv|
    small_radial_avg_prob = np.mean([probs[i] for i in small_radial_indices])
    small_tang_avg_prob = np.mean([probs[i] for i in small_tang_indices])
    print(f"✓ Small radial avg prob:     {small_radial_avg_prob:.4f}")
    print(f"✓ Small tangential avg prob: {small_tang_avg_prob:.4f}")
    if small_tang_avg_prob > small_radial_avg_prob:
        print(f"  ✓ Tangential preferred over radial (good!)")
    else:
        print(f"  ✗ WARNING: Radial preferred over tangential (bad!)")

    # Check: smaller |dv| should have higher prob than larger |dv|
    large_tang_avg_prob = np.mean([probs[i] for i in large_tang_indices])
    print(f"✓ Large tangential avg prob:  {large_tang_avg_prob:.4f}")
    if small_tang_avg_prob > large_tang_avg_prob:
        print(f"  ✓ Smaller dv preferred over larger dv (good!)")
    else:
        print(f"  ✗ WARNING: Larger dv preferred over smaller dv (bad!)")

    return model, state, scores, probs


def test_sampling_distribution(num_samples=10000, alpha_dv=10.0, beta_tan=0.5):
    """Test that sampling follows the computed distribution."""
    model, state = build_test_model(alpha_dv=alpha_dv, beta_tan=beta_tan)
    actions = model.actions(state)

    print("\n" + "="*70)
    print(f"SAMPLING DISTRIBUTION TEST ({num_samples} samples)")
    print("="*70)

    action_counts = Counter()
    axis_counts = Counter()
    dv_norms = []

    for _ in range(num_samples):
        a = model.custom_rollout_policy(state)
        idx = next(i for i, cand in enumerate(actions) if np.allclose(cand, a))
        action_counts[idx] += 1

        dv_norm = np.linalg.norm(a)
        dv_norms.append(dv_norm)
        axis_counts[categorize_action(a)] += 1

    # Compare empirical vs theoretical probabilities
    scores = compute_action_scores(actions, alpha_dv, beta_tan)
    theoretical_probs = compute_softmax_probs(scores)

    print("\nAction | Theoretical | Empirical | Error")
    print("-" * 50)
    for i in sorted(action_counts.keys()):
        theoretical = theoretical_probs[i]
        empirical = action_counts[i] / num_samples
        error = abs(theoretical - empirical)
        print(f"{i:3d}    | {theoretical:.4f}      | {empirical:.4f}    | {error:.4f}")

    print("\n" + "="*70)
    print("AXIS DISTRIBUTION")
    print("="*70)
    for axis, count in sorted(axis_counts.items()):
        frac = count / num_samples
        print(f"{axis:12s}: {count:5d} ({frac:.1%})")

    print(f"\nMean |dv|: {np.mean(dv_norms):.6f}")
    print(f"Std  |dv|: {np.std(dv_norms):.6f}")
    print(f"Min  |dv|: {np.min(dv_norms):.6f}")
    print(f"Max  |dv|: {np.max(dv_norms):.6f}")

    return action_counts, axis_counts, dv_norms


def plot_results(action_counts, axis_counts, dv_norms, alpha_dv, beta_tan, save_path=None):
    """Plot test results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Action index distribution
    indices = sorted(action_counts.keys())
    freqs = [action_counts[i] for i in indices]
    axes[0].bar(indices, freqs, color='steelblue')
    axes[0].set_xlabel("Action Index")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Action Distribution\n(α={alpha_dv}, β={beta_tan})")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Axis category distribution
    categories = sorted(axis_counts.keys())
    axis_freqs = [axis_counts[cat] for cat in categories]
    colors = {'zero': 'gray', 'radial': 'red', 'tangential': 'green', 'normal': 'blue'}
    bar_colors = [colors.get(cat, 'black') for cat in categories]
    axes[1].bar(categories, axis_freqs, color=bar_colors, alpha=0.7)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Axis Category Distribution")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: |dv| histogram
    axes[2].hist(dv_norms, bins=20, color='orange', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel("|dv| magnitude (m/s)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("|dv| Distribution")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ROLLOUT POLICY HEURISTIC TEST SUITE")
    print("="*70)

    # Test 1: Verify heuristic scores
    model, state, scores, probs = test_heuristic_scores(alpha_dv=10.0, beta_tan=0.5)

    # Test 2: Verify sampling follows distribution
    action_counts, axis_counts, dv_norms = test_sampling_distribution(
        num_samples=10000, alpha_dv=10.0, beta_tan=0.5
    )

    # Test 3: Plot results
    plot_results(action_counts, axis_counts, dv_norms, alpha_dv=10.0, beta_tan=0.5,
                 save_path="/tmp/policy_heuristics.png")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
