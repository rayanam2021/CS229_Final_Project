import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
from datetime import datetime

# Import local modules
from roe.propagation import propagateGeomROE, ROEDynamics
from roe.dynamics import apply_impulsive_dv

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return None
    with open(config_path, 'r') as f:
        return json.load(f)

def get_prototype_candidates_meters():
    """
    Returns a list of ROE state vectors defined in METERS and their labels.
    Format: [da, dl, dex, dey, dix, diy]
    """
    candidates = []
    labels = []

    # 1. Hold case
    candidates.append(np.array([0, 120, -20, 20, 20, -20]))
    labels.append("Hold_Configuration")

    # 2. Near equatorial
    candidates.append(np.array([0, 80, 20, 20, 1, 1]))
    labels.append("Near_Equatorial")
        
    # 3. Low inclination - Positive
    candidates.append(np.array([0, 80, 20, 20, 20, 20]))
    labels.append("Low_Inc_Pos")

    # 4. Low inclination - Negative
    candidates.append(np.array([0, 80, -20, -20, 20, 20]))
    labels.append("Low_Inc_Neg")
    
    # 5. Medium inclination - Positive
    candidates.append(np.array([0, 80, 20, 20, 50, 50]))
    labels.append("Med_Inc_Pos")

    # 6. Medium inclination - Negative
    candidates.append(np.array([0, 80, -20, -20, 50, 50]))
    labels.append("Med_Inc_Neg")

    # 7. High inclination - Positive
    candidates.append(np.array([0, 80, 20, 20, 80, 80]))
    labels.append("High_Inc_Pos")

    # 8. High inclination - Negative
    candidates.append(np.array([0, 80, -20, -20, 80, 80]))
    labels.append("High_Inc_Neg")
    
    return candidates, labels

def generate_mcts_actions():
    """Generates the 13 discrete actions used in the MCTS controller."""
    delta_v_small = 0.01 # m/s
    delta_v_large = 0.05 # m/s
    
    actions = []
    labels = []
    
    # 1. No-op
    actions.append(np.zeros(3))
    labels.append("No_Op")
    
    # Directions: R, T, N
    directions = ["R", "T", "N"]
    
    for axis, dir_name in enumerate(directions):
        for mag, size_name in [(delta_v_small, "Small"), (delta_v_large, "Large")]:
            # Positive
            e_pos = np.zeros(3)
            e_pos[axis] = mag
            actions.append(e_pos)
            labels.append(f"Pos_{dir_name}_{size_name}")
            
            # Negative
            e_neg = np.zeros(3)
            e_neg[axis] = -mag
            actions.append(e_neg)
            labels.append(f"Neg_{dir_name}_{size_name}")
            
    return actions, labels

def set_equal_aspect_3d(ax, x, y, z):
    """
    Sets 3D plot axes to equal scale so orbits aren't distorted.
    UPDATED: Explicitly includes (0,0,0) [The Chief] in the bounds calculation.
    """
    # Append the Chief's position (0,0,0) to the data for bounds calculation
    X = np.append(x, 0.0)
    Y = np.append(y, 0.0)
    Z = np.append(z, 0.0)

    max_range = np.array([
        np.max(X) - np.min(X),
        np.max(Y) - np.min(Y),
        np.max(Z) - np.min(Z)
    ]).max() / 2.0

    mid_x = (np.max(X) + np.min(X)) * 0.5
    mid_y = (np.max(Y) + np.min(Y)) * 0.5
    mid_z = (np.max(Z) + np.min(Z)) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def run_trajectory_tests():
    # --- 1. Load Configuration ---
    config = load_config("config.json")
    
    # Default parameters if config missing
    mu = 398600.4418
    a = 7000.0
    e = 0.001
    i = np.radians(98.0)
    omega = np.radians(30.0)
    out_base = "output"

    if config:
        print("Loaded configuration from config.json")
        orbit_conf = config['orbit']
        mu = orbit_conf.get('mu_earth', mu)
        a = orbit_conf.get('a_chief_km', a)
        e = orbit_conf.get('e_chief', e)
        i = np.radians(orbit_conf.get('i_chief_deg', 98.0))
        omega = np.radians(orbit_conf.get('omega_chief_deg', 30.0))
        out_base = config['simulation'].get('output_dir', "output")

    # --- 2. Setup Output Directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_folder = os.path.join(out_base, f"{timestamp}_trajectory_tests")
    os.makedirs(out_folder, exist_ok=True)
    print(f"Output directory: {out_folder}")

    n = np.sqrt(mu / a**3)
    period = 2 * np.pi / n
    print(f"Orbit Period: {period/60:.2f} minutes")
    
    # Initialize Dynamics Model
    dyn_model = ROEDynamics(a, e, i, omega, mu)

    # --- TASK 1: Plot All Natural Motions ---
    print("\nGenerating Natural Motion Plot (All Candidates)...")
    
    candidates_m, cand_labels = get_prototype_candidates_meters()
    
    # Propagate for 3 orbits
    t_natural = np.arange(0, period * 3, 60.0)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Chief
    ax.scatter(0, 0, 0, c='k', marker='*', s=150, label='Chief')
    
    all_x, all_y, all_z = [], [], [] # For axis scaling

    colors = plt.cm.jet(np.linspace(0, 1, len(candidates_m)))

    for idx, (roe_m, label) in enumerate(zip(candidates_m, cand_labels)):
        # Normalize to dimensionless
        roe_dimless = roe_m / (a * 1000.0)
        
        # Propagate (t0=0)
        pos_rtn, _ = propagateGeomROE(roe_dimless, a, e, i, omega, n, t_natural, t0=0.0)
        
        # Convert to meters
        x, y, z = pos_rtn[0]*1000, pos_rtn[1]*1000, pos_rtn[2]*1000
        
        ax.plot(x, y, z, label=f"C{idx}: {label}", color=colors[idx], linewidth=1.5)
        
        # Collect limits
        all_x.extend(x); all_y.extend(y); all_z.extend(z)

    # Formatting
    set_equal_aspect_3d(ax, all_x, all_y, all_z)
    ax.set_xlabel('Radial [m]')
    ax.set_ylabel('Along-Track [m]')
    ax.set_zlabel('Cross-Track [m]')
    ax.set_title(f'Natural Motion of Candidates (3 Orbits)\n{len(candidates_m)} Scenarios')
    ax.legend(loc='upper right', fontsize='small', ncol=2)
    
    save_path = os.path.join(out_folder, "00_All_Candidates_Natural_Motion.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # --- TASK 2: Action Sensitivity (Single Candidate) ---
    target_idx = 0 # Hold Configuration
    target_roe_m = candidates_m[target_idx]
    target_label = cand_labels[target_idx]
    
    print(f"\nGenerating Action Sensitivity Plots for Candidate: {target_label}...")
    
    # Get MCTS Actions
    actions, action_labels = generate_mcts_actions()
    
    # Timing for Maneuver
    t_burn = period * 0.5
    t_end = period * 1.5 # 1.5 orbits total duration
    dt_step = 30.0
    
    # --- UPDATED TIME ARRAYS TO PREVENT GAPS ---
    # Phase 1: 0 to t_burn (inclusive)
    t_seg1 = np.arange(0, t_burn + 0.001, dt_step)
    if t_seg1[-1] > t_burn + 0.001: t_seg1 = t_seg1[:-1] # Trim overshoot
    if t_seg1[-1] != t_burn: t_seg1 = np.append(t_seg1, t_burn) # Ensure t_burn is exact last point
    
    # Phase 2: t_burn to t_end (inclusive)
    t_seg2 = np.arange(t_burn, t_end + 0.001, dt_step)
    
    # Normalized Initial ROE
    roe_initial_dimless = target_roe_m / (a * 1000.0)
    
    # Propagate Phase 1 (Common for all actions)
    pos_seg1, _ = propagateGeomROE(roe_initial_dimless, a, e, i, omega, n, t_seg1, t0=0.0)
    x1, y1, z1 = pos_seg1[0]*1000, pos_seg1[1]*1000, pos_seg1[2]*1000
    
    # Get Pre-Burn ROE State (Exact) at t_burn
    roe_pre_burn = dyn_model.propagate(roe_initial_dimless, t_burn, second_order=True)

    for i, (action, act_label) in enumerate(zip(actions, action_labels)):
        
        # Apply Impulse
        # Pass t=[t_burn] for correct Mean Anomaly calc in GVEs
        roe_post_burn = apply_impulsive_dv(
            roe_pre_burn, action, a, n, np.array([t_burn]), 
            e=e, i=i, omega=omega
        )
        
        # Propagate Phase 2 (Post-Burn)
        pos_seg2, _ = propagateGeomROE(roe_post_burn, a, e, i, omega, n, t_seg2, t0=t_burn)
        x2, y2, z2 = pos_seg2[0]*1000, pos_seg2[1]*1000, pos_seg2[2]*1000
        
        # Also propagate "Ghost" (No Burn) for comparison
        pos_ghost, _ = propagateGeomROE(roe_pre_burn, a, e, i, omega, n, t_seg2, t0=t_burn)
        xg, yg, zg = pos_ghost[0]*1000, pos_ghost[1]*1000, pos_ghost[2]*1000
        
        # --- Plotting ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Chief
        ax.scatter(0, 0, 0, c='k', marker='*', s=150, label='Chief')
        
        # Plot Phase 1 (Blue)
        ax.plot(x1, y1, z1, color='blue', linewidth=2, label='Pre-Maneuver', alpha=0.6)
        
        # Plot Phase 2 Ghost (Dotted Blue)
        ax.plot(xg, yg, zg, color='blue', linestyle=':', linewidth=1, label='Natural Path (No Burn)')
        
        # Plot Phase 2 Actual (Red)
        ax.plot(x2, y2, z2, color='red', linewidth=2, label='Post-Maneuver')
        
        # Mark Burn (Use the last point of seg 1)
        ax.scatter(x1[-1], y1[-1], z1[-1], c='orange', s=100, marker='^', label=f'Burn: {act_label}', zorder=10)
        
        # Formatting
        combine_x = np.concatenate([x1, x2, xg])
        combine_y = np.concatenate([y1, y2, yg])
        combine_z = np.concatenate([z1, z2, zg])
        
        # Ensure aspect ratio includes the CHIEF (0,0,0)
        set_equal_aspect_3d(ax, combine_x, combine_y, combine_z)
        
        ax.set_xlabel('R [m]')
        ax.set_ylabel('T [m]')
        ax.set_zlabel('N [m]')
        
        dv_str = f"[{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}] m/s"
        ax.set_title(f"Action {i}: {act_label}\nVector: {dv_str}")
        ax.legend(loc='upper right')
        
        file_name = f"Action_{i:02d}_{act_label}.png"
        plt.savefig(os.path.join(out_folder, file_name))
        plt.close(fig)
        print(f"  Saved plot: {file_name}")

    print("\nDone!")

if __name__ == "__main__":
    # If run standalone, use default params
    run_trajectory_tests()