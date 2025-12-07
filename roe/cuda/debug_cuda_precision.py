import numpy as np
from roe.propagation import ROEDynamics, map_roe_to_rtn
from roe.dynamics import apply_impulsive_dv
from roe.cuda.cuda_roe_wrapper import batch_propagate_roe_cuda

# Test single action with detailed output
a_chief = 6778.0
e_chief = 0.001
i_chief = np.deg2rad(45.0)
omega_chief = np.deg2rad(0.0)
n_chief = np.sqrt(398600.4418 / a_chief**3)

roe_initial = np.array([0.0, 0.0, 0.0, 50.0, 0.0, 0.0], dtype=np.float64)
dv_action = np.array([0.01, 0.0, 0.0]) * 1000.0  # 10 m/s in R direction

t_burn = 0.0
dt = 100.0

print("="*80)
print("DETAILED PRECISION DEBUG")
print("="*80)
print(f"\nInitial ROE (float64): {roe_initial}")
print(f"Delta-v [m/s]: {dv_action/1000.0}")

# CPU calculation (float64)
dyn_model = ROEDynamics(a_chief, e_chief, i_chief, omega_chief)
roe_after_dv = apply_impulsive_dv(
    roe_initial, dv_action, a_chief, n_chief, np.array([t_burn]),
    e=e_chief, i=i_chief, omega=omega_chief
)
print(f"\nCPU ROE after dv (float64): {roe_after_dv}")

roe_propagated = dyn_model.propagate(roe_after_dv, dt, second_order=True)
print(f"CPU ROE propagated (float64): {roe_propagated}")

t_final = t_burn + dt
f_final = n_chief * t_final
pos_cpu, _ = map_roe_to_rtn(roe_propagated, a_chief, n_chief, f=f_final, omega=omega_chief)
print(f"CPU Position [km] (float64): {pos_cpu}")
print(f"CPU Position [m] (float64): {pos_cpu * 1000.0}")

# CUDA calculation (float32)
dv_actions = np.array([dv_action/1000.0], dtype=np.float32)  # Back to m/s for CUDA
roe_cuda, pos_cuda = batch_propagate_roe_cuda(
    roe_initial.astype(np.float32),
    dv_actions,
    a_chief, e_chief, i_chief, omega_chief, n_chief, t_burn, dt
)

print(f"\nCUDA ROE propagated (float32): {roe_cuda[0]}")
print(f"CUDA Position [km] (float32): {pos_cuda[0]}")
print(f"CUDA Position [m] (float32): {pos_cuda[0] * 1000.0}")

# Differences
roe_diff = np.linalg.norm((roe_propagated.astype(np.float32) - roe_cuda[0]))
pos_diff_km = np.linalg.norm((pos_cpu.astype(np.float32) - pos_cuda[0]))
pos_diff_m = pos_diff_km * 1000.0

print("\n" + "="*80)
print("DIFFERENCES (CPU float64 cast to float32 vs CUDA float32)")
print("="*80)
print(f"ROE difference norm: {roe_diff:.10e}")
print(f"Position difference [km]: {pos_diff_km:.10e}")
print(f"Position difference [m]: {pos_diff_m:.10e}")

# Component-wise position differences
print(f"\nPosition differences by component [m]:")
print(f"  R: {(pos_cpu[0] - pos_cuda[0]) * 1000.0:.6f}")
print(f"  T: {(pos_cpu[1] - pos_cuda[0][1]) * 1000.0:.6f}")  
print(f"  N: {(pos_cpu[2] - pos_cuda[0][2]) * 1000.0:.6f}")

# Check intermediate values
print("\n" + "="*80)
print("INTERMEDIATE VALUES COMPARISON")
print("="*80)

# True anomaly
f_burn_cpu = n_chief * t_burn
f_final_cpu = n_chief * t_final
print(f"f_burn: {f_burn_cpu:.10f} rad")
print(f"f_final: {f_final_cpu:.10f} rad")

# Argument of latitude
u_final = omega_chief + f_final_cpu
print(f"u_final (omega + f): {u_final:.10f} rad")
print(f"sin(u_final): {np.sin(u_final):.10f}")
print(f"cos(u_final): {np.cos(u_final):.10f}")

# Position calculation components
print(f"\nPosition calculation (CPU float64):")
print(f"  a_chief * dl = {a_chief * roe_propagated[1]:.6f} km")
print(f"  2*a*sin(u)*dex = {2.0 * a_chief * np.sin(u_final) * roe_propagated[2]:.6f} km")
print(f"  -2*a*cos(u)*dey = {-2.0 * a_chief * np.cos(u_final) * roe_propagated[3]:.6f} km")

