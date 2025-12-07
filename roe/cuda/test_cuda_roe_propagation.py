#!/usr/bin/env python3
"""
Test CUDA-accelerated ROE (Relative Orbital Elements) propagation.
Compares CPU vs CUDA implementations for correctness and performance.
"""

import numpy as np
import time
import sys

from roe.propagation import ROEDynamics
from roe.dynamics import apply_impulsive_dv
from roe.cuda.cuda_roe_wrapper import batch_propagate_roe_cuda, is_cuda_available

def test_correctness():
    """Verify CUDA propagation produces identical results to CPU version"""
    print("="*80)
    print("CORRECTNESS TEST: CUDA vs CPU ROE Propagation")
    print("="*80)

    # Check CUDA availability
    if not is_cuda_available():
        print("\n❌ CUDA ROE propagation library not available")
        print("   Make sure libroe_propagation.so is compiled in roe/cuda/")
        return False

    # Orbital parameters (matching training configuration)
    a_chief = 6778.0  # km
    e_chief = 0.001
    i_chief = np.deg2rad(45.0)
    omega_chief = np.deg2rad(0.0)
    n_chief = np.sqrt(398600.4418 / a_chief**3)

    # Initial ROE state
    roe_initial = np.array([0.0, 0.0, 0.0, 50.0, 0.0, 0.0], dtype=np.float64)

    # Generate test actions (13 actions: no-op + 6 axes × 2 magnitudes)
    delta_v_small = 0.01
    delta_v_large = 0.05
    actions = [np.zeros(3)]
    for axis in range(3):
        for mag in [delta_v_small, delta_v_large]:
            e = np.zeros(3)
            e[axis] = mag
            actions.append(e.copy())
            actions.append(-e.copy())

    # Actions are in km/s (dimensionless small values), convert to m/s
    dv_actions = np.array(actions, dtype=np.float64) * 1000.0

    # Simulation parameters
    t_burn = 0.0
    dt = 100.0

    print(f"\nTest Parameters:")
    print(f"  Chief orbit: a={a_chief:.1f} km, e={e_chief}, i={np.rad2deg(i_chief):.1f}°")
    print(f"  Initial ROE: {roe_initial}")
    print(f"  Number of actions: {len(actions)}")
    print(f"  Time step: {dt:.1f} s")

    # ===== CPU PROPAGATION =====
    print(f"\n1. Running CPU propagation ({len(actions)} actions)...")
    cpu_roes = []
    cpu_positions = []

    dyn_model = ROEDynamics(a_chief, e_chief, i_chief, omega_chief)

    for action in actions:
        # Apply impulsive dv
        roe_after_dv = apply_impulsive_dv(
            roe_initial, action * 1000.0, a_chief, n_chief, np.array([t_burn]),
            e=e_chief, i=i_chief, omega=omega_chief
        )

        # Propagate
        roe_propagated = dyn_model.propagate(roe_after_dv, dt, second_order=True)

        # Map to RTN
        from roe.propagation import map_roe_to_rtn
        t_final = t_burn + dt
        f_final = n_chief * t_final
        pos_rtn, _ = map_roe_to_rtn(roe_propagated, a_chief, n_chief, f=f_final, omega=omega_chief)

        cpu_roes.append(roe_propagated)
        cpu_positions.append(pos_rtn)

    cpu_roes = np.array(cpu_roes)
    cpu_positions = np.array(cpu_positions)

    # ===== CUDA PROPAGATION =====
    print(f"2. Running CUDA propagation ({len(actions)} actions)...")

    cuda_roes, cuda_positions = batch_propagate_roe_cuda(
        roe_initial,  # Already float64
        dv_actions,
        a_chief,
        e_chief,
        i_chief,
        omega_chief,
        n_chief,
        t_burn,
        dt
    )

    # ===== COMPARISON =====
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    max_roe_diff = 0.0
    max_pos_diff = 0.0

    print(f"\n{'Action':<8} {'ROE Diff (m)':<20} {'Position Diff (m)':<20}")
    print("-"*50)

    for i in range(len(actions)):
        # ROE difference (convert to meters for semi-major axis component)
        roe_diff_km = np.linalg.norm(cpu_roes[i] - cuda_roes[i])
        roe_diff_m = roe_diff_km * 1000.0  # Most components are dimensionless, but da is in km

        # Position difference
        pos_diff_km = np.linalg.norm(cpu_positions[i] - cuda_positions[i])
        pos_diff_m = pos_diff_km * 1000.0

        max_roe_diff = max(max_roe_diff, roe_diff_m)
        max_pos_diff = max(max_pos_diff, pos_diff_m)

        print(f"{i:<8} {roe_diff_m:<20.4e} {pos_diff_m:<20.4e}")

    print("-"*50)
    print(f"{'MAX:':<8} {max_roe_diff:<20.4e} {max_pos_diff:<20.4e}")

    # Pass/Fail criteria (float64 has ~15 significant digits)
    # For orbital mechanics with float64 (double precision):
    # ROE tolerance: 1µm (excellent agreement)
    # Position tolerance: 1mm (excellent for float64 orbital calculations)
    ROE_TOLERANCE_M = 1e-6  # 1µm tolerance for ROE
    POS_TOLERANCE_M = 1e-3  # 1mm tolerance for positions (float64 precision)

    roe_pass = max_roe_diff < ROE_TOLERANCE_M
    pos_pass = max_pos_diff < POS_TOLERANCE_M

    print("\n" + "="*80)
    if roe_pass and pos_pass:
        print("✅ CORRECTNESS TEST PASSED (float64 precision)")
        print(f"   ROE differences: {max_roe_diff:.4e} m < {ROE_TOLERANCE_M:.1e} m")
        print(f"   Position differences: {max_pos_diff:.4e} m < {POS_TOLERANCE_M:.1e} m")
    else:
        print("❌ CORRECTNESS TEST FAILED")
        if not roe_pass:
            print(f"   ROE differences: {max_roe_diff:.4e} m >= {ROE_TOLERANCE_M:.1e} m")
        if not pos_pass:
            print(f"   Position differences: {max_pos_diff:.4e} m >= {POS_TOLERANCE_M:.1e} m")
    print("="*80)

    return roe_pass and pos_pass


def test_performance():
    """Benchmark CPU vs CUDA propagation performance"""
    print("\n\n" + "="*80)
    print("PERFORMANCE BENCHMARK: CUDA vs CPU ROE Propagation")
    print("="*80)

    if not is_cuda_available():
        print("\n❌ CUDA ROE propagation library not available")
        return

    # Orbital parameters
    a_chief = 6778.0
    e_chief = 0.001
    i_chief = np.deg2rad(45.0)
    omega_chief = np.deg2rad(0.0)
    n_chief = np.sqrt(398600.4418 / a_chief**3)

    roe_initial = np.array([0.0, 0.0, 0.0, 50.0, 0.0, 0.0], dtype=np.float64)

    # Generate 13 actions
    delta_v_small = 0.01
    delta_v_large = 0.05
    actions = [np.zeros(3)]
    for axis in range(3):
        for mag in [delta_v_small, delta_v_large]:
            e = np.zeros(3)
            e[axis] = mag
            actions.append(e.copy())
            actions.append(-e.copy())

    dv_actions = np.array(actions, dtype=np.float64) * 1000.0
    num_actions = len(actions)

    t_burn = 0.0
    dt = 100.0

    print(f"\nBenchmark Parameters:")
    print(f"  Number of actions: {num_actions}")
    print(f"  Runs per method: 100")

    # Warmup CUDA
    print("\nWarming up CUDA...")
    _ = batch_propagate_roe_cuda(
        roe_initial, dv_actions,
        a_chief, e_chief, i_chief, omega_chief, n_chief, t_burn, dt
    )

    # ===== CPU BENCHMARK =====
    print("Benchmarking CPU propagation...")
    dyn_model = ROEDynamics(a_chief, e_chief, i_chief, omega_chief)
    num_runs = 100
    cpu_times = []

    for run in range(num_runs):
        start = time.time()

        for action in actions:
            roe_after_dv = apply_impulsive_dv(
                roe_initial, action * 1000.0, a_chief, n_chief, np.array([t_burn]),
                e=e_chief, i=i_chief, omega=omega_chief
            )
            roe_propagated = dyn_model.propagate(roe_after_dv, dt, second_order=True)

            from roe.propagation import map_roe_to_rtn
            t_final = t_burn + dt
            f_final = n_chief * t_final
            _, _ = map_roe_to_rtn(roe_propagated, a_chief, n_chief, f=f_final, omega=omega_chief)

        elapsed = time.time() - start
        cpu_times.append(elapsed)

    # ===== CUDA BENCHMARK =====
    print("Benchmarking CUDA propagation...")
    cuda_times = []

    for run in range(num_runs):
        start = time.time()

        _ = batch_propagate_roe_cuda(
            roe_initial, dv_actions,
            a_chief, e_chief, i_chief, omega_chief, n_chief, t_burn, dt
        )

        elapsed = time.time() - start
        cuda_times.append(elapsed)

    # ===== RESULTS =====
    cpu_mean = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    cuda_mean = np.mean(cuda_times)
    cuda_std = np.std(cuda_times)
    speedup = cpu_mean / cuda_mean

    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"\nCPU Propagation ({num_actions} actions):")
    print(f"  Mean: {cpu_mean*1000:.3f} ± {cpu_std*1000:.3f} ms")
    print(f"  Per action: {cpu_mean/num_actions*1000:.3f} ms")

    print(f"\nCUDA Propagation ({num_actions} actions):")
    print(f"  Mean: {cuda_mean*1000:.3f} ± {cuda_std*1000:.3f} ms")
    print(f"  Per action: {cuda_mean/num_actions*1000:.3f} ms")

    print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster")
    print("="*80)

    return speedup


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CUDA ROE PROPAGATION TEST SUITE")
    print("="*80)

    # Run correctness test
    correctness_passed = test_correctness()

    if not correctness_passed:
        print("\n❌ Skipping performance test due to correctness failures")
        sys.exit(1)

    # Run performance test
    speedup = test_performance()

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"✅ Correctness: PASSED")
    print(f"🚀 Performance: {speedup:.2f}x speedup over CPU")
    print("="*80)
