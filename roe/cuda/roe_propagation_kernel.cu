/**
 * CUDA Kernel for Batched ROE (Relative Orbital Elements) Propagation
 *
 * Implements parallel propagation of multiple ROE states through:
 * 1. Impulsive delta-v application (Gauss Variational Equations)
 * 2. Linear propagation (State Transition Matrix)
 * 3. Second-order corrections
 * 4. RTN position mapping
 *
 * Based on Willis "Analytical Theory of Satellite Relative Motion" (2023)
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Device function: Apply impulsive delta-v to ROE state
 * Maps RTN velocity change to ROE change using control matrix (Gauss Variational Equations)
 *
 * @param roe_in Input ROE state [da, dlambda, dex, dey, dix, diy]
 * @param dv_rtn Delta-v in RTN frame [m/s]
 * @param roe_out Output ROE state after delta-v
 * @param a_chief Chief semi-major axis [km]
 * @param e_chief Eccentricity
 * @param omega Argument of perigee [rad]
 * @param n Mean motion [rad/s]
 * @param f_burn True anomaly at burn time [rad]
 */
__device__ void apply_impulsive_dv_device(
    const double* roe_in,
    const double* dv_rtn,
    double* roe_out,
    double a_chief,
    double e_chief,
    double omega,
    double n,
    double f_burn
) {
    // Convert dv from m/s to km/s
    double dv_r = dv_rtn[0] / 1000.0;
    double dv_t = dv_rtn[1] / 1000.0;
    double dv_n = dv_rtn[2] / 1000.0;

    // Argument of latitude
    double theta = omega + f_burn;
    double st = sin(theta);
    double ct = cos(theta);
    double sf = sin(f_burn);

    // Normalization factor
    double inv_na = 1.0f / (n * a_chief);

    // Control matrix B (scaled by inv_na)
    // Delta ROE = B * dv_rtn

    // da = 2 * dv_t * inv_na
    double delta_da = 2.0 * dv_t * inv_na;

    // dlambda = -2 * dv_r * inv_na - 3 * e * sin(f) * dv_t * inv_na
    double delta_dlambda = (-2.0 * dv_r - 3.0 * e_chief * sf * dv_t) * inv_na;

    // dex = sin(theta) * dv_r * inv_na + 2 * cos(theta) * dv_t * inv_na
    double delta_dex = (st * dv_r + 2.0 * ct * dv_t) * inv_na;

    // dey = -cos(theta) * dv_r * inv_na + 2 * sin(theta) * dv_t * inv_na
    double delta_dey = (-ct * dv_r + 2.0 * st * dv_t) * inv_na;

    // dix = cos(theta) * dv_n * inv_na
    double delta_dix = ct * dv_n * inv_na;

    // diy = sin(theta) * dv_n * inv_na
    double delta_diy = st * dv_n * inv_na;

    // Apply changes
    roe_out[0] = roe_in[0] + delta_da;
    roe_out[1] = roe_in[1] + delta_dlambda;
    roe_out[2] = roe_in[2] + delta_dex;
    roe_out[3] = roe_in[3] + delta_dey;
    roe_out[4] = roe_in[4] + delta_dix;
    roe_out[5] = roe_in[5] + delta_diy;
}

/**
 * Device function: Propagate ROE state forward in time
 * Uses State Transition Matrix (STM) with second-order correction
 *
 * @param roe_in Input ROE state
 * @param roe_out Output propagated ROE state
 * @param dt Time step [s]
 * @param n Mean motion [rad/s]
 */
__device__ void propagate_roe_device(
    const double* roe_in,
    double* roe_out,
    double dt,
    double n
) {
    // State Transition Matrix (STM) - mostly identity with drift term
    // Phi[1,0] = -1.5 * n * dt (linear drift of relative mean longitude)

    roe_out[0] = roe_in[0];  // da unchanged
    roe_out[1] = roe_in[1] - 1.5 * n * dt * roe_in[0];  // dlambda with drift
    roe_out[2] = roe_in[2];  // dex unchanged
    roe_out[3] = roe_in[3];  // dey unchanged
    roe_out[4] = roe_in[4];  // dix unchanged
    roe_out[5] = roe_in[5];  // diy unchanged

    // Second-order correction (Willis Eq 3.41)
    double da = roe_in[0];
    double second_order_term = (15.0 / 8.0) * n * dt * (da * da);
    roe_out[1] += second_order_term;
}

/**
 * Device function: Map ROE to RTN position
 * Standard linear mapping (Willis Eq 2.45)
 *
 * @param roe ROE state
 * @param pos_rtn Output position in RTN frame [km]
 * @param a_chief Chief semi-major axis [km]
 * @param omega Argument of perigee [rad]
 * @param f True anomaly [rad]
 */
__device__ void map_roe_to_rtn_device(
    const double* roe,
    double* pos_rtn,
    double a_chief,
    double omega,
    double f
) {
    double da = roe[0];
    double dl = roe[1];
    double dex = roe[2];
    double dey = roe[3];
    double dix = roe[4];
    double diy = roe[5];

    double u = omega + f;
    double su = sin(u);
    double cu = cos(u);

    // Relative position (linear mapping)
    pos_rtn[0] = -a_chief * cu * dex - a_chief * su * dey + a_chief * da;  // r_r
    pos_rtn[1] = a_chief * dl + 2.0 * a_chief * su * dex - 2.0 * a_chief * cu * dey;  // r_t
    pos_rtn[2] = a_chief * su * dix - a_chief * cu * diy;  // r_n
}

/**
 * CUDA Kernel: Batch propagate multiple ROE states with different delta-v maneuvers
 *
 * Each thread handles one action (delta-v vector):
 * 1. Apply impulsive delta-v
 * 2. Propagate forward in time
 * 3. Map to RTN position
 *
 * @param roe_initial Initial ROE state [6] (shared across all threads)
 * @param dv_actions Array of delta-v actions [num_actions x 3]
 * @param roe_final Output propagated ROE states [num_actions x 6]
 * @param positions_rtn Output RTN positions [num_actions x 3] in km
 * @param num_actions Number of actions to process
 * @param a_chief Chief semi-major axis [km]
 * @param e_chief Eccentricity
 * @param i_chief Inclination [rad]
 * @param omega Argument of perigee [rad]
 * @param n Mean motion [rad/s]
 * @param t_burn Burn time [s]
 * @param dt Propagation time step [s]
 */
__global__ void batch_propagate_roe_kernel(
    const double* roe_initial,     // [6]
    const double* dv_actions,      // [num_actions x 3]
    double* roe_final,             // [num_actions x 6]
    double* positions_rtn,         // [num_actions x 3]
    int num_actions,
    double a_chief,
    double e_chief,
    double i_chief,
    double omega,
    double n,
    double t_burn,
    double dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_actions) return;

    // Temporary ROE states
    double roe_after_burn[6];
    double roe_propagated[6];

    // Calculate true anomaly at burn time (M ~ f for near-circular)
    double f_burn = n * t_burn;

    // 1. Apply impulsive delta-v
    const double* dv = &dv_actions[idx * 3];
    apply_impulsive_dv_device(
        roe_initial,
        dv,
        roe_after_burn,
        a_chief,
        e_chief,
        omega,
        n,
        f_burn
    );

    // 2. Propagate forward by dt
    propagate_roe_device(
        roe_after_burn,
        roe_propagated,
        dt,
        n
    );

    // 3. Store final ROE state
    double* out_roe = &roe_final[idx * 6];
    for (int i = 0; i < 6; i++) {
        out_roe[i] = roe_propagated[i];
    }

    // 4. Map to RTN position at final time
    double t_final = t_burn + dt;
    double f_final = n * t_final;
    double* out_pos = &positions_rtn[idx * 3];
    map_roe_to_rtn_device(
        roe_propagated,
        out_pos,
        a_chief,
        omega,
        f_final
    );
}

/**
 * Host wrapper function for CUDA kernel
 *
 * Allocates GPU memory, launches kernel, and copies results back
 */
extern "C" {

void batch_propagate_roe_cuda(
    const double* roe_initial_host,     // [6]
    const double* dv_actions_host,      // [num_actions x 3]
    double* roe_final_host,             // [num_actions x 6]
    double* positions_rtn_host,         // [num_actions x 3]
    int num_actions,
    double a_chief,
    double e_chief,
    double i_chief,
    double omega,
    double n,
    double t_burn,
    double dt
) {
    // Allocate device memory
    double *d_roe_initial, *d_dv_actions, *d_roe_final, *d_positions;

    CUDA_CHECK(cudaMalloc(&d_roe_initial, 6 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dv_actions, num_actions * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_roe_final, num_actions * 6 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_positions, num_actions * 3 * sizeof(double)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_roe_initial, roe_initial_host, 6 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dv_actions, dv_actions_host, num_actions * 3 * sizeof(double), cudaMemcpyHostToDevice));

    // Launch kernel
    int block_size = 256;
    int num_blocks = (num_actions + block_size - 1) / block_size;

    batch_propagate_roe_kernel<<<num_blocks, block_size>>>(
        d_roe_initial,
        d_dv_actions,
        d_roe_final,
        d_positions,
        num_actions,
        a_chief,
        e_chief,
        i_chief,
        omega,
        n,
        t_burn,
        dt
    );

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(roe_final_host, d_roe_final, num_actions * 6 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(positions_rtn_host, d_positions, num_actions * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_roe_initial);
    cudaFree(d_dv_actions);
    cudaFree(d_roe_final);
    cudaFree(d_positions);
}

} // extern "C"
