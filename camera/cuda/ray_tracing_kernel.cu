#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define EPSILON 1e-9f
#define MAX_STEPS 120

// CUDA kernel for DDA ray marching
__global__ void dda_ray_march_kernel(
    const float* __restrict__ ray_origins,    // (N, 3)
    const float* __restrict__ ray_dirs,       // (N, 3)
    const bool* __restrict__ rso_shape,       // (grid_x, grid_y, grid_z)
    const float* __restrict__ grid_origin,    // (3,)
    const float voxel_size,
    const int grid_x,
    const int grid_y,
    const int grid_z,
    const float p_hit_occ,
    const float p_hit_emp,
    const unsigned long long seed,
    int* __restrict__ hit_coords,             // (N, 3) - output
    int* __restrict__ hit_count,              // (N,) - output (1 if hit, 0 otherwise)
    int* __restrict__ miss_coords,            // (N * MAX_STEPS, 3) - output
    int* __restrict__ miss_counts,            // (N,) - output
    const int num_rays
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ray_idx >= num_rays) return;

    // Initialize random number generator for this thread
    curandState state;
    curand_init(seed, ray_idx, 0, &state);

    // Load ray data
    float ox = ray_origins[ray_idx * 3 + 0];
    float oy = ray_origins[ray_idx * 3 + 1];
    float oz = ray_origins[ray_idx * 3 + 2];

    float dx = ray_dirs[ray_idx * 3 + 0];
    float dy = ray_dirs[ray_idx * 3 + 1];
    float dz = ray_dirs[ray_idx * 3 + 2];

    // Safe division
    if (fabsf(dx) < EPSILON) dx = EPSILON;
    if (fabsf(dy) < EPSILON) dy = EPSILON;
    if (fabsf(dz) < EPSILON) dz = EPSILON;

    // Grid bounds
    float gx = grid_origin[0];
    float gy = grid_origin[1];
    float gz = grid_origin[2];
    float gx_max = gx + grid_x * voxel_size;
    float gy_max = gy + grid_y * voxel_size;
    float gz_max = gz + grid_z * voxel_size;

    // Compute entry/exit points for grid (ray-box intersection)
    float t1x = (gx - ox) / dx;
    float t2x = (gx_max - ox) / dx;
    float t1y = (gy - oy) / dy;
    float t2y = (gy_max - oy) / dy;
    float t1z = (gz - oz) / dz;
    float t2z = (gz_max - oz) / dz;

    float tnx = fminf(t1x, t2x);
    float txx = fmaxf(t1x, t2x);
    float tny = fminf(t1y, t2y);
    float txy = fmaxf(t1y, t2y);
    float tnz = fminf(t1z, t2z);
    float txz = fmaxf(t1z, t2z);

    float t_enter = fmaxf(fmaxf(tnx, tny), tnz);
    float t_exit = fminf(fminf(txx, txy), txz);

    // Check if ray misses grid
    if (t_enter > t_exit || t_exit < 0) {
        hit_count[ray_idx] = 0;
        miss_counts[ray_idx] = 0;
        return;
    }

    // Start point
    float start_x, start_y, start_z;
    if (t_enter < 0) {
        start_x = ox;
        start_y = oy;
        start_z = oz;
    } else {
        start_x = ox + dx * t_enter;
        start_y = oy + dy * t_enter;
        start_z = oz + dz * t_enter;
    }

    // Convert to grid coordinates
    int curr_x = (int)floorf((start_x - gx) / voxel_size);
    int curr_y = (int)floorf((start_y - gy) / voxel_size);
    int curr_z = (int)floorf((start_z - gz) / voxel_size);

    // DDA setup
    int step_x = (dx > 0) ? 1 : -1;
    int step_y = (dy > 0) ? 1 : -1;
    int step_z = (dz > 0) ? 1 : -1;

    float t_delta_x = fabsf(voxel_size / dx);
    float t_delta_y = fabsf(voxel_size / dy);
    float t_delta_z = fabsf(voxel_size / dz);

    float bound_x = gx + (curr_x + (step_x > 0 ? 1 : 0)) * voxel_size;
    float bound_y = gy + (curr_y + (step_y > 0 ? 1 : 0)) * voxel_size;
    float bound_z = gz + (curr_z + (step_z > 0 ? 1 : 0)) * voxel_size;

    float t_max_x = (bound_x - ox) / dx;
    float t_max_y = (bound_y - oy) / dy;
    float t_max_z = (bound_z - oz) / dz;

    // DDA march
    int miss_offset = ray_idx * MAX_STEPS;
    int local_miss_count = 0;
    bool found_hit = false;

    for (int step = 0; step < MAX_STEPS; step++) {
        // Check bounds
        if (curr_x < 0 || curr_x >= grid_x ||
            curr_y < 0 || curr_y >= grid_y ||
            curr_z < 0 || curr_z >= grid_z) {
            break;
        }

        // Get voxel index (row-major order: z changes fastest)
        int voxel_idx = curr_x * (grid_y * grid_z) + curr_y * grid_z + curr_z;
        bool is_occupied = rso_shape[voxel_idx];

        // Determine hit probability based on occupancy
        float hit_prob = is_occupied ? p_hit_occ : p_hit_emp;
        float random_val = curand_uniform(&state);

        bool is_hit = random_val < hit_prob;

        if (is_hit) {
            // Store hit
            hit_coords[ray_idx * 3 + 0] = curr_x;
            hit_coords[ray_idx * 3 + 1] = curr_y;
            hit_coords[ray_idx * 3 + 2] = curr_z;
            hit_count[ray_idx] = 1;
            found_hit = true;
            break;
        } else {
            // Store miss
            if (local_miss_count < MAX_STEPS) {
                miss_coords[(miss_offset + local_miss_count) * 3 + 0] = curr_x;
                miss_coords[(miss_offset + local_miss_count) * 3 + 1] = curr_y;
                miss_coords[(miss_offset + local_miss_count) * 3 + 2] = curr_z;
                local_miss_count++;
            }
        }

        // Advance to next voxel (DDA step)
        if (t_max_x < t_max_y) {
            if (t_max_x < t_max_z) {
                t_max_x += t_delta_x;
                curr_x += step_x;
            } else {
                t_max_z += t_delta_z;
                curr_z += step_z;
            }
        } else {
            if (t_max_y < t_max_z) {
                t_max_y += t_delta_y;
                curr_y += step_y;
            } else {
                t_max_z += t_delta_z;
                curr_z += step_z;
            }
        }
    }

    // Store counts
    if (!found_hit) {
        hit_count[ray_idx] = 0;
    }
    miss_counts[ray_idx] = local_miss_count;
}

// Host function to launch kernel
std::vector<torch::Tensor> trace_rays_cuda(
    torch::Tensor ray_origins,     // (N, 3) float32
    torch::Tensor ray_dirs,        // (N, 3) float32
    torch::Tensor rso_shape,       // (grid_x, grid_y, grid_z) bool
    torch::Tensor grid_origin,     // (3,) float32
    float voxel_size,
    float p_hit_occ,
    float p_hit_emp,
    unsigned long long seed
) {
    const int num_rays = ray_origins.size(0);
    const int grid_x = rso_shape.size(0);
    const int grid_y = rso_shape.size(1);
    const int grid_z = rso_shape.size(2);

    // Allocate output tensors
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(ray_origins.device());

    torch::Tensor hit_coords = torch::zeros({num_rays, 3}, options);
    torch::Tensor hit_count = torch::zeros({num_rays}, options);
    torch::Tensor miss_coords = torch::zeros({num_rays * MAX_STEPS, 3}, options);
    torch::Tensor miss_counts = torch::zeros({num_rays}, options);

    // Launch kernel
    const int threads = 256;
    const int blocks = (num_rays + threads - 1) / threads;

    dda_ray_march_kernel<<<blocks, threads>>>(
        ray_origins.data_ptr<float>(),
        ray_dirs.data_ptr<float>(),
        rso_shape.data_ptr<bool>(),
        grid_origin.data_ptr<float>(),
        voxel_size,
        grid_x, grid_y, grid_z,
        p_hit_occ, p_hit_emp, seed,
        hit_coords.data_ptr<int>(),
        hit_count.data_ptr<int>(),
        miss_coords.data_ptr<int>(),
        miss_counts.data_ptr<int>(),
        num_rays
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Return as vector: [hit_coords, hit_count, miss_coords, miss_counts]
    std::vector<torch::Tensor> outputs;
    outputs.push_back(hit_coords);
    outputs.push_back(hit_count);
    outputs.push_back(miss_coords);
    outputs.push_back(miss_counts);
    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trace_rays", &trace_rays_cuda, "DDA Ray Marching (CUDA)");
}
