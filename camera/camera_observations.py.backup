import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

# PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU acceleration disabled.")

# CUDA kernel for ultra-fast ray tracing
try:
    from camera.cuda.cuda_wrapper import trace_rays_cuda, CUDA_AVAILABLE as CUDA_KERNEL_AVAILABLE
except ImportError:
    CUDA_KERNEL_AVAILABLE = False
    print("Note: Custom CUDA kernel not available. Using PyTorch GPU fallback.")

# --- Helper functions & Classes ---
def logit(p):
    """Works with both numpy and torch tensors"""
    if isinstance(p, torch.Tensor) if TORCH_AVAILABLE else False:
        p = torch.clamp(p, 1e-6, 1.0 - 1e-6)
        return torch.log(p / (1.0 - p))
    else:
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

def sigmoid(L):
    """Works with both numpy and torch tensors"""
    if isinstance(L, torch.Tensor) if TORCH_AVAILABLE else False:
        return 1.0 / (1.0 + torch.exp(-L))
    else:
        return 1.0 / (1.0 + np.exp(-L))

def calculate_entropy(belief):
    """Calculate entropy, works with both numpy and torch"""
    eps = 1e-9
    if isinstance(belief, torch.Tensor) if TORCH_AVAILABLE else False:
        belief_clipped = torch.clamp(belief, eps, 1 - eps)
        entropy = -torch.sum(belief_clipped * torch.log(belief_clipped) +
                          (1 - belief_clipped) * torch.log(1 - belief_clipped))
        return float(entropy.item())
    else:
        belief_clipped = np.clip(belief, eps, 1 - eps)
        entropy = -np.sum(belief_clipped * np.log(belief_clipped) +
                          (1 - belief_clipped) * np.log(1 - belief_clipped))
        return float(entropy)

class VoxelGrid:
    def __init__(self, grid_dims=(20, 20, 20), voxel_size=1.0, origin=(-10, -10, -10), 
                 use_torch=False, device='cpu'):
        """
        Voxel grid for belief state representation.
        
        Args:
            grid_dims: Tuple of (nx, ny, nz) voxel counts
            voxel_size: Size of each voxel in meters
            origin: World coordinates of grid origin
            use_torch: If True, use PyTorch tensors for GPU acceleration
            device: 'cpu' or 'cuda' - device for torch tensors
        """
        self.dims = grid_dims
        self.voxel_size = voxel_size
        self.origin = np.array(origin)
        self.max_bound = self.origin + np.array(self.dims) * self.voxel_size
        
        # GPU acceleration
        self.use_torch = use_torch and TORCH_AVAILABLE
        self.device = device if self.use_torch else 'cpu'
        
        if self.use_torch:
            self.belief = torch.full(self.dims, 0.5, dtype=torch.float32, device=self.device)
            self.log_odds = logit(self.belief)
            
            P_HIT_OCC = 0.95
            P_HIT_EMP = 0.001
            self.L_hit = logit(torch.tensor(P_HIT_OCC, device=self.device)) - \
                         logit(torch.tensor(P_HIT_EMP, device=self.device))
            self.L_miss = logit(torch.tensor(1-P_HIT_OCC, device=self.device)) - \
                          logit(torch.tensor(1-P_HIT_EMP, device=self.device))
        else:
            self.belief = np.full(self.dims, 0.5)
            self.log_odds = logit(self.belief)
            
            P_HIT_OCC = 0.95
            P_HIT_EMP = 0.001
            self.L_hit = logit(np.array(P_HIT_OCC)) - logit(np.array(P_HIT_EMP))
            self.L_miss = logit(np.array(1-P_HIT_OCC)) - logit(np.array(1-P_HIT_EMP))

    def get_entropy(self) -> float:
        return calculate_entropy(self.belief)

    def grid_to_world_coords(self, indices):
        """Convert grid indices to world coordinates"""
        if isinstance(indices, torch.Tensor) if TORCH_AVAILABLE else False:
            indices = indices.cpu().numpy()
        if indices.size == 0:
            return np.array([])
        return self.origin + (indices + 0.5) * self.voxel_size

    def world_to_grid_coords(self, world_pos):
        """Convert world coordinates to grid indices"""
        if isinstance(world_pos, torch.Tensor) if TORCH_AVAILABLE else False:
            world_pos = world_pos.cpu().numpy()
        if world_pos.ndim == 1:
            world_pos = world_pos[np.newaxis, :]
        indices = np.floor((world_pos - self.origin) / self.voxel_size)
        return indices.astype(int).squeeze()

    def is_in_bounds(self, grid_indices):
        """Check if grid indices are within bounds"""
        if isinstance(grid_indices, torch.Tensor) if TORCH_AVAILABLE else False:
            if grid_indices.ndim == 1:
                return (torch.all(grid_indices >= 0) and 
                        grid_indices[0] < self.dims[0] and
                        grid_indices[1] < self.dims[1] and
                        grid_indices[2] < self.dims[2])
            return torch.all((grid_indices >= 0) & (grid_indices < torch.tensor(self.dims, device=grid_indices.device)), dim=1)
        else:
            if grid_indices.ndim == 1:
                return (all(grid_indices >= 0) and 
                        grid_indices[0] < self.dims[0] and
                        grid_indices[1] < self.dims[1] and
                        grid_indices[2] < self.dims[2])
            return np.all((grid_indices >= 0) & (grid_indices < self.dims), axis=1)

    def update_belief(self, hit_voxels, missed_voxels):
        """Update belief state with observation results

        Args:
            hit_voxels: list of tuples OR torch.Tensor (M, 3) of hit coordinates
            missed_voxels: list of tuples OR torch.Tensor (K, 3) of miss coordinates
        """
        if self.use_torch:
            # GPU path
            hit_mask = torch.zeros(self.dims, dtype=torch.bool, device=self.device)

            # Handle both tensor and list inputs for hits
            if isinstance(hit_voxels, torch.Tensor):
                if len(hit_voxels) > 0:
                    # GPU tensor path (fast)
                    hit_indices = hit_voxels.long()
                    # Bounds checking on GPU
                    valid_mask = ((hit_indices >= 0).all(dim=1) &
                                 (hit_indices[:, 0] < self.dims[0]) &
                                 (hit_indices[:, 1] < self.dims[1]) &
                                 (hit_indices[:, 2] < self.dims[2]))
                    valid_hits = hit_indices[valid_mask]
                    if len(valid_hits) > 0:
                        hit_mask[valid_hits[:, 0], valid_hits[:, 1], valid_hits[:, 2]] = True
            elif hit_voxels:
                # List path (legacy, slower)
                valid = [idx for idx in hit_voxels if self.is_in_bounds(np.array(idx))]
                if valid:
                    hit_indices = torch.tensor(valid, dtype=torch.long, device=self.device)
                    hit_mask[hit_indices[:, 0], hit_indices[:, 1], hit_indices[:, 2]] = True

            miss_mask = torch.zeros(self.dims, dtype=torch.bool, device=self.device)

            # Handle both tensor and list inputs for misses
            if isinstance(missed_voxels, torch.Tensor):
                if len(missed_voxels) > 0:
                    # GPU tensor path (fast)
                    miss_indices = missed_voxels.long()
                    # Bounds checking on GPU
                    valid_mask = ((miss_indices >= 0).all(dim=1) &
                                 (miss_indices[:, 0] < self.dims[0]) &
                                 (miss_indices[:, 1] < self.dims[1]) &
                                 (miss_indices[:, 2] < self.dims[2]))
                    valid_misses = miss_indices[valid_mask]
                    if len(valid_misses) > 0:
                        miss_mask[valid_misses[:, 0], valid_misses[:, 1], valid_misses[:, 2]] = True
            elif missed_voxels:
                # List path (legacy, slower)
                valid = [idx for idx in missed_voxels if self.is_in_bounds(np.array(idx))]
                if valid:
                    miss_indices = torch.tensor(valid, dtype=torch.long, device=self.device)
                    miss_mask[miss_indices[:, 0], miss_indices[:, 1], miss_indices[:, 2]] = True

            self.log_odds[hit_mask] += self.L_hit
            self.log_odds[miss_mask] += self.L_miss
            self.belief = sigmoid(self.log_odds)
        else:
            # CPU path (original implementation)
            hit_mask = np.zeros(self.dims, dtype=bool)
            if hit_voxels:
                valid = [idx for idx in hit_voxels if self.is_in_bounds(np.array(idx))]
                if valid:
                    hit_mask[tuple(np.array(valid).T)] = True
            
            miss_mask = np.zeros(self.dims, dtype=bool)
            if missed_voxels:
                valid = [idx for idx in missed_voxels if self.is_in_bounds(np.array(idx))]
                if valid:
                    miss_mask[tuple(np.array(valid).T)] = True
            
            self.log_odds[hit_mask] += self.L_hit
            self.log_odds[miss_mask] += self.L_miss
            self.belief = sigmoid(self.log_odds)

class GroundTruthRSO:
    def __init__(self, grid: VoxelGrid):
        self.dims = grid.dims
        self.use_torch = grid.use_torch
        self.device = grid.device if grid.use_torch else 'cpu'
        
        if self.use_torch:
            self.shape = torch.zeros(self.dims, dtype=torch.bool, device=self.device)
        else:
            self.shape = np.zeros(self.dims, dtype=bool)
        
        self._create_simple_shape()

    def _create_simple_shape(self):
        """Create a simple spacecraft shape"""
        center = (self.dims[0] // 2, self.dims[1] // 2, self.dims[2] // 2)
        s = 4
        
        if self.use_torch:
            self.shape[center[0]-s:center[0]+s, center[1]-s:center[1]+s, center[2]-s:center[2]+s] = True
            self.shape[center[0]-s:center[0]+s, center[1]+s:center[1]+s+6, center[2]-1:center[2]+1] = True
            self.shape[center[0]-1:center[0]+1, center[1]-1:center[1]+1, center[2]+s:center[2]+s+3] = True
        else:
            self.shape[center[0]-s:center[0]+s, center[1]-s:center[1]+s, center[2]-s:center[2]+s] = True
            self.shape[center[0]-s:center[0]+s, center[1]+s:center[1]+s+6, center[2]-1:center[2]+1] = True
            self.shape[center[0]-1:center[0]+1, center[1]-1:center[1]+1, center[2]+s:center[2]+s+3] = True

# --- Ray Tracing ---
def get_camera_rays(camera_pos, view_dir, fov_degrees, sensor_res):
    """Generate camera rays (CPU version, converted to GPU in simulate_observation)"""
    view_dir = view_dir / np.linalg.norm(view_dir)
    global_up = np.array([0, 0, 1])
    if np.allclose(np.abs(view_dir), global_up):
        global_up = np.array([0, 1, 0])
    cam_right = np.cross(view_dir, global_up)
    if np.linalg.norm(cam_right) < 1e-6:
        global_up = np.array([0, 1, 0])
        cam_right = np.cross(view_dir, global_up)
    cam_right /= np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, view_dir)

    fov_rad = np.deg2rad(fov_degrees)
    aspect_ratio = sensor_res[0] / sensor_res[1]
    h_half = np.tan(fov_rad / 2.0)
    w_half = h_half * aspect_ratio
    
    rays = []
    for u in range(sensor_res[0]):
        for v in range(sensor_res[1]):
            nu = (u + 0.5) / sensor_res[0] * 2.0 - 1.0
            nv = (v + 0.5) / sensor_res[1] * 2.0 - 1.0
            r = (view_dir + cam_right * nu * w_half + cam_up * nv * h_half)
            rays.append(r / np.linalg.norm(r))
    return np.array(rays)


def _trace_rays_gpu_vectorized(ray_origins, ray_dirs, grid, rso, noise_params):
    """
    GPU-accelerated vectorized ray tracing for all rays simultaneously.
    
    Args:
        ray_origins: (N, 3) tensor of ray origins
        ray_dirs: (N, 3) tensor of ray directions
        grid: VoxelGrid object
        rso: GroundTruthRSO object
        noise_params: dict with 'p_hit_given_occupied' and 'p_hit_given_empty'
    
    Returns:
        hits: list of hit voxel tuples
        misses: list of missed voxel tuples
    """
    device = ray_origins.device
    N = ray_origins.shape[0]
    eps = 1e-9
    
    # Safe division
    ray_dirs_safe = ray_dirs.clone()
    ray_dirs_safe[torch.abs(ray_dirs_safe) < eps] = eps
    
    # Grid bounds
    grid_origin = torch.tensor(grid.origin, dtype=torch.float32, device=device)
    grid_max = torch.tensor(grid.max_bound, dtype=torch.float32, device=device)
    
    # Compute entry/exit points for grid
    t1 = (grid_origin - ray_origins) / ray_dirs_safe
    t2 = (grid_max - ray_origins) / ray_dirs_safe
    tn = torch.minimum(t1, t2)
    tx = torch.maximum(t1, t2)
    t_enter = torch.max(tn, dim=1)[0]
    t_exit = torch.min(tx, dim=1)[0]
    
    # Filter rays that miss the grid
    valid_rays = (t_enter <= t_exit) & (t_exit >= 0)
    
    hits = []
    misses = []
    
    if not valid_rays.any():
        return hits, misses
    
    # Process only valid rays
    valid_indices = torch.where(valid_rays)[0]
    ray_origins_valid = ray_origins[valid_indices]
    ray_dirs_valid = ray_dirs[valid_indices]
    ray_dirs_safe_valid = ray_dirs_safe[valid_indices]
    t_enter_valid = t_enter[valid_indices]
    
    # Start points
    start_points = torch.where(
        (t_enter_valid < 0).unsqueeze(1),
        ray_origins_valid,
        ray_origins_valid + ray_dirs_valid * t_enter_valid.unsqueeze(1)
    )
    
    # Convert to grid coordinates
    grid_coords = torch.floor((start_points - grid_origin) / grid.voxel_size).long()
    
    # DDA setup
    step = torch.sign(ray_dirs_valid).long()
    step[step == 0] = 1
    t_delta = torch.abs(grid.voxel_size / ray_dirs_safe_valid)
    
    bound = grid_origin + (grid_coords + (step > 0).long()).float() * grid.voxel_size
    t_max = (bound - ray_origins_valid) / ray_dirs_safe_valid
    
    # Max steps
    max_steps = int(sum(grid.dims)) * 2
    
    # Track active rays
    active = torch.ones(len(valid_indices), dtype=torch.bool, device=device)
    current_coords = grid_coords.clone()
    
    # DDA march (vectorized across all rays)
    for _ in range(max_steps):
        if not active.any():
            break
        
        # Check bounds
        in_bounds = (
            (current_coords[:, 0] >= 0) & (current_coords[:, 0] < grid.dims[0]) &
            (current_coords[:, 1] >= 0) & (current_coords[:, 1] < grid.dims[1]) &
            (current_coords[:, 2] >= 0) & (current_coords[:, 2] < grid.dims[2])
        )
        active = active & in_bounds
        
        if not active.any():
            break
        
        active_indices = torch.where(active)[0]
        
        # Get current voxels for active rays
        for idx in active_indices:
            coord = current_coords[idx]
            voxel_idx = tuple(coord.cpu().numpy())
            
            # Add to misses
            misses.append(voxel_idx)
            
            # Check for hit
            is_occupied = rso.shape[voxel_idx].item() if isinstance(rso.shape, torch.Tensor) else rso.shape[voxel_idx]
            
            if is_occupied:
                if np.random.rand() < noise_params["p_hit_given_occupied"]:
                    hits.append(voxel_idx)
                    misses.pop()  # Remove from misses since it's a hit
                    active[idx] = False
            elif np.random.rand() < noise_params["p_hit_given_empty"]:
                hits.append(voxel_idx)
                misses.pop()  # Remove from misses
                active[idx] = False
        
        # Advance rays
        if active.any():
            active_mask = active
            axis_to_step = torch.argmin(t_max[active_mask], dim=1)
            
            for i, idx in enumerate(torch.where(active_mask)[0]):
                axis = axis_to_step[i]
                t_max[idx, axis] += t_delta[idx, axis]
                current_coords[idx, axis] += step[idx, axis]
    
    return hits, misses


def _trace_ray(ray_origin, ray_dir, grid, rso, noise_params):
    """
    CPU-based single ray tracing (original implementation).
    Used when GPU is not available.
    """
    eps = 1e-9
    ray_dir_safe = ray_dir.copy()
    ray_dir_safe[np.abs(ray_dir_safe) < eps] = eps
    t1 = (grid.origin - ray_origin) / ray_dir_safe
    t2 = (grid.max_bound - ray_origin) / ray_dir_safe
    tn, tx = np.minimum(t1, t2), np.maximum(t1, t2)
    t_enter, t_exit = np.max(tn), np.min(tx)

    if t_enter > t_exit or t_exit < 0:
        return None, "miss", []
    start_point = ray_origin if t_enter < 0 else ray_origin + ray_dir * t_enter
    
    curr = grid.world_to_grid_coords(start_point)
    if not grid.is_in_bounds(curr):
        return None, "miss", []

    step = np.sign(ray_dir).astype(int)
    step[step == 0] = 1
    t_delta = np.abs(grid.voxel_size / ray_dir_safe)
    bound = grid.origin + (curr + (step > 0)) * grid.voxel_size
    t_max_march = (bound - ray_origin) / ray_dir_safe
    
    missed = []
    for _ in range(int(np.sum(grid.dims)) * 2):
        if not grid.is_in_bounds(curr):
            break
        v_idx = tuple(curr)
        missed.append(v_idx)
        
        # Check hit
        is_occupied = rso.shape[v_idx].item() if isinstance(rso.shape, torch.Tensor) else rso.shape[v_idx]
        if is_occupied:
            if np.random.rand() < noise_params["p_hit_given_occupied"]:
                return v_idx, "hit", missed[:-1]
        elif np.random.rand() < noise_params["p_hit_given_empty"]:
            return v_idx, "hit", missed[:-1]
        
        axis = np.argmin(t_max_march)
        t_max_march[axis] += t_delta[axis]
        curr[axis] += step[axis]
    
    return None, "miss", missed


def simulate_observation(grid, rso, camera_fn, servicer_rtn):
    """
    Simulate camera observation with optional GPU acceleration.

    Priority: CUDA kernel > PyTorch GPU > CPU
    """
    cam_pos = servicer_rtn[-1] if servicer_rtn.ndim > 1 else servicer_rtn
    view_dir = -cam_pos / np.linalg.norm(cam_pos)
    rays = get_camera_rays(cam_pos, view_dir, camera_fn['fov_degrees'], camera_fn['sensor_res'])

    if grid.use_torch and TORCH_AVAILABLE:
        ray_origins = torch.tensor(np.tile(cam_pos, (len(rays), 1)), dtype=torch.float32, device=grid.device)
        ray_dirs = torch.tensor(rays, dtype=torch.float32, device=grid.device)
        grid_origin_tensor = torch.tensor(grid.origin, dtype=torch.float32, device=grid.device)

        # Try CUDA kernel first (fastest)
        if CUDA_KERNEL_AVAILABLE and grid.device == 'cuda':
            hits, misses = trace_rays_cuda(
                ray_origins, ray_dirs, rso.shape, grid_origin_tensor,
                grid.voxel_size,
                camera_fn['noise_params']['p_hit_given_occupied'],
                camera_fn['noise_params']['p_hit_given_empty'],
                return_tensors=True  # Return GPU tensors for speed
            )
            # Update belief with GPU tensors directly (no CPU transfer)
            grid.update_belief(hits, misses)
            # Only convert to lists for return value if needed by caller
            return hits, misses
        else:
            # Fall back to PyTorch GPU (slower but still GPU-accelerated)
            hits, misses = _trace_rays_gpu_vectorized(ray_origins, ray_dirs, grid, rso, camera_fn['noise_params'])
            grid.update_belief(list(hits), list(misses))
            return list(hits), list(misses)
    else:
        # CPU path - sequential ray tracing (original)
        hits, misses = set(), set()
        for r in rays:
            h, s, m = _trace_ray(cam_pos, r, grid, rso, camera_fn['noise_params'])
            misses.update(m)
            if s == 'hit':
                hits.add(h)
        hits, misses = list(hits), list(misses)
        grid.update_belief(hits, misses)
        return hits, misses


# --- VISUALIZATION (unchanged) ---

def draw_spacecraft(ax, position, direction, color="gray", scale=(4.0, 2.0, 2.0)):
    """
    Draws a spacecraft as a 3D prism.
    Scale is a tuple (L, W, H) in meters.
    """
    direction = direction / np.linalg.norm(direction)
    global_z = np.array([0, 0, 1])
    temp_up = np.array([0, 1, 0]) if np.allclose(np.abs(direction), global_z) else global_z
    right = np.cross(direction, temp_up); right /= np.linalg.norm(right)
    up = np.cross(right, direction); up /= np.linalg.norm(up)

    L, W, H = scale
    d_vec = direction * (L / 2)
    r_vec = right * (W / 2)
    u_vec = up * (H / 2)

    c = np.array([
        position + d_vec + r_vec + u_vec, position + d_vec - r_vec + u_vec,
        position + d_vec - r_vec - u_vec, position + d_vec + r_vec - u_vec,
        position - d_vec + r_vec + u_vec, position - d_vec - r_vec + u_vec,
        position - d_vec - r_vec - u_vec, position - d_vec + r_vec - u_vec
    ])

    faces = [[c[0], c[1], c[2], c[3]], [c[4], c[5], c[6], c[7]], 
             [c[0], c[3], c[7], c[4]], [c[1], c[2], c[6], c[5]], 
             [c[0], c[1], c[5], c[4]], [c[3], c[2], c[6], c[7]]]

    box = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors="k", alpha=1.0)
    ax.add_collection3d(box)
    return box

def plot_scenario(grid, rso, camera_pos_world, view_direction, fov_degrees, sensor_res, fig=None, ax=None):
    if fig is None:
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('T [m]')
        ax.set_zlabel('N [m]')
        ax.set_title('RSO Characterization Mission', fontsize=16)

        artists = {}
        
        # 1. Servicer Path Line
        artists['servicer_path_line'], = ax.plot([], [], [], c='blue', ls='-', lw=1.5, alpha=0.6, label='Trajectory')
        
        # 2. Burn Markers
        artists['burn_scatter'] = ax.scatter([], [], [], c='orange', marker='^', s=150, label='Maneuver', zorder=10)
        
        # 3. Servicer Spacecraft
        artists['servicer_prism'] = draw_spacecraft(ax, camera_pos_world, view_direction, scale=(4.0, 2.0, 2.0))
        ax.plot([], [], [], color='gray', marker='s', markersize=10, linestyle='None', label='Servicer')
        
        # 4. FOV Lines
        artists['fov_lines'] = [ax.plot([], [], [], c='magenta', ls='-', lw=2.0, label='Camera FOV' if i==0 else None)[0] for i in range(8)]
        
        # 5. View Direction
        artists['view_line'] = ax.plot([], [], [], c='darkgreen', ls='--', lw=2.5, label='View Direction')[0]
        
        # 6. Belief Scatter
        artists['belief_scatter'] = ax.scatter([], [], [], c='green', marker='s', s=30, label='Belief (P>0.7)', depthshade=True)
        
        # 7. Ground Truth
        if isinstance(rso.shape, torch.Tensor) if TORCH_AVAILABLE else False:
            truth_pts = torch.nonzero(rso.shape).cpu().numpy()
        else:
            truth_pts = np.argwhere(rso.shape)
        
        if len(truth_pts) > 0:
            world_pts = grid.grid_to_world_coords(truth_pts)
            ax.scatter(world_pts[:,0], world_pts[:,1], world_pts[:,2], c='red', marker='.', alpha=0.15, label='Target (Hidden)')

        artists['entropy_text'] = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize='small')
        return fig, ax, artists
    return fig, ax, None

def update_plot(frame, grid, rso, camera_positions, view_directions, fov_degrees, sensor_res, noise_params, ax, artists, target_com, burn_indices=None):
    """
    Updates the plot for the given frame.
    """
    if frame >= len(camera_positions):
        return []

    cam_pos = camera_positions[frame]
    view_dir = view_directions[frame]
   
    # 1. Update Belief
    if isinstance(grid.belief, torch.Tensor) if TORCH_AVAILABLE else False:
        belief_np = grid.belief.cpu().numpy()
    else:
        belief_np = grid.belief
    
    mask = (belief_np > 0.7)
    indices = np.argwhere(mask)
    if indices.size > 0:
        world = grid.grid_to_world_coords(indices)
        probs = belief_np[indices[:,0], indices[:,1], indices[:,2]]
        colors = np.array([[0.0, 1.0, 0.0, a] for a in np.clip(probs * 1.5 - 0.5, 0.3, 1.0)])
        artists['belief_scatter']._offsets3d = (world[:, 0], world[:, 1], world[:, 2])
        artists['belief_scatter'].set_facecolors(colors)
    else:
        artists['belief_scatter']._offsets3d = ([], [], [])

    # 2. Update Path
    artists['servicer_path_line'].set_data_3d(camera_positions[:frame+1, 0], 
                                              camera_positions[:frame+1, 1], 
                                              camera_positions[:frame+1, 2])

    # 3. Update Burn Markers
    if burn_indices:
        past_burns = [b_idx for b_idx in burn_indices if b_idx <= frame]
        if past_burns:
            burn_x = camera_positions[past_burns, 0]
            burn_y = camera_positions[past_burns, 1]
            burn_z = camera_positions[past_burns, 2]
            artists['burn_scatter']._offsets3d = (burn_x, burn_y, burn_z)
        else:
            artists['burn_scatter']._offsets3d = ([], [], [])

    # 4. Update Spacecraft
    artists['servicer_prism'].remove()
    artists['servicer_prism'] = draw_spacecraft(ax, cam_pos, view_dir, scale=(4.0, 2.0, 2.0))
    
    # 5. Update View Direction
    artists['view_line'].set_data_3d([cam_pos[0], target_com[0]], [cam_pos[1], target_com[1]], [cam_pos[2], target_com[2]])
    
    # 6. Update FOV
    global_up = np.array([0, 0, 1])
    if np.allclose(np.abs(view_dir), global_up):
        global_up = np.array([0, 1, 0])
    right = np.cross(view_dir, global_up); right /= np.linalg.norm(right)
    up = np.cross(right, view_dir)
    
    fov_rad = np.deg2rad(fov_degrees)
    ar = sensor_res[0]/sensor_res[1]
    h_half = np.tan(fov_rad/2)
    w_half = h_half * ar
    
    dist = np.linalg.norm(cam_pos - target_com)
    corners = [np.array([-1,-1]), np.array([1,-1]), np.array([1,1]), np.array([-1,1])]
    pts = []
    for u,v in corners:
        d = (view_dir + right*u*w_half + up*v*h_half); d /= np.linalg.norm(d)
        pts.append(cam_pos + d * dist)
        
    for i in range(4):
        artists['fov_lines'][i].set_data_3d([cam_pos[0], pts[i][0]], [cam_pos[1], pts[i][1]], [cam_pos[2], pts[i][2]])
    pts.append(pts[0])
    for i in range(4):
        artists['fov_lines'][4+i].set_data_3d([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]], [pts[i][2], pts[i+1][2]])

    artists['entropy_text'].set_text(f"Entropy: {grid.get_entropy():.2f}")
    
    return [artists['belief_scatter'], artists['servicer_path_line'], artists['servicer_prism']]
