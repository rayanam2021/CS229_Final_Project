import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # For custom 3D shapes
import matplotlib.cm as cm
import sys # Using sys.float_info.epsilon for robust ray tracing

# --- Helper functions for Log-Odds Bayesian Update ---

def logit(p: np.ndarray) -> np.ndarray:
    """Converts probabilities (p) to log-odds (L)."""
    p = np.clip(p, 1e-6, 1.0 - 1e-6) # Clip to avoid log(0)
    return np.log(p / (1.0 - p))

def sigmoid(L: np.ndarray) -> np.ndarray:
    """Converts log-odds (L) back to probabilities (p)."""
    return 1.0 / (1.0 + np.exp(-L))

# --- Original Information Gain Functions ---

def calculate_entropy(b_shape: np.ndarray) -> float:
    """
    Calculates the total Shannon entropy of a probabilistic belief state.
    """
    p = b_shape
    one_minus_p = 1.0 - p
    
    # Use np.where to avoid log(0)
    log_p = np.where(p == 0, 0, np.log2(p))
    term1 = p * log_p
    
    log_one_minus_p = np.where(one_minus_p == 0, 0, np.log2(one_minus_p))
    term2 = one_minus_p * log_one_minus_p

    voxel_entropies = - (term1 + term2)
    total_entropy = np.sum(voxel_entropies)
    
    return total_entropy

def calculate_information_gain(b_shape_before: np.ndarray, 
                               b_shape_after: np.ndarray) -> float:
    """
    Calculates the information gain as the reduction in entropy.
    """
    entropy_before = calculate_entropy(b_shape_before)
    entropy_after = calculate_entropy(b_shape_after)
    
    info_gain = entropy_before - entropy_after
    
    return info_gain

# --- Simulation Classes and Functions ---

class VoxelGrid:
    """Manages the 3D probabilistic belief state."""
    
    def __init__(self, grid_dims: tuple = (20, 20, 20), voxel_size: float = 1.0, origin: tuple = (-10, -10, -10)):
        """
        Initializes the voxel grid.

        Args:
            grid_dims: (Nx, Ny, Nz) dimensions of the grid.
            voxel_size: The physical size of each voxel (e.g., in meters).
            origin: The (x, y, z) world coordinate of the (0,0,0) grid index.
        """
        self.dims = grid_dims
        self.voxel_size = voxel_size
        self.origin = np.array(origin)
        self.max_bound = self.origin + np.array(self.dims) * self.voxel_size
        
        self.belief = np.full(self.dims, 0.5)
        self.log_odds = logit(self.belief)

        # Sensor model parameters (log-odds)
        # This is the AGENT'S *belief* about its sensor, used for updates.
        # Use realistic parameters
        P_HIT_GIVEN_OCCUPIED = 0.95
        P_HIT_GIVEN_EMPTY = 0.001
        
        P_MISS_GIVEN_OCCUPIED = 1.0 - P_HIT_GIVEN_OCCUPIED
        P_MISS_GIVEN_EMPTY = 1.0 - P_HIT_GIVEN_EMPTY

        self.L_hit_update = logit(np.array(P_HIT_GIVEN_OCCUPIED)) - logit(np.array(P_HIT_GIVEN_EMPTY)) 
        self.L_miss_update = logit(np.array(P_MISS_GIVEN_OCCUPIED)) - logit(np.array(P_MISS_GIVEN_EMPTY))

    def get_entropy(self) -> float:
        """Calculates the total entropy of the current belief grid."""
        return calculate_entropy(self.belief)

    def grid_to_world_coords(self, indices: np.ndarray) -> np.ndarray:
        """Converts grid indices (i, j, k) to world coordinates (x, y, z)."""
        if indices.size == 0:
            return np.array([])
        return self.origin + (indices + 0.5) * self.voxel_size

    def world_to_grid_coords(self, world_pos: np.ndarray) -> np.ndarray:
        """Converts world coordinates (x, y, z) to grid indices (i, j, k)."""
        if world_pos.ndim == 1:
            world_pos = world_pos[np.newaxis, :]
            
        indices = np.floor((world_pos - self.origin) / self.voxel_size)
        return indices.astype(int).squeeze()

    def is_in_bounds(self, grid_indices: np.ndarray) -> bool:
        """Checks if (i, j, k) grid indices are within the grid dimensions."""
        if grid_indices.ndim == 1:
            return (all(grid_indices >= 0) and 
                    grid_indices[0] < self.dims[0] and
                    grid_indices[1] < self.dims[1] and
                    grid_indices[2] < self.dims[2])
        else:
            # Handle array of indices
            return np.all((grid_indices >= 0) & (grid_indices < self.dims), axis=1)


    def update_belief(self, hit_voxels: list, missed_voxels: list):
        """
        Updates the belief grid using a log-odds update.
        """
        print(f"Updating belief with {len(hit_voxels)} hits and {len(missed_voxels)} misses...")
        
        # Create boolean masks for efficient numpy updates
        hit_mask = np.zeros(self.dims, dtype=bool)
        # Handle empty lists gracefully
        if hit_voxels:
            # Filter out-of-bounds indices that may have resulted from noise
            valid_hits = [idx for idx in hit_voxels if self.is_in_bounds(np.array(idx))]
            if valid_hits:
                hit_indices = tuple(np.array(valid_hits).T)
                hit_mask[hit_indices] = True
        
        miss_mask = np.zeros(self.dims, dtype=bool)
        if missed_voxels:
            # Filter out-of-bounds indices
            valid_misses = [idx for idx in missed_voxels if self.is_in_bounds(np.array(idx))]
            if valid_misses:
                miss_indices = tuple(np.array(valid_misses).T)
                miss_mask[miss_indices] = True

        # Apply updates using pre-calculated log-odds ratios
        self.log_odds[hit_mask] += self.L_hit_update
        self.log_odds[miss_mask] += self.L_miss_update
            
        self.belief = sigmoid(self.log_odds)
        print("Belief update complete.")


class GroundTruthRSO:
    """Represents the actual, hidden ground-truth shape of the RSO."""
    
    def __init__(self, grid: VoxelGrid):
        self.dims = grid.dims
        self.shape = np.zeros(self.dims, dtype=bool)
        self._create_simple_shape()

    def _create_simple_shape(self):
        """Creates a simple 'T' or 'L' shape for demonstration."""
        center = (self.dims[0] // 2, self.dims[1] // 2, self.dims[2] // 2)
        s = 4 # half-size for main body
        
        # Main body (cube)
        self.shape[center[0]-s:center[0]+s, 
                   center[1]-s:center[1]+s, 
                   center[2]-s:center[2]+s] = True
        
        # "Solar panel" extending in Y direction
        panel_thickness = 1
        panel_length = 6
        self.shape[center[0]-s:center[0]+s, 
                   center[1]+s:center[1]+s+panel_length, 
                   center[2]-panel_thickness:center[2]+panel_thickness] = True
        
        # Add a small antenna on top
        self.shape[center[0]-1:center[0]+1,
                   center[1]-1:center[1]+1,
                   center[2]+s:center[2]+s+3] = True
                   
        print(f"Created ground truth RSO with {np.sum(self.shape)} filled voxels.")

# --- New Pinhole Camera and Ray Tracing Functions ---

def get_camera_rays(camera_pos: np.ndarray, view_dir: np.ndarray, fov_degrees: float, sensor_res: tuple) -> np.ndarray:
    """
    Generates a list of ray direction vectors for a pinhole camera.
    """
    # Ensure view_dir is normalized
    view_dir = view_dir / np.linalg.norm(view_dir)
    
    # Create camera coordinate frame
    # Use a "global up" vector, handling singularity at poles
    global_up = np.array([0, 0, 1])
    if np.allclose(np.abs(view_dir), global_up):
        global_up = np.array([0, 1, 0]) # Use Y-up if looking straight up/down
        
    cam_right = np.cross(view_dir, global_up)
    cam_right /= np.linalg.norm(cam_right)
    
    cam_up = np.cross(cam_right, view_dir)
    # cam_up is already normalized
    
    # Calculate sensor plane dimensions
    fov_rad = np.deg2rad(fov_degrees)
    aspect_ratio = sensor_res[0] / sensor_res[1]
    sensor_height = 2.0 * np.tan(fov_rad / 2.0)
    sensor_width = sensor_height * aspect_ratio
    
    # Create ray directions for each pixel
    rays = []
    for u in range(sensor_res[0]): # Pixel column
        for v in range(sensor_res[1]): # Pixel row
            # Convert pixel coords (0 to Res) to normalized screen coords (-1 to 1)
            norm_u = (u + 0.5) / sensor_res[0] * 2.0 - 1.0
            norm_v = (v + 0.5) / sensor_res[1] * 2.0 - 1.0
            
            # Calculate direction for this ray
            ray_dir = (view_dir + 
                       cam_right * norm_u * sensor_width / 2.0 + 
                       cam_up * norm_v * sensor_height / 2.0)
            
            ray_dir /= np.linalg.norm(ray_dir)
            rays.append(ray_dir)
            
    return np.array(rays)

def _trace_ray(ray_origin: np.ndarray, ray_dir: np.ndarray, grid: VoxelGrid, rso: GroundTruthRSO, noise_params: dict) -> (tuple, str, list):
    """
    Traces a single ray through the voxel grid using 3D DDA (Amanatides-Woo).
    Applies noise model to simulate sensor.
    """
    
    # --- 1. Find Bounding Box Intersection (Corrected Slab Test) ---
    # Add epsilon to avoid division by zero if ray is axis-aligned
    # Use a small epsilon
    epsilon = sys.float_info.epsilon
    ray_dir_safe = np.where(np.abs(ray_dir) < epsilon, np.sign(ray_dir) * epsilon + epsilon, ray_dir)
    
    t1 = (grid.origin - ray_origin) / ray_dir_safe
    t2 = (grid.max_bound - ray_origin) / ray_dir_safe
    
    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)
    
    t_enter = np.max(t_min)
    t_exit = np.min(t_max)

    if t_enter > t_exit or t_exit < 0:
        return None, 'miss', [] # Ray misses the bounding box

    # Start point is the entry point
    if t_enter < 0:
        t_enter = 0 # Ray starts inside the box
        
    start_point_world = ray_origin + ray_dir * t_enter
    
    # --- 2. Initialize 3D DDA ---
    current_voxel_idx = grid.world_to_grid_coords(start_point_world)
    
    # Handle starting right on a boundary by clipping
    current_voxel_idx = np.clip(current_voxel_idx, 0, np.array(grid.dims) - 1)
            
    if not grid.is_in_bounds(current_voxel_idx):
        return None, 'miss', [] # Started outside bounds
        
    # Voxel step direction (+1 or -1)
    step = np.sign(ray_dir).astype(int)
    
    # tDelta: distance to cross one voxel in each dimension
    t_delta = np.abs(grid.voxel_size / ray_dir_safe)
    
    # tMax: distance to next voxel boundary
    next_boundary = (current_voxel_idx + (step > 0)) * grid.voxel_size + grid.origin
    t_max = (next_boundary - ray_origin) / ray_dir_safe
    
    missed_voxels_list = []

    # --- 3. Voxel Traversal Loop ---
    for _ in range(int(np.max(grid.dims) * 3)): # Safety break
        
        # Voxel to check
        voxel_tuple = tuple(current_voxel_idx)
        
        if rso.shape[voxel_tuple]:
            # Ray hit an OCCUPIED voxel. Apply noise.
            if np.random.rand() < noise_params['p_hit_given_occupied']:
                # TRUE HIT: Sensor correctly reports hit.
                return voxel_tuple, 'hit', missed_voxels_list
            else:
                # FALSE MISS (Type II): Sensor misses the occupied voxel.
                # The object is still opaque, so the ray stops here.
                # We add it to the missed list (as the sensor *reported* a miss)
                # but we do NOT continue tracing.
                missed_voxels_list.append(voxel_tuple)
                return None, 'miss', missed_voxels_list # <-- FIX: Stop the ray
        else:
            # Ray hit an EMPTY voxel. Apply noise.
            if np.random.rand() < noise_params['p_hit_given_empty']:
                # FALSE HIT (Type I): Sensor incorrectly reports hit.
                return voxel_tuple, 'hit', missed_voxels_list
            else:
                # TRUE MISS: Sensor correctly reports miss.
                missed_voxels_list.append(voxel_tuple)
                # Continue tracing ray.

        # Step to next voxel
        if t_max[0] < t_max[1]:
            if t_max[0] < t_max[2]:
                current_voxel_idx[0] += step[0]
                t_max[0] += t_delta[0]
            else:
                current_voxel_idx[2] += step[2]
                t_max[2] += t_delta[2]
        else:
            if t_max[1] < t_max[2]:
                current_voxel_idx[1] += step[1]
                t_max[1] += t_delta[1]
            else:
                current_voxel_idx[2] += step[2]
                t_max[2] += t_delta[2]

        # Check if ray has exited the grid
        if not grid.is_in_bounds(current_voxel_idx):
            return None, 'miss', missed_voxels_list # Ray exited
            
    return None, 'miss', missed_voxels_list # Hit safety break


def simulate_observation(camera_pos_world: np.ndarray, view_direction: np.ndarray, 
                         fov_degrees: float, sensor_res: tuple, 
                         noise_params: dict, grid: VoxelGrid, rso: GroundTruthRSO) -> (list, list):
    """
    Simulates a noisy pinhole camera observation.
    """
    print(f"Simulating noisy pinhole camera observation...")
    
    # 1. Get all ray directions for the sensor
    ray_directions = get_camera_rays(camera_pos_world, view_direction, fov_degrees, sensor_res)
    
    all_hit_voxels = set()
    all_missed_voxels = set()
    
    # 2. Trace each ray
    for ray_dir in ray_directions:
        hit_voxel_idx, status, missed_list = _trace_ray(
            camera_pos_world, ray_dir, grid, rso, noise_params
        )
        
        all_missed_voxels.update(missed_list)
        if status == 'hit':
            all_hit_voxels.add(hit_voxel_idx)
            
    return list(all_hit_voxels), list(all_missed_voxels)


def get_random_rtn_position(min_radius: float, max_radius: float) -> np.ndarray:
    """
    Generates a random (x,y,z) position in a spherical shell.
    "RTN" is simplified to a standard spherical coordinate system.
    """
    # Random radius in the shell
    radius = np.random.uniform(min_radius, max_radius)
    
    # Random azimuth (longitude)
    azimuth = np.random.uniform(0, 2 * np.pi)
    
    # Random elevation (latitude) - use cosine distribution for uniform sampling
    cos_elevation = np.random.uniform(-1, 1)
    elevation = np.arccos(cos_elevation) # Angle from Z-axis (phi)
    
    # Convert spherical to Cartesian
    x = radius * np.sin(elevation) * np.cos(azimuth)
    y = radius * np.sin(elevation) * np.sin(azimuth) # <-- FIX: Was 'azim'
    z = radius * np.cos(elevation)
    
    return np.array([x, y, z])


def plot_scenario(grid: VoxelGrid, rso: GroundTruthRSO, camera_pos_world: np.ndarray, 
                  view_direction: np.ndarray, fov_degrees: float, sensor_res: tuple):
    """
    Generates a 3D plot of the ground truth, belief state, and camera with a standard background.
    """
    print("Generating 3D plot...")
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # --- Standard Background Settings ---
    ax.grid(True)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('RSO Characterization Mission: Simulated Observation', fontsize=16)

    # --- Plotting Elements ---

    # 0. Plot Bounding Box
    min_c = grid.origin
    max_c = grid.max_bound
    
    # Define 8 corners
    corners = np.array([
        [min_c[0], min_c[1], min_c[2]],
        [max_c[0], min_c[1], min_c[2]],
        [max_c[0], max_c[1], min_c[2]],
        [min_c[0], max_c[1], min_c[2]],
        [min_c[0], min_c[1], max_c[2]],
        [max_c[0], min_c[1], max_c[2]],
        [max_c[0], max_c[1], max_c[2]],
        [min_c[0], max_c[1], max_c[2]]
    ])
    
    # Define 12 edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4), # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical edges
    ]
    
    # Plot edges
    for (i, j) in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]], 
            [corners[i, 1], corners[j, 1]], 
            [corners[i, 2], corners[j, 2]], 
            c='gray', linestyle=':', linewidth=1
        )
    # Add dummy plot for legend
    ax.plot([], [], [], c='gray', linestyle=':', label='Grid Bounding Box')


    # 1. Plot Ground Truth (Hidden, but for visualization here)
    # Find only the exposed faces of the ground truth
    truth_filled_voxels = np.argwhere(rso.shape)
    exposed_faces = [] # List to store (voxel_index_tuple, face_direction_index)
    
    # Pad the shape array to handle boundary checks easily
    padded_shape = np.pad(rso.shape, pad_width=1, mode='constant', constant_values=False)
    
    truth_world_coords_for_lims = [] # To store coords for plot limits
    
    for x, y, z in truth_filled_voxels:
        # Padded index
        px, py, pz = x + 1, y + 1, z + 1
        
        # Add center for plot limits
        origin_voxel_lim = grid.origin + np.array([x, y, z]) * grid.voxel_size
        truth_world_coords_for_lims.append(origin_voxel_lim + grid.voxel_size/2.0)
        
        # Check 6 neighbors
        if not padded_shape[px+1, py, pz]:
            exposed_faces.append(((x,y,z), 0)) # +X face exposed
        if not padded_shape[px-1, py, pz]:
            exposed_faces.append(((x,y,z), 1)) # -X face exposed
        if not padded_shape[px, py+1, pz]:
            exposed_faces.append(((x,y,z), 2)) # +Y face exposed
        if not padded_shape[px, py-1, pz]:
            exposed_faces.append(((x,y,z), 3)) # -Y face exposed
        if not padded_shape[px, py, pz+1]:
            exposed_faces.append(((x,y,z), 4)) # +Z face exposed
        if not padded_shape[px, py, pz-1]:
            exposed_faces.append(((x,y,z), 5)) # -Z face exposed

    print(f"Plotting {len(exposed_faces)} ground truth exposed faces.")

    # Create meshes for only the exposed faces
    face_verts = []
    
    for (x, y, z), face_dir in exposed_faces: 
        # Get world coordinates of the voxel's lower-left-front corner
        origin_voxel = grid.origin + np.array([x, y, z]) * grid.voxel_size
        size = grid.voxel_size

        # Define 8 corners of the cube
        c000 = origin_voxel
        c100 = origin_voxel + np.array([size, 0, 0])
        c010 = origin_voxel + np.array([0, size, 0])
        c001 = origin_voxel + np.array([0, 0, size])
        c110 = origin_voxel + np.array([size, size, 0])
        c101 = origin_voxel + np.array([size, 0, size])
        c011 = origin_voxel + np.array([0, size, size])
        c111 = origin_voxel + np.array([size, size, size])
        
        if face_dir == 0: # +X face (Right)
            face_verts.append([c100, c110, c111, c101])
        elif face_dir == 1: # -X face (Left)
            face_verts.append([c000, c001, c011, c010])
        elif face_dir == 2: # +Y face (Back)
            face_verts.append([c010, c110, c111, c011])
        elif face_dir == 3: # -Y face (Front)
            face_verts.append([c000, c100, c101, c001])
        elif face_dir == 4: # +Z face (Top)
            face_verts.append([c001, c101, c111, c011])
        elif face_dir == 5: # -Z face (Bottom)
            face_verts.append([c000, c100, c110, c010])

    if face_verts:
        # --- FIX: Replaced Poly3DCollection with manual edge plotting ---
        # Poly3DCollection is buggy, so we plot each edge manually.
        
        edges_set = set()
        for face in face_verts:
            # A face is a list of 4 corner points
            p1, p2, p3, p4 = face
            # Add the 4 edges of the face to a set to avoid duplicates
            # (sorting tuples to make edge (A,B) == edge (B,A))
            edges_set.add(tuple(sorted((tuple(p1), tuple(p2)))))
            edges_set.add(tuple(sorted((tuple(p2), tuple(p3)))))
            edges_set.add(tuple(sorted((tuple(p3), tuple(p4)))))
            edges_set.add(tuple(sorted((tuple(p4), tuple(p1)))))

        # Plot each unique edge
        for p1_tuple, p2_tuple in edges_set:
            p1 = np.array(p1_tuple)
            p2 = np.array(p2_tuple)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color=(1.0, 0.6, 0.6), linewidth=0.5, alpha=0.5)
        # --- End of fix ---

        # Add a dummy artist for the legend
        ax.plot([], [], [], color=(1.0, 0.6, 0.6), linestyle='-', alpha=0.5, markersize=10, label='Ground Truth (Hidden Surface)')


    # 2. Plot Belief State
    # Only show voxels where belief is strong (P > 0.7)
    certain_mask = (grid.belief > 0.7)
    certain_indices = np.argwhere(certain_mask)
    
    if certain_indices.size > 0:
        certain_world = grid.grid_to_world_coords(certain_indices)
        certain_probabilities = grid.belief[certain_indices[:,0], certain_indices[:,1], certain_indices[:,2]]

        # Using green color for belief voxels
        # Alpha based on certainty (more certain = less transparent)
        colors = np.array([[0.0, 1.0, 0.0, alpha] for alpha in np.clip(certain_probabilities * 1.5 - 0.5, 0.3, 1.0)]) # Green (R,G,B,A)

        ax.scatter(certain_world[:, 0], certain_world[:, 1], certain_world[:, 2], 
                   c=colors, marker='o', s=50, label='Belief Voxels (P > 0.7)', depthshade=False)
    else:
        # Add a dummy artist if no certain voxels, to show in legend
        ax.plot([], [], [], color='gray', marker='o', linestyle='None', markersize=5, label='Belief Voxels (No Certainty > 0.7)')
    
    # 3. Plot Camera and View Direction
    ax.scatter(camera_pos_world[0], camera_pos_world[1], camera_pos_world[2], 
               c='blue', marker='^', s=400, label='Servicer Camera (Agent)', edgecolors='black', linewidths=1.5)
    
    # Plot a line indicating the viewing direction
    # Show view line pointing from camera to CoM (0,0,0)
    ax.plot([camera_pos_world[0], 0], 
            [camera_pos_world[1], 0], 
            [camera_pos_world[2], 0], 
            c='green', linestyle='--', linewidth=2, label='Viewing Direction (to CoM)')

    # 4. Plot Camera FOV Cone
    # Get camera frame vectors
    view_dir = view_direction # Already normalized from __main__
    global_up = np.array([0, 0, 1])
    if np.allclose(np.abs(view_dir), global_up):
        global_up = np.array([0, 1, 0])
    cam_right = np.cross(view_dir, global_up)
    cam_right /= np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, view_dir)
    
    # Get sensor plane dimensions
    fov_rad = np.deg2rad(fov_degrees)
    aspect_ratio = sensor_res[0] / sensor_res[1]
    sensor_height = 2.0 * np.tan(fov_rad / 2.0)
    sensor_width = sensor_height * aspect_ratio
    
    # Define 4 corner points (norm_u, norm_v)
    corners_norm = [
        np.array([-1, -1]), # Bottom-left
        np.array([1, -1]),  # Bottom-right
        np.array([1, 1]),   # Top-right
        np.array([-1, 1])   # Top-left
    ]
    
    # Define cone length (e.g., distance to RSO CoM)
    cone_length = np.linalg.norm(camera_pos_world)
    
    corner_points_world = []
    for norm_uv in corners_norm:
        norm_u, norm_v = norm_uv
        # Calculate direction for this corner ray
        ray_dir = (view_dir + 
                   cam_right * norm_u * sensor_width / 2.0 + 
                   cam_up * norm_v * sensor_height / 2.0)
        ray_dir /= np.linalg.norm(ray_dir)
        
        # Calculate corner point in world
        corner_point = camera_pos_world + ray_dir * cone_length
        corner_points_world.append(corner_point)
        
        # Plot line from camera to corner
        ax.plot([camera_pos_world[0], corner_point[0]], 
                [camera_pos_world[1], corner_point[1]], 
                [camera_pos_world[2], corner_point[2]], 
                c='cyan', linestyle=':', linewidth=1)
    
    # Plot the base of the cone
    cp = corner_points_world
    ax.plot([cp[0][0], cp[1][0]], [cp[0][1], cp[1][1]], [cp[0][2], cp[1][2]], c='cyan', linestyle=':', linewidth=1)
    ax.plot([cp[1][0], cp[2][0]], [cp[1][1], cp[2][1]], [cp[1][2], cp[2][2]], c='cyan', linestyle=':', linewidth=1)
    ax.plot([cp[2][0], cp[3][0]], [cp[2][1], cp[3][1]], [cp[2][2], cp[3][2]], c='cyan', linestyle=':', linewidth=1)
    ax.plot([cp[3][0], cp[0][0]], [cp[3][1], cp[0][1]], [cp[3][2], cp[0][2]], c='cyan', linestyle=':', linewidth=1)
        
    # Add dummy legend item
    ax.plot([], [], [], c='cyan', linestyle=':', label='Camera FOV Cone')

    # --- Set Plot Limits and View ---
    # Auto-adjust limits based on ground truth, ensuring camera is visible
    if 'truth_world_coords_for_lims' in locals() and truth_world_coords_for_lims:
        truth_world = np.array(truth_world_coords_for_lims)
        all_coords_x = truth_world[:, 0]
        all_coords_y = truth_world[:, 1]
        all_coords_z = truth_world[:, 2]
    else:
        all_coords_x, all_coords_y, all_coords_z = np.array([]), np.array([]), np.array([])

    all_coords_x = np.append(all_coords_x, camera_pos_world[0])
    all_coords_y = np.append(all_coords_y, camera_pos_world[1])
    all_coords_z = np.append(all_coords_z, camera_pos_world[2])
    
    if all_coords_x.size > 0:
        max_range = np.array([
            np.ptp(all_coords_x), # ptp = peak-to-peak (max-min)
            np.ptp(all_coords_y),
            np.ptp(all_coords_z)
        ]).max() / 2.0
        
        # Handle case where all points are identical (max_range = 0)
        if max_range == 0:
            max_range = 10.0 # Default to 10m
        
        mid_x = (all_coords_x.max()+all_coords_x.min()) * 0.5
        mid_y = (all_coords_y.max()+all_coords_y.min()) * 0.5
        mid_z = (all_coords_z.max()+all_coords_z.min()) * 0.5
        
        # Add some padding to the limits
        padding = max_range * 0.5
        ax.set_xlim(mid_x - max_range - padding, mid_x + max_range + padding)
        ax.set_ylim(mid_y - max_range - padding, mid_y + max_range + padding)
        ax.set_zlim(mid_z - max_range - padding, mid_z + max_range + padding)

    ax.legend()
    
    # Adjust viewpoint for better visibility
    ax.view_init(elev=20, azim=-60) 
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()
    print("Plot displayed.")

# --- Main execution block ---
if __name__ == "__main__":
    
    # 1. Initialize the Environment
    grid_size_meters = 20 # RSO fits in a 20x20x20m cube
    voxel_resolution = 1.0 # Each voxel is 1x1x1m
    grid_dimensions = (int(grid_size_meters/voxel_resolution),)*3 # (20,20,20)
    grid_origin = (-grid_size_meters/2,)*3 # Centered at (0,0,0)
    
    grid = VoxelGrid(grid_dims=grid_dimensions, voxel_size=voxel_resolution, origin=grid_origin)
    rso = GroundTruthRSO(grid) # The actual RSO, hidden from the agent logic
    
    # 2. Get Initial State
    entropy_before = grid.get_entropy()
    
    # 3. Simulate one step
    # --- New Camera Parameters ---
    FOV_DEGREES = 30.0
    SENSOR_RESOLUTION = (50, 50) # 50x50 pixels
    # These parameters define the *simulation's* noise.
    # FIX: Using much more realistic noise parameters
    SENSOR_NOISE_PARAMS = {
        'p_hit_given_occupied': 0.95, # 95% chance of correctly seeing a hit
        'p_hit_given_empty': 0.001     # 0.1% chance of a false positive
    }
    
    # Generate random camera position
    camera_position = get_random_rtn_position(min_radius=25.0, max_radius=35.0)
    
    # Calculate view direction to point at RSO center of mass (0,0,0)
    target_com_world = np.array([0, 0, 0])
    view_direction = target_com_world - camera_position
    view_direction = view_direction / np.linalg.norm(view_direction) # Normalize
    
    print(f"\nSimulating from random position: {camera_position.round(2)}")
    print(f"View direction: {view_direction.round(2)}")
    
    # Take an observation
    hits, misses = simulate_observation(
        camera_position, 
        view_direction, 
        FOV_DEGREES, 
        SENSOR_RESOLUTION,
        SENSOR_NOISE_PARAMS,
        grid, 
        rso
    )
    
    # Store "before" belief for gain calculation
    b_before = np.copy(grid.belief)
    
    # Update the belief grid using the observation
    grid.update_belief(hits, misses)
    
    # 4. Calculate Results
    entropy_after = grid.get_entropy()
    info_gain = calculate_information_gain(b_before, grid.belief)
    
    print("\n--- Simulation Results ---")
    print(f"Entropy (Before): {entropy_before:.2f}")
    print(f"Entropy (After):  {entropy_after:.2f}")
    print(f"Total Info Gain:  {info_gain:.2f}")
    
    # 5. Visualize
    plot_scenario(grid, rso, camera_position, view_direction, FOV_DEGREES, SENSOR_RESOLUTION)

