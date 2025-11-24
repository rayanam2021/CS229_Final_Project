import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import sys
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation as R

# --- Helper functions for Log-Odds Bayesian Update ---

def logit(p: np.ndarray) -> np.ndarray:
    """Converts probabilities (p) to log-odds (L)."""
    p = np.clip(p, 1e-6, 1.0 - 1e-6) # Clip to avoid log(0)
    return np.log(p / (1.0 - p))

def sigmoid(L: np.ndarray) -> np.ndarray:
    """Converts log-odds (L) back to probabilities (p)."""
    return 1.0 / (1.0 + np.exp(-L))

# --- Original Information Gain Functions ---

def calculate_entropy(belief: np.ndarray) -> float:
    """
    Computes Shannon entropy of the voxel grid belief.
    """
    eps = 1e-9
    belief_clipped = np.clip(belief, eps, 1 - eps)
    entropy = -np.sum(belief_clipped * np.log(belief_clipped) +
                      (1 - belief_clipped) * np.log(1 - belief_clipped))
    return float(entropy)

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
        P_HIT_GIVEN_OCCUPIED = 0.95
        P_HIT_GIVEN_EMPTY = 0.001
        P_MISS_GIVEN_OCCUPIED = 1.0 - P_HIT_GIVEN_OCCUPIED
        P_MISS_GIVEN_EMPTY = 1.0 - P_HIT_GIVEN_EMPTY

        self.L_hit_update = logit(np.array(P_HIT_GIVEN_OCCUPIED)) - logit(np.array(P_HIT_GIVEN_EMPTY)) 
        self.L_miss_update = logit(np.array(P_MISS_GIVEN_OCCUPIED)) - logit(np.array(P_MISS_GIVEN_EMPTY))

    def clone(self):
        new = VoxelGrid(self.dims, self.voxel_size, tuple(self.origin))
        new.belief = self.belief.copy()
        new.log_odds = self.log_odds.copy()
        return new

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
        
        hit_mask = np.zeros(self.dims, dtype=bool)
        if hit_voxels:
            valid_hits = [idx for idx in hit_voxels if self.is_in_bounds(np.array(idx))]
            if valid_hits:
                hit_indices = tuple(np.array(valid_hits).T)
                hit_mask[hit_indices] = True
        
        miss_mask = np.zeros(self.dims, dtype=bool)
        if missed_voxels:
            valid_misses = [idx for idx in missed_voxels if self.is_in_bounds(np.array(idx))]
            if valid_misses:
                miss_indices = tuple(np.array(valid_misses).T)
                miss_mask[miss_indices] = True

        self.log_odds[hit_mask] += self.L_hit_update
        self.log_odds[miss_mask] += self.L_miss_update
            
        self.belief = sigmoid(self.log_odds)


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
    sensor_height_half = np.tan(fov_rad / 2.0) 
    sensor_width_half = sensor_height_half * aspect_ratio
    
    rays = []
    for u in range(sensor_res[0]):
        for v in range(sensor_res[1]):
            norm_u = (u + 0.5) / sensor_res[0] * 2.0 - 1.0
            norm_v = (v + 0.5) / sensor_res[1] * 2.0 - 1.0
            
            ray_dir = (view_dir + 
                       cam_right * norm_u * sensor_width_half + 
                       cam_up * norm_v * sensor_height_half)
            
            ray_dir /= np.linalg.norm(ray_dir)
            rays.append(ray_dir)
            
    return np.array(rays)



def _trace_ray(ray_origin: np.ndarray, ray_dir: np.ndarray,
               grid: VoxelGrid, rso: GroundTruthRSO, noise_params: dict):
    """
    Traces a ray through the voxel grid using 3D DDA (Amanatides–Woo).
    """

    # --- 1. Bounding box intersection ---
    eps = sys.float_info.epsilon
    ray_dir_safe = np.where(np.abs(ray_dir) < eps, np.sign(ray_dir) * eps + eps, ray_dir)

    t1 = (grid.origin - ray_origin) / ray_dir_safe
    t2 = (grid.max_bound - ray_origin) / ray_dir_safe

    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)

    t_enter = np.max(t_min)
    t_exit = np.min(t_max)

    if t_enter > t_exit or t_exit < 0:
        return None, "miss", []

    if t_enter < 0:
        start_point_world = ray_origin
        t_enter = 0.0
    else:
        start_point_world = ray_origin + ray_dir * t_enter

    # --- 2. Initialize traversal ---
    current_voxel = grid.world_to_grid_coords(start_point_world)
    if not grid.is_in_bounds(current_voxel):
        # This can happen if ray starts inside but numerical error
        # pushes start_point_world just outside grid.
        return None, "miss", []

    step = np.sign(ray_dir).astype(int)
    step[step == 0] = 1

    voxel_size = grid.voxel_size
    t_delta = np.abs(voxel_size / ray_dir_safe)

    voxel_boundary = grid.origin + (current_voxel + (step > 0)) * voxel_size
    t_max = (voxel_boundary - ray_origin) / ray_dir_safe

    missed_voxels = []

    # --- 3. DDA traversal ---
    max_steps = int(np.sum(grid.dims))
    for _ in range(max_steps):
        if not grid.is_in_bounds(current_voxel):
            break

        voxel_tuple = tuple(current_voxel)
        missed_voxels.append(voxel_tuple)

        # --- Hit test ---
        if rso.shape[voxel_tuple]:
            # True occupied voxel (part of RSO)
            if np.random.rand() < noise_params["p_hit_given_occupied"]:
                return voxel_tuple, "hit", missed_voxels[:-1]
            else:
                continue # Sensor failed to detect
        else:
            # False positive test for empty space
            if np.random.rand() < noise_params["p_hit_given_empty"]:
                return voxel_tuple, "hit", missed_voxels[:-1]

        # --- Move to next voxel ---
        axis = np.argmin(t_max)
        t_enter = t_max[axis]
        t_max[axis] += t_delta[axis]
        current_voxel[axis] += step[axis]

        if t_enter > t_exit:
            break

    return None, "miss", missed_voxels


def simulate_observation(grid: VoxelGrid, rso, camera_fn: dict, servicer_rtn: np.ndarray):
    """
    Simulates an observation from a servicer at given RTN position.
    Updates the voxel grid belief according to hits/misses.
    """
    camera_pos = servicer_rtn[-1] if servicer_rtn.ndim > 1 else servicer_rtn
    view_dir = -camera_pos / np.linalg.norm(camera_pos)

    rays = get_camera_rays(camera_pos, view_dir, camera_fn['fov_degrees'], camera_fn['sensor_res'])

    all_hit_voxels, all_missed_voxels = set(), set()
    for ray_dir in rays:
        hit_voxel_idx, status, missed_list = _trace_ray(
            camera_pos, ray_dir, grid, rso, camera_fn['noise_params']
        )
        all_missed_voxels.update(missed_list)
        if status == 'hit':
            all_hit_voxels.add(hit_voxel_idx)

    grid.update_belief(list(all_hit_voxels), list(all_missed_voxels))
    return list(all_hit_voxels), list(all_missed_voxels)


def draw_spacecraft(ax, position, direction, color="gray", scale=(6.0, 4.0, 3.0)):
    """
    Draws a simple rectangular-prism spacecraft centered at `position`,
    oriented along `direction`.
    """
    direction = direction / np.linalg.norm(direction)

    global_z = np.array([0, 0, 1])
    if np.allclose(np.abs(direction), global_z):
        temp_up = np.array([0, 1, 0])
    else:
        temp_up = global_z
        
    right = np.cross(direction, temp_up)
    right /= np.linalg.norm(right)
    
    up = np.cross(right, direction)
    up /= np.linalg.norm(up)

    L, W, H = scale
    scaled_direction_vec = direction * (L / 2)
    scaled_right_vec = right * (W / 2)
    scaled_up_vec = up * (H / 2)

    corners = np.array([
        position + scaled_direction_vec + scaled_right_vec + scaled_up_vec,
        position + scaled_direction_vec - scaled_right_vec + scaled_up_vec,
        position + scaled_direction_vec - scaled_right_vec - scaled_up_vec,
        position + scaled_direction_vec + scaled_right_vec - scaled_up_vec,
        position - scaled_direction_vec + scaled_right_vec + scaled_up_vec,
        position - scaled_direction_vec - scaled_right_vec + scaled_up_vec,
        position - scaled_direction_vec - scaled_right_vec - scaled_up_vec,
        position - scaled_direction_vec + scaled_right_vec - scaled_up_vec
    ])

    faces = [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[3], corners[7], corners[4]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[3], corners[2], corners[6], corners[7]]
    ]

    box = Poly3DCollection(faces, facecolors=color, linewidths=0.3, edgecolors="k", alpha=0.9)
    ax.add_collection3d(box)
    return box


def plot_scenario(grid: VoxelGrid, rso: GroundTruthRSO, camera_pos_world: np.ndarray, 
                  view_direction: np.ndarray, fov_degrees: float, sensor_res: tuple,
                  fig=None, ax=None):
    """
    Generates or updates a 3D plot of the ground truth, belief state, and camera.
    Returns: fig, ax, artists_dict (containing artists to be updated)
    """
    
    initial_plot = (fig is None or ax is None)
    
    if initial_plot:
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('RSO Characterization Mission: Simulated Observation', fontsize=16)

        artists = {}

        # Plot Bounding Box
        min_c = grid.origin
        max_c = grid.max_bound
        corners = np.array([
            [min_c[0], min_c[1], min_c[2]], [max_c[0], min_c[1], min_c[2]],
            [max_c[0], max_c[1], min_c[2]], [min_c[0], max_c[1], min_c[2]],
            [min_c[0], min_c[1], max_c[2]], [max_c[0], min_c[1], max_c[2]],
            [max_c[0], max_c[1], max_c[2]], [min_c[0], max_c[1], max_c[2]]
        ])
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        bb_lines = []
        for (i, j) in edges:
            line, = ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]], 
                             [corners[i, 2], corners[j, 2]], c='gray', linestyle=':', linewidth=1)
            bb_lines.append(line)
        artists['bb_lines'] = bb_lines
        ax.plot([], [], [], c='gray', linestyle=':', label='Grid Bounding Box')

        # Plot Ground Truth (Hidden Surface Outline)
        truth_filled_voxels = np.argwhere(rso.shape)
        exposed_faces = [] 
        padded_shape = np.pad(rso.shape, pad_width=1, mode='constant', constant_values=False)
        for x, y, z in truth_filled_voxels:
            px, py, pz = x + 1, y + 1, z + 1
            if not padded_shape[px+1, py, pz]: exposed_faces.append(((x,y,z), 0))
            if not padded_shape[px-1, py, pz]: exposed_faces.append(((x,y,z), 1))
            if not padded_shape[px, py+1, pz]: exposed_faces.append(((x,y,z), 2))
            if not padded_shape[px, py-1, pz]: exposed_faces.append(((x,y,z), 3))
            if not padded_shape[px, py, pz+1]: exposed_faces.append(((x,y,z), 4))
            if not padded_shape[px, py, pz-1]: exposed_faces.append(((x,y,z), 5))

        face_verts = []
        for (x, y, z), face_dir in exposed_faces: 
            origin_voxel = grid.origin + np.array([x, y, z]) * grid.voxel_size
            size = grid.voxel_size
            c = [
                origin_voxel, origin_voxel + [size, 0, 0], origin_voxel + [0, size, 0], 
                origin_voxel + [0, 0, size], origin_voxel + [size, size, 0], 
                origin_voxel + [size, 0, size], origin_voxel + [0, size, size], 
                origin_voxel + [size, size, size]
            ]
            if face_dir == 0: face_verts.append([c[1], c[4], c[7], c[5]]) # +X
            elif face_dir == 1: face_verts.append([c[0], c[3], c[6], c[2]]) # -X
            elif face_dir == 2: face_verts.append([c[2], c[4], c[7], c[6]]) # +Y
            elif face_dir == 3: face_verts.append([c[0], c[1], c[5], c[3]]) # -Y
            elif face_dir == 4: face_verts.append([c[3], c[5], c[7], c[6]]) # +Z
            elif face_dir == 5: face_verts.append([c[0], c[1], c[4], c[2]]) # -Z

        gt_lines = []
        if face_verts:
            edges_set = set()
            for face in face_verts:
                p1, p2, p3, p4 = face
                edges_set.add(tuple(sorted((tuple(p1), tuple(p2)))))
                edges_set.add(tuple(sorted((tuple(p2), tuple(p3)))))
                edges_set.add(tuple(sorted((tuple(p3), tuple(p4)))))
                edges_set.add(tuple(sorted((tuple(p4), tuple(p1)))))
            
            for p1_tuple, p2_tuple in edges_set:
                p1 = np.array(p1_tuple)
                p2 = np.array(p2_tuple)
                line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                                 color=(1.0, 0.6, 0.6), linewidth=0.5, alpha=0.5)
                gt_lines.append(line)
        artists['gt_lines'] = gt_lines
        ax.plot([], [], [], color=(1.0, 0.6, 0.6), linestyle='-', alpha=0.5, markersize=10, label='Ground Truth (Hidden Surface)')
        
        artists['belief_scatter'] = ax.scatter([], [], [], c='green', marker='o', s=50, label='Belief Voxels (P > 0.7)', depthshade=False)
        
        artists['servicer_path_line'], = ax.plot([camera_pos_world[0]], [camera_pos_world[1]], [camera_pos_world[2]], 
                                                 c='blue', linestyle='-', linewidth=2, alpha=0.5, label='Servicer Path')

        artists['servicer_prism'] = draw_spacecraft(ax, camera_pos_world, view_direction, color="gray", scale=(6.0, 4.0, 3.0))
        ax.plot([], [], [], 
        color='gray', 
        marker='s',        
        markersize=8,      
        linestyle='None',  
        label='Servicer Camera (Agent)')
        
        artists['view_line'] = ax.plot([], [], [], c='green', linestyle='--', linewidth=2, label='Viewing Direction (to CoM)')[0]
        artists['fov_lines'] = [ax.plot([], [], [], c='cyan', linestyle=':', linewidth=1)[0] for _ in range(8)] 
        artists['entropy_text'] = ax.text2D(0.05, 0.95, f"Entropy: {grid.get_entropy():.2f}", transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        ax.legend()
        ax.view_init(elev=20, azim=-60) 
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Axis limits will be set by the calling function (create_visualization_frames)

        return fig, ax, artists

    else:
        return fig, ax, artists


# --- MODIFIED update_plot FUNCTION ---

def update_plot(frame, grid, rso, camera_positions, view_directions, 
                fov_degrees, sensor_res, noise_params, ax, artists, target_com_world):
    """ Animation update function. This function only plots, it does NOT update the grid belief. """
    
    if frame >= len(camera_positions):
        return []

    camera_pos_world = camera_positions[frame]
    view_direction = view_directions[frame]
   
    # --- Update Artists ---
    belief_scatter = artists['belief_scatter']
    servicer_prism = artists['servicer_prism']
    servicer_path_line = artists['servicer_path_line']
    view_line = artists['view_line']
    fov_lines = artists['fov_lines']
    entropy_text = artists['entropy_text']

    # Update belief scatter
    certain_mask = (grid.belief > 0.7)
    certain_indices = np.argwhere(certain_mask)
    if certain_indices.size > 0:
        certain_world = grid.grid_to_world_coords(certain_indices)
        certain_probabilities = grid.belief[certain_indices[:,0], certain_indices[:,1], certain_indices[:,2]]
        colors = np.array([[0.0, 1.0, 0.0, alpha] for alpha in np.clip(certain_probabilities * 1.5 - 0.5, 0.3, 1.0)])
        belief_scatter._offsets3d = (certain_world[:, 0], certain_world[:, 1], certain_world[:, 2])
        belief_scatter.set_facecolors(colors)
    else:
        belief_scatter._offsets3d = ([], [], [])
        belief_scatter.set_facecolors([])

    # Update the Servicer Path Line
    path_x = camera_positions[:frame+1, 0]
    path_y = camera_positions[:frame+1, 1]
    path_z = camera_positions[:frame+1, 2]
    servicer_path_line.set_data_3d(path_x, path_y, path_z)

    # Remove old prism and draw new one
    servicer_prism.remove()
    new_prism = draw_spacecraft(ax, camera_pos_world, view_direction, color="gray", scale=(6.0, 4.0, 3.0))
    artists['servicer_prism'] = new_prism
    
    # Update view line
    view_line.set_data_3d([camera_pos_world[0], target_com_world[0]], 
                           [camera_pos_world[1], target_com_world[1]], 
                           [camera_pos_world[2], target_com_world[2]])

    # --- Update FOV Cone ---
    global_up = np.array([0, 0, 1])
    if np.allclose(np.abs(view_direction), global_up):
        global_up = np.array([0, 1, 0])
    cam_right_cross = np.cross(view_direction, global_up)
    if np.linalg.norm(cam_right_cross) < 1e-6:
        global_up = np.array([0, 1, 0])
        cam_right_cross = np.cross(view_direction, global_up)
    cam_right = cam_right_cross / np.linalg.norm(cam_right_cross)
    cam_up = np.cross(cam_right, view_direction)
    
    fov_rad = np.deg2rad(fov_degrees)
    aspect_ratio = sensor_res[0] / sensor_res[1]
    sensor_height_half = np.tan(fov_rad / 2.0)
    sensor_width_half = sensor_height_half * aspect_ratio
    
    corners_norm = [np.array([-1, -1]), np.array([1, -1]), np.array([1, 1]), np.array([-1, 1])]
    cone_length = np.linalg.norm(camera_pos_world - target_com_world)
    corner_points_world = []
    for norm_uv in corners_norm:
        norm_u, norm_v = norm_uv
        ray_dir = (view_direction + cam_right * norm_u * sensor_width_half + cam_up * norm_v * sensor_height_half)
        ray_dir /= np.linalg.norm(ray_dir)
        corner_point = camera_pos_world + ray_dir * cone_length
        corner_points_world.append(corner_point)
    
    for i in range(4):
        fov_lines[i].set_data_3d([camera_pos_world[0], corner_points_world[i][0]], 
                                 [camera_pos_world[1], corner_points_world[i][1]], 
                                 [camera_pos_world[2], corner_points_world[i][2]])
    cp = corner_points_world
    fov_lines[4].set_data_3d([cp[0][0], cp[1][0]], [cp[0][1], cp[1][1]], [cp[0][2], cp[1][2]])
    fov_lines[5].set_data_3d([cp[1][0], cp[2][0]], [cp[1][1], cp[2][1]], [cp[1][2], cp[2][2]])
    fov_lines[6].set_data_3d([cp[2][0], cp[3][0]], [cp[2][1], cp[3][1]], [cp[2][2], cp[3][2]])
    fov_lines[7].set_data_3d([cp[3][0], cp[0][0]], [cp[3][1], cp[0][1]], [cp[3][2], cp[0][2]])

    # --- Update entropy text ---
    current_entropy = grid.get_entropy()
    total_frames = len(camera_positions) if camera_positions is not None else frame + 1
    entropy_text.set_text(f"Entropy: {current_entropy:.2f}")
    ax.set_title(f'RSO Characterization - Frame {frame+1}/{total_frames}')
    print(f"Frame {frame+1}/{total_frames}, Entropy: {current_entropy:.2f}") # Log for user

    artists_list = [belief_scatter, view_line, entropy_text, new_prism, servicer_path_line]
    artists_list.extend(fov_lines)
    return artists_list