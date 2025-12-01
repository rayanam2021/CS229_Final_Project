import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

# --- Helper functions (Unchanged) ---
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))

def sigmoid(L: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-L))

def calculate_entropy(belief: np.ndarray) -> float:
    eps = 1e-9
    belief_clipped = np.clip(belief, eps, 1 - eps)
    entropy = -np.sum(belief_clipped * np.log(belief_clipped) +
                      (1 - belief_clipped) * np.log(1 - belief_clipped))
    return float(entropy)

# --- Classes (Unchanged) ---
class VoxelGrid:
    def __init__(self, grid_dims=(20, 20, 20), voxel_size=1.0, origin=(-10, -10, -10)):
        self.dims = grid_dims
        self.voxel_size = voxel_size
        self.origin = np.array(origin)
        self.max_bound = self.origin + np.array(self.dims) * self.voxel_size
        self.belief = np.full(self.dims, 0.5)
        self.log_odds = logit(self.belief)
        
        P_HIT_OCC = 0.95
        P_HIT_EMP = 0.001
        self.L_hit = logit(np.array(P_HIT_OCC)) - logit(np.array(P_HIT_EMP))
        self.L_miss = logit(np.array(1-P_HIT_OCC)) - logit(np.array(1-P_HIT_EMP))

    def get_entropy(self) -> float:
        return calculate_entropy(self.belief)

    def grid_to_world_coords(self, indices: np.ndarray) -> np.ndarray:
        if indices.size == 0: return np.array([])
        return self.origin + (indices + 0.5) * self.voxel_size

    def world_to_grid_coords(self, world_pos: np.ndarray) -> np.ndarray:
        if world_pos.ndim == 1: world_pos = world_pos[np.newaxis, :]
        indices = np.floor((world_pos - self.origin) / self.voxel_size)
        return indices.astype(int).squeeze()

    def is_in_bounds(self, grid_indices: np.ndarray) -> bool:
        if grid_indices.ndim == 1:
            return (all(grid_indices >= 0) and 
                    grid_indices[0] < self.dims[0] and
                    grid_indices[1] < self.dims[1] and
                    grid_indices[2] < self.dims[2])
        return np.all((grid_indices >= 0) & (grid_indices < self.dims), axis=1)

    def update_belief(self, hit_voxels: list, missed_voxels: list):
        hit_mask = np.zeros(self.dims, dtype=bool)
        if hit_voxels:
            valid = [idx for idx in hit_voxels if self.is_in_bounds(np.array(idx))]
            if valid: hit_mask[tuple(np.array(valid).T)] = True
        miss_mask = np.zeros(self.dims, dtype=bool)
        if missed_voxels:
            valid = [idx for idx in missed_voxels if self.is_in_bounds(np.array(idx))]
            if valid: miss_mask[tuple(np.array(valid).T)] = True
        self.log_odds[hit_mask] += self.L_hit
        self.log_odds[miss_mask] += self.L_miss
        self.belief = sigmoid(self.log_odds)

class GroundTruthRSO:
    def __init__(self, grid: VoxelGrid):
        self.dims = grid.dims
        self.shape = np.zeros(self.dims, dtype=bool)
        self._create_simple_shape()

    def _create_simple_shape(self):
        center = (self.dims[0] // 2, self.dims[1] // 2, self.dims[2] // 2)
        s = 4 
        # Main body
        self.shape[center[0]-s:center[0]+s, center[1]-s:center[1]+s, center[2]-s:center[2]+s] = True
        # Panel
        panel_length = 6
        self.shape[center[0]-s:center[0]+s, center[1]+s:center[1]+s+panel_length, center[2]-1:center[2]+1] = True
        # Antenna
        self.shape[center[0]-1:center[0]+1, center[1]-1:center[1]+1, center[2]+s:center[2]+s+3] = True

# --- Ray Tracing ---
def get_camera_rays(camera_pos, view_dir, fov_degrees, sensor_res):
    view_dir = view_dir / np.linalg.norm(view_dir)
    global_up = np.array([0, 0, 1])
    if np.allclose(np.abs(view_dir), global_up): global_up = np.array([0, 1, 0])
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

def _trace_ray(ray_origin, ray_dir, grid, rso, noise_params):
    eps = 1e-9
    ray_dir_safe = np.where(np.abs(ray_dir) < eps, np.sign(ray_dir) * eps + eps, ray_dir)
    t1 = (grid.origin - ray_origin) / ray_dir_safe
    t2 = (grid.max_bound - ray_origin) / ray_dir_safe
    t_min, t_max = np.minimum(t1, t2), np.maximum(t1, t2)
    t_enter, t_exit = np.max(t_min), np.min(t_max)

    if t_enter > t_exit or t_exit < 0: return None, "miss", []
    start_point = ray_origin if t_enter < 0 else ray_origin + ray_dir * t_enter
    
    curr = grid.world_to_grid_coords(start_point)
    if not grid.is_in_bounds(curr): return None, "miss", []

    step = np.sign(ray_dir).astype(int)
    step[step == 0] = 1
    t_delta = np.abs(grid.voxel_size / ray_dir_safe)
    bound = grid.origin + (curr + (step > 0)) * grid.voxel_size
    t_max_march = (bound - ray_origin) / ray_dir_safe
    
    missed = []
    for _ in range(int(np.sum(grid.dims)) * 2):
        if not grid.is_in_bounds(curr): break
        v_idx = tuple(curr)
        missed.append(v_idx)
        
        if rso.shape[v_idx]:
            if np.random.rand() < noise_params["p_hit_given_occupied"]:
                return v_idx, "hit", missed[:-1]
        elif np.random.rand() < noise_params["p_hit_given_empty"]:
            return v_idx, "hit", missed[:-1]

        axis = np.argmin(t_max_march)
        t_max_march[axis] += t_delta[axis]
        curr[axis] += step[axis]
    return None, "miss", missed

def simulate_observation(grid, rso, camera_fn, servicer_rtn):
    cam_pos = servicer_rtn[-1] if servicer_rtn.ndim > 1 else servicer_rtn
    view_dir = -cam_pos / np.linalg.norm(cam_pos)
    rays = get_camera_rays(cam_pos, view_dir, camera_fn['fov_degrees'], camera_fn['sensor_res'])
    
    hits, misses = set(), set()
    for r in rays:
        h, s, m = _trace_ray(cam_pos, r, grid, rso, camera_fn['noise_params'])
        misses.update(m)
        if s == 'hit': hits.add(h)
    
    grid.update_belief(list(hits), list(misses))
    return list(hits), list(misses)

# --- VISUALIZATION ---

def draw_spacecraft(ax, position, direction, color="gray", scale_factor=1.5):
    """
    Draws a spacecraft prism. 
    Base dimensions ~ 4x2x2 meters. 
    Default scale_factor=1.5 -> ~6x3x3 meters (Comparable to RSO which is ~8m).
    """
    direction = direction / np.linalg.norm(direction)
    global_z = np.array([0, 0, 1])
    temp_up = np.array([0, 1, 0]) if np.allclose(np.abs(direction), global_z) else global_z
        
    right = np.cross(direction, temp_up); right /= np.linalg.norm(right)
    up = np.cross(right, direction); up /= np.linalg.norm(up)

    # Base dimensions in meters
    L, W, H = 4.0, 2.0, 2.0 
    
    L *= scale_factor
    W *= scale_factor
    H *= scale_factor

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
    """
    Initialize the plot artists.
    """
    if fig is None:
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('T [m]')
        ax.set_zlabel('N [m]')
        ax.set_title('RSO Characterization Mission', fontsize=16)

        artists = {}

        # REMOVED: Grid Bounding Box lines

        # 1. Servicer Path Line (Starts Empty)
        artists['servicer_path_line'], = ax.plot([], [], [], 
                                                 c='blue', ls='-', lw=1.5, alpha=0.6, label='Trajectory')

        # 2. Burn Markers
        artists['burn_scatter'] = ax.scatter([], [], [], c='orange', marker='^', s=150, label='Maneuver', zorder=10)

        # 3. Servicer Spacecraft
        artists['servicer_prism'] = draw_spacecraft(ax, camera_pos_world, view_direction, scale_factor=1.5)
        ax.plot([], [], [], color='gray', marker='s', markersize=10, linestyle='None', label='Servicer')
        
        # 4. FOV Lines (Thicker, Magenta for visibility)
        artists['fov_lines'] = [ax.plot([], [], [], c='magenta', ls='-', lw=2.0, label='Camera FOV' if i==0 else None)[0] for i in range(8)]

        # 5. View Direction Line (Deep Green, thick)
        artists['view_line'] = ax.plot([], [], [], c='darkgreen', ls='--', lw=2.5, label='View Direction')[0]
        
        # 6. Belief Scatter
        artists['belief_scatter'] = ax.scatter([], [], [], c='green', marker='s', s=30, label='Belief (P>0.7)', depthshade=True)
        
        # 7. Ground Truth (Wireframe)
        truth_filled_voxels = np.argwhere(rso.shape)
        gt_lines = []
        
        # (Ground truth drawing logic unchanged for brevity, kept same as previous)
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
            if face_dir == 0: face_verts.append([c[1], c[4], c[7], c[5]]) 
            elif face_dir == 1: face_verts.append([c[0], c[3], c[6], c[2]])
            elif face_dir == 2: face_verts.append([c[2], c[4], c[7], c[6]]) 
            elif face_dir == 3: face_verts.append([c[0], c[1], c[5], c[3]]) 
            elif face_dir == 4: face_verts.append([c[3], c[5], c[7], c[6]]) 
            elif face_dir == 5: face_verts.append([c[0], c[1], c[4], c[2]]) 

        if face_verts:
            edges_set = set()
            for face in face_verts:
                p1, p2, p3, p4 = face
                edges_set.add(tuple(sorted((tuple(p1), tuple(p2)))))
                edges_set.add(tuple(sorted((tuple(p2), tuple(p3)))))
                edges_set.add(tuple(sorted((tuple(p3), tuple(p4)))))
                edges_set.add(tuple(sorted((tuple(p4), tuple(p1)))))
            
            for idx, (p1_tuple, p2_tuple) in enumerate(edges_set):
                p1 = np.array(p1_tuple); p2 = np.array(p2_tuple)
                label = 'Ground Truth' if idx == 0 else None
                line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                                 color=(1.0, 0.6, 0.6), linewidth=0.8, alpha=0.6, label=label)
                gt_lines.append(line)
        artists['gt_lines'] = gt_lines

        artists['entropy_text'] = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize='small')
        return fig, ax, artists
    return fig, ax, None

def update_plot(frame, grid, rso, camera_positions, view_directions, fov_degrees, sensor_res, noise_params, ax, artists, target_com, burn_indices=None):
    """
    Updates the plot for the given frame.
    """
    if frame >= len(camera_positions): return []

    cam_pos = camera_positions[frame]
    view_dir = view_directions[frame]
   
    # 1. Update Belief Scatter
    mask = (grid.belief > 0.7)
    indices = np.argwhere(mask)
    if indices.size > 0:
        world = grid.grid_to_world_coords(indices)
        probs = grid.belief[indices[:,0], indices[:,1], indices[:,2]]
        colors = np.array([[0.0, 1.0, 0.0, a] for a in np.clip(probs * 1.5 - 0.5, 0.3, 1.0)])
        artists['belief_scatter']._offsets3d = (world[:, 0], world[:, 1], world[:, 2])
        artists['belief_scatter'].set_facecolors(colors)
    else:
        artists['belief_scatter']._offsets3d = ([], [], [])

    # 2. Update Servicer Path (Growing Line)
    # Draw from start (0) to current (frame + 1) to show history
    path_x = camera_positions[:frame+1, 0]
    path_y = camera_positions[:frame+1, 1]
    path_z = camera_positions[:frame+1, 2]
    artists['servicer_path_line'].set_data_3d(path_x, path_y, path_z)

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
    # Use Fixed Scaling (1.5 -> ~6m) to keep it visible but not huge
    artists['servicer_prism'] = draw_spacecraft(ax, cam_pos, view_dir, scale_factor=1.5)
    
    # 5. Update View Direction Line
    artists['view_line'].set_data_3d([cam_pos[0], target_com[0]], [cam_pos[1], target_com[1]], [cam_pos[2], target_com[2]])
    
    # 6. Update FOV Lines
    global_up = np.array([0, 0, 1])
    if np.allclose(np.abs(view_dir), global_up): global_up = np.array([0, 1, 0])
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
        d = (view_dir + right*u*w_half + up*v*h_half)
        d /= np.linalg.norm(d)
        pts.append(cam_pos + d * dist)
        
    for i in range(4):
        artists['fov_lines'][i].set_data_3d([cam_pos[0], pts[i][0]], [cam_pos[1], pts[i][1]], [cam_pos[2], pts[i][2]])
    pts.append(pts[0])
    for i in range(4):
        artists['fov_lines'][4+i].set_data_3d([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]], [pts[i][2], pts[i+1][2]])

    artists['entropy_text'].set_text(f"Entropy: {grid.get_entropy():.2f}")
    
    return [artists['belief_scatter'], artists['servicer_path_line'], artists['servicer_prism']]