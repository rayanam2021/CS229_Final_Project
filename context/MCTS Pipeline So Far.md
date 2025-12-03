
## 1. High-Level Overview

At each simulation step, the system:

1. Maintains:
   - The **relative orbital state** in ROE coordinates.
   - The **belief** over the target’s 3D shape as a `VoxelGrid`.
   - The **ground-truth RSO** shape.
   - A **camera model** (`camera_fn`).

2. Builds an `MCTSController`, which internally owns:
   - An `OrbitalMCTSModel` (`mcts/orbital_mdp_model.py`).
   - An `MCTS` object (`mcts/mcts.py`).

3. Runs MCTS for a given number of iterations (`mcts_iters`) and depth (`horizon`) to select the **best root action**.

4. Executes **only the first action** of the best sequence in the real environment:
   - Propagates the orbit.
   - Simulates a camera observation.
   - Updates the real `VoxelGrid` belief.
   - Logs state, action, reward, etc.
   - Repeats for the next time step.

---

## 2. MDP Formulation Used by MCTS

### 2.1 State

Each MCTS state is an `OrbitalState`:

- Physical part (ROE):
  $$
  \delta \mathbf{x} = [\Delta a,\ \Delta \lambda,\ \Delta e_x,\ \Delta e_y,\ \Delta i_x,\ \Delta i_y]^\top
  $$
- Belief part: a `VoxelGrid` with belief values
  $$
  b_i = \Pr(\text{voxel } i \text{ is occupied}), \quad i = 1, \dots, N_{\text{voxels}}
  $$

So:
$$
s = (\delta \mathbf{x}, \mathbf{b})
$$

where $\mathbf{b}$ is represented internally either as NumPy arrays or torch tensors.

### 2.2 Actions

`OrbitalMCTSModel.actions(state)` returns a discrete set of 13 impulsive Δv actions in RTN:

- One **no-op**: $\mathbf{a}_0 = [0, 0, 0]^\top$
- For each axis $k \in \{r,t,n\}$ (radial, tangential, normal):
  - Small ±Δv: $\pm \delta v_{\text{small}}$
  - Large ±Δv: $\pm \delta v_{\text{large}}$

So the action space is:
$$
\mathcal{A} = \{ \mathbf{0} \} \cup \{\pm \delta v_{\text{small}}\mathbf{e}_k, \pm \delta v_{\text{large}}\mathbf{e}_k\}_{k \in \{r,t,n\}}
$$

### 2.3 Transition (`OrbitalMCTSModel.step`)

Given state $s = (\delta \mathbf{x}, \mathbf{b})$ and action $\mathbf{a} = \Delta \mathbf{v}_{RTN}$:

1. **Apply impulsive Δv to ROE**  
   Using a GVE-based mapping:
   $$
   \delta \mathbf{x}' = \delta \mathbf{x} + \Delta \delta \mathbf{x}(\Delta \mathbf{v}_{RTN})
   $$

2. **Propagate relative motion** over one time step $ \Delta t $ using `propagateGeomROE` to get RTN relative position and velocity, then convert back to ROE with `rtn_to_roe`.

3. **Clone the belief grid**:
   ```python
   grid = state.grid.clone()
   ```

4. **Compute entropy before observation**:
   $$
   H_{\text{before}} = -\sum_i \left[ b_i \log b_i + (1 - b_i)\log(1 - b_i) \right]
   $$

   Implemented via `grid.get_entropy()` which calls `calculate_entropy`.

5. **Simulate camera observation** from the new relative position, updating `grid` in-place:
   ```python
   simulate_observation(grid, self.rso, self.camera_fn, pos_child)
   ```

6. **Compute entropy after observation**:
   $$
   H_{\text{after}} = -\sum_i \left[ b'_i \log b'_i + (1 - b'_i)\log(1 - b'_i) \right]
   $$

7. **Information gain**:
   $$
   I = H_{\text{before}} - H_{\text{after}}
   $$

8. **Δv cost**:
   $$
   C_{\Delta v} = \|\Delta \mathbf{v}_{RTN}\|_2
   $$

9. **Reward definition**:
   $$
   r = I - \lambda_{\Delta v} \cdot C_{\Delta v}
   $$

The next state is:
$$
s' = (\delta \mathbf{x}', \mathbf{b}')
$$
and `step` returns `(next_state, reward)`.

### 2.4 Discounting

MCTS uses a discount factor $\gamma$ during rollouts and backpropagation:

- Immediate rewards from `step` are combined as:
  $$
  G = r_0 + \gamma r_1 + \gamma^2 r_2 + \dots
  $$

---

## 3. MCTS Algorithm in This Code

### 3.1 Nodes and Statistics

Each node stores:

- `state`: an `OrbitalState`.
- `actions`: list of available actions.
- `children`: child nodes.
- `untried_action_indices`: which actions haven’t been expanded yet.
- Visit counts:
  - `N`: total visits to this node.
  - `N_sa[i]`: visits of action $a_i$ from this node.
- Value estimates:
  - `Q_sa[i]`: estimated return for action $a_i$ from this node.
- `reward`: immediate reward of the edge from parent → this node.
- `action_index`: which action from the parent led here.

### 3.2 Main Loop

`MCTS.get_best_root_action(root_state, ...)`:

1. Construct root node with:
   - `state = root_state`
   - `actions = model.actions(root_state)`

2. For `iters` times:
   - Call `self._search(root, depth=0)`.

3. After the search:
   - Choose the best root action as:
     $$
     a^* = \arg\max_i Q_{sa}[i]
     $$

3. Return $(a^*, Q_{sa}[a^*])$ and some statistics.

### 3.3 Tree Policy (Selection)

For a fully expanded node (no untried actions), `_select_ucb1_action_index` uses UCB1:

$$
\text{UCB1}_i = Q_{sa}[i] + c \sqrt{\frac{\ln N}{N_{sa}[i]}}
$$

- $c$ is the exploration constant (`mcts_c`).
- $N$ is the total number of visits to this node.
- $N_{sa}[i]$ is the number of times action $a_i$ has been chosen here.

The action with largest $\text{UCB1}_i$ is selected for expansion deeper into the tree.

### 3.4 Expansion

If a node has untried actions:

1. Pop an `action_index` from `untried_action_indices`.
2. Call `_expand(node, action_index)`:
   - `next_state, reward = self.mdp.step(node.state, action)`
   - `next_actions = self.mdp.actions(next_state)`
   - Create a new child node with these values.
3. Immediately perform a rollout from the child.

### 3.5 Rollouts (Simulation Policy)

`_rollout(state, depth)`:

- While `depth < max_depth`:
  1. Get available actions `self.mdp.actions(state)`.
  2. Sample an action via `self.mdp.rollout_policy(state)`.
  3. Call `next_state, reward = self.mdp.step(state, action)`.
  4. Accumulate discounted rewards:
     $$
     G \leftarrow G + \gamma^k r_k
     $$
  5. Move to `next_state` and increment depth.

- Return total discounted return `G`.

> **Important:** `step` is the same function used during expansion and rollouts, so **full camera observations and belief updates are performed in every rollout step**, on cloned voxel grids.

### 3.6 Backpropagation

Given a leaf node and a rollout return `simulation_return`:

- Start with `G = simulation_return` at the leaf.
- Traverse up to root:

  - At each node (except the root), define:
    $$
    G \leftarrow r_{\text{edge}} + \gamma G
    $$
    where $r_{\text{edge}}$ is the node’s `reward` (from its parent).

  - Update parent’s statistics for the corresponding action index $i$:
    - `N_sa[i] += 1`
    - Update running mean:
      $$
      Q_{sa}[i] \leftarrow Q_{sa}[i] + \frac{G - Q_{sa}[i]}{N_{sa}[i]}
      $$

- Also increment `N` (visit count) for each node.

---

## 4. GPU Usage and Torch-Backed Grids

### 4.1 What Goes to the GPU

When `VoxelGrid` is created with `use_torch=True` and the device is CUDA:

- `belief` and `log_odds` are torch tensors on the GPU.
- Sensor model log-odds parameters are also stored as torch tensors.
- `calculate_entropy` detects torch tensors and performs all math (log, multiply, sum) in torch.

This means:

- Log-odds updates and entropy computation on the belief grid can be GPU-accelerated.
- Ray-tracing itself (3D DDA) is still done in NumPy on CPU and then converted to masks for torch.

### 4.2 What Stays on CPU

- Orbital propagation (`apply_impulsive_dv`, `propagateGeomROE`, `rtn_to_roe`) is NumPy on CPU.
- MCTS tree logic and Python loops.
- Ray generation and DDA traversal in the camera model.

So the GPU is currently **optimizing only the belief grid math**, not the entire pipeline.

---

## 5. Rollout Policy Heuristic (With Math)

The rollout policy is implemented inside `OrbitalMCTSModel.rollout_policy`. The idea:

- Assign a **score** $s(a)$ to each candidate action $a$.
- Convert scores into probabilities with a softmax:
  $$
  p(a) = \frac{\exp(s(a))}{\sum_{a' \in \mathcal{A}_{\text{valid}}} \exp(s(a'))}
  $$
- Sample the rollout action from this distribution.

### 5.1 Score Components

Let:

- $\Delta \mathbf{v} = a$ be the action’s RTN Δv.
- $\| \Delta \mathbf{v} \| = \text{dv\_norm}$.
- `alpha_dv` > 0: penalty weight for Δv magnitude.
- `beta_tan`: bonus for tangential/normal actions.
- `target_radius` = $r^\star$: desired orbital radius (in meters).
- `r_min_rollout`, `r_max_rollout`: allowed radius envelope during rollouts.

The score for an action is roughly:

$$
s(a) = s_{\Delta v}(a) + s_{\text{axis}}(a) + s_r(a)
$$

#### 5.1.1 Δv Magnitude Penalty

A Boltzmann-like penalty:
$$
s_{\Delta v}(a) = -\alpha_{\Delta v} \cdot \|\Delta \mathbf{v}\|_2
$$

This makes small-magnitude Δv more likely in rollouts, encouraging fuel-efficient behavior.

#### 5.1.2 Axis Preference (Parallax Heuristic)

Let `main_axis = argmax(|Δv_k|)` over $k \in \{r,t,n\}$.

- If the main axis is tangential or normal (i.e., $k\in\{t,n\}$), then:
  $$
  s_{\text{axis}}(a) = \beta_{\text{tan}}
  $$
- Otherwise:
  $$
  s_{\text{axis}}(a) = 0
  $$

This encodes the idea that tangential/normal maneuvers are better for inducing parallax and diverse viewing geometries.

#### 5.1.3 Radius-Based Shaping and Constraints

For each candidate action $a$:

1. Apply an impulsive Δv and single-step propagation to get an approximate child radius $r(a)$.

2. If $r(a)$ is outside the allowed envelope:
   $$
   r(a) \notin [r_{\min}^{\text{rollout}}, r_{\max}^{\text{rollout}}]
   $$
   then **reject** this action from consideration (it will not be in $\mathcal{A}_{\text{valid}}$).

3. Optionally, a shaping term could be:
   $$
   s_r(a) = -\gamma_r \cdot |r(a) - r^\star|
   $$
   favoring actions that move the servicer towards the preferred radius.

(Your code uses this kind of shaping idea conceptually; the exact weighting and clipping behavior is tuned inside `rollout_policy`.)

### 5.2 Softmax Sampling

For the remaining valid actions $\mathcal{A}_{\text{valid}}$:

$$
p(a) = \frac{\exp(s(a))}{\sum_{a' \in \mathcal{A}_{\text{valid}}} \exp(s(a'))}
$$

If no actions pass the radius filter, the policy falls back to a simpler softmax distribution relying primarily on the Δv magnitude term.

Thus, rollouts are:

- **Stochastic**: provide exploration in the planning phase.
- **Heuristic-guided**: biased towards:
  - small Δv,
  - tangential/normal impulses,
  - safe and informative radii.

---

## 6. Observations During Rollouts

A key design choice: MCTS rollouts are **not cheap surrogate simulations** right now — they are **full simulations**:

- Every call to `step` during a rollout:
  - Applies orbital dynamics.
  - Clones the `VoxelGrid`.
  - Runs the full camera `simulate_observation`:
    - Generates rays,
    - Runs 3D DDA through the grid,
    - Updates log-odds and belief.
  - Computes entropy before/after.
  - Computes info gain and Δv cost.
  - Returns the new state and reward.

Because the grid is cloned at each step, each branch of the MCTS tree maintains its **own independent belief evolution**. The real environment/grid is only updated once per real time step when you execute the chosen root action.

This is **computationally expensive** but conceptually straightforward:
- The planning model and the real environment share the exact same sensor model and reward.

---

## 7. Current Limitations and Opportunities

1. **Performance Bottleneck**:
   - Full ray-tracing + entropy computation in all rollouts is extremely expensive.
   - This limits `mcts_iters` and `horizon`.

2. **GPU Underuse**:
   - GPU only accelerates belief grid math (log-odds and entropy).
   - Ray-casting and dynamics remain CPU-bound.

3. **Possible Future Directions**:
   - Introduce a **cheap surrogate reward** for rollouts (e.g., radius + Δv shaping only), and only compute real info gain at:
     - The root, or
     - A sparse subset of nodes.
   - Use a learned model or approximate ray-tracing for belief updates during rollouts.
   - Parallelize MCTS across CPU cores or GPU (e.g., batched rollouts, parallel tree expansion).

---

## 8. Summary

- The current pipeline implements a **full-information MCTS**:
  - State = (ROE, voxel-grid belief).
  - Reward = entropy reduction − Δv penalty.
  - Transitions include full orbital dynamics + camera observation.
- MCTS:
  - Uses UCB1 for tree search.
  - Uses a **radius-aware, Δv-penalized, axis-biased** rollout policy with softmax sampling.
- GPU:
  - Accelerates belief grid log-odds updates and entropy, but not ray-tracing or tree search.
- Rollouts:
  - Perform full observations and belief updates on deep-copied grids.
  - Are therefore accurate but costly.

This document reflects how the system is currently configured and serves as a reference for understanding and modifying the approach (e.g., swapping in cheap rewards, changing rollout policies, or restructuring GPU use).

## 9. GPU vs CPU Performance Considerations

### 9.1 Why GPU and CPU Runtime Are Similar

Although the VoxelGrid supports GPU acceleration through torch tensors, only a small portion of the workload currently runs on the GPU:

- **GPU-accelerated:**
    - Log-odds updates
    - Entropy computation (`calculate_entropy`)
        
- **CPU-bound (dominant cost):**
    - Full ray-tracing (3D DDA) through the voxel grid
    - Ray generation for the camera model
    - Orbital propagation (NumPy)
    - MCTS tree logic (Python loops, recursion, sampling during rollouts)
        

Since raycasting and Python overhead dominate the runtime, GPU acceleration only affects a small fraction of total compute time, resulting in **nearly identical CPU and GPU performance**.

### 9.2 Why Entropy Reduction Is Still Strong with Few Iterations

Even with shallow tree depth and few MCTS iterations, the planner shows surprisingly strong entropy reduction due to several structural advantages:

1. **Full-info reward in rollouts**  
    Each `step()` during planning computes real information gain:
    - full camera simulation
    - belief update
    - entropy before/after
    - Δv penalty  
        This makes each simulated trajectory highly informative.
        
2. **Strong rollout policy**  
    The rollout policy is not random; it incorporates:
    - Δv magnitude penalties
    - preference for tangential/normal Δv (parallax)
    - radius shaping toward a target sensing distance
    - filtering of actions outside the allowed radius envelope
        
3. **Small action space (13 actions)**  
    With a small discrete action set, even limited sampling can effectively distinguish promising maneuvers.
    
4. **Closed-loop replanning each timestep**  
    Shallow MCTS depth is compensated by frequent replanning, with new real observations updating the belief grid at every step.
    

The combination makes the algorithm perform well even under limited compute budgets.

### 9.3 When GPU Performance Would Matter More

GPU acceleration would become impactful if any of the following were moved to the GPU:
- Batched ray-casting (3D DDA implemented in CUDA)
- Batched rollouts (parallel simulation of multiple MCTS playouts)
- Larger voxel grids (e.g., 50³–100³)
- Learned models to approximate info gain, reducing ray-tracing frequency

Under these conditions, GPU parallelism could dramatically reduce planning time.

### 9.4 Summary

- GPU acceleration is currently lightly used because **the bottlenecks are CPU‑bound**.
- Despite this, **MCTS performs very well** thanks to informative rewards, a strong rollout policy, and closed‑loop correction.
- Significant GPU speedups would require porting ray‑tracing or rollout batching to CUDA.