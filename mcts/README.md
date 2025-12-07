# MCTS Implementations

This directory contains various Monte Carlo Tree Search (MCTS) implementations for orbital planning.

## File Overview

### Core MCTS Implementations

#### `mcts.py` - Serial Pure MCTS
**Type:** Pure MCTS with UCB1
**Status:** ⚠️ Legacy (replaced by parallel version)
**Use case:** Reference implementation only

**Features:**
- Classic UCB1-based tree search
- Random rollout policy
- No neural network guidance
- Single-threaded execution

**When to use:** Don't use directly - use `mcts_parallel.py` instead for 5-8x speedup

---

#### `mcts_parallel.py` - **Parallel Pure MCTS** ✅ RECOMMENDED
**Type:** Pure MCTS with root parallelization
**Status:** ✅ **Currently used** by `run_pure_mcts.py`
**Use case:** Production pure MCTS planning

**Features:**
- **Root parallelization**: Runs N independent MCTS trees in parallel
- **5-8x speedup** over serial version using multiprocessing
- Each worker runs separate MCTS search with same root state
- Aggregates Q-values and visit counts from all workers
- **GPU support**: Uses CUDA ray tracing and ROE propagation
- **Multiprocessing**: Uses `spawn` method for CUDA compatibility

**Key optimizations:**
- Distributes `total_iters` across `num_workers` (default: CPU count - 1)
- Each worker has independent tree (no shared state complexity)
- Batched result aggregation with weighted averaging
- Automatic GPU/CPU detection and fallback

**Parameters:**
```python
ParallelMCTS(
    model,                    # OrbitalMCTSModel instance
    iters=3000,              # Total MCTS iterations (distributed across workers)
    max_depth=5,             # Maximum tree depth
    c=1.4,                   # UCB exploration constant
    gamma=1.0,               # Discount factor
    num_workers=None         # Auto-detect (CPU count - 1)
)
```

**Performance:**
- 11 workers on 12-core CPU
- ~273 iterations per worker (3000 / 11)
- Observed speedup: 5-8x (accounting for ~10% multiprocessing overhead)

**Example usage:**
```python
from mcts.mcts_parallel import ParallelMCTS
from mcts.orbital_mdp_model import OrbitalMCTSModel

# Initialize parallel MCTS
mcts = ParallelMCTS(
    model=mdp,
    iters=3000,
    num_workers=11  # or None for auto
)

# Run search
best_action, value, stats = mcts.get_best_root_action(
    root_state=state,
    step=0,
    out_folder="outputs/"
)
```

---

### Controller Wrappers

#### `mcts_controller.py` - Serial Controller
**Status:** ⚠️ Deprecated (replaced by parallel controller)
**Use case:** Reference only

**Features:**
- Wraps `MCTS` class from `mcts.py`
- Provides high-level interface for episode execution
- Manages replay buffer
- Fixed 3000 iterations (hardcoded)

**When to use:** Don't use - replaced by `mcts_controller_parallel.py`

---

#### `mcts_controller_parallel.py` - **Parallel Controller** ✅ RECOMMENDED
**Status:** ✅ **Currently used** by `scenario_full_mcts.py`
**Use case:** Production pure MCTS controller

**Features:**
- Wraps `ParallelMCTS` for root parallelization
- Same API as `MCTSController` (drop-in replacement)
- Configurable MCTS iterations and workers
- GPU support via `use_torch` and `device` parameters
- Manages replay buffer and episode data

**Key difference from serial:**
- Uses `ParallelMCTS` instead of `MCTS`
- Configurable `num_workers` parameter
- Records parallel timing stats

**Parameters:**
```python
MCTSControllerParallel(
    mu_earth, a_chief, e_chief, i_chief, omega_chief, n_chief,
    time_step=30.0,
    horizon=3,
    lambda_dv=0.0,
    mcts_iters=3000,      # Total iterations
    num_workers=None,     # Auto-detect
    use_torch=False,      # Enable GPU
    device='cpu'          # 'cuda' or 'cpu'
)
```

**Example usage:**
```python
from mcts.mcts_controller_parallel import MCTSControllerParallel

controller = MCTSControllerParallel(
    mu_earth, a_chief, e_chief, i_chief, omega_chief, n_chief,
    time_step=120.0,
    horizon=5,
    mcts_iters=3000,
    use_torch=True,      # GPU acceleration
    device='cuda'
)

action, value, stats = controller.select_action(
    state, time, tspan, grid, rso, camera_fn,
    step=0, out_folder="outputs/"
)
```

---

### AlphaZero MCTS

#### `mcts_alphazero_controller.py` - AlphaZero MCTS
**Type:** Neural network-guided MCTS with PUCT
**Status:** ✅ **Currently used** by AlphaZero training
**Use case:** AlphaZero self-play and evaluation

**Features:**
- **PUCT algorithm**: Polynomial Upper Confidence Trees
- Neural network guidance via policy and value networks
- Dirichlet noise for exploration at root
- Serial execution within each episode
- **Not batched** - uses single network forward pass per leaf

**Key differences from Pure MCTS:**
- Uses neural network predictions instead of random rollouts
- PUCT formula: `Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
- Requires `PolicyValueNetwork` for policy and value estimates
- Lower iteration count (100-500) since network guides search

**Why not batched?**
- Network evaluation is only 10-20% of runtime
- Physics simulation (`model.step()`) dominates (80-90%)
- Batching would provide only ~1.08x speedup
- Current parallel episodes approach gives 9x speedup instead

**Parameters:**
```python
MCTSAlphaZeroCPU(
    model,              # OrbitalMCTSModel instance
    network,            # PolicyValueNetwork
    c_puct=1.4,        # PUCT exploration constant
    n_iters=100,       # MCTS iterations (lower than pure MCTS)
    gamma=0.99,        # Discount factor
    device='cpu'       # 'cuda' or 'cpu'
)
```

**Example usage:**
```python
from mcts.mcts_alphazero_controller import MCTSAlphaZeroCPU
from learning.policy_value_network import PolicyValueNetwork

network = PolicyValueNetwork(grid_dims=(20,20,20), num_actions=13)
mcts = MCTSAlphaZeroCPU(
    model=mdp,
    network=network,
    c_puct=1.4,
    n_iters=100
)

policy, value, root = mcts.search(root_state)
```

---

### Supporting Files

#### `orbital_mdp_model.py` - MDP Formulation
**Type:** Environment/Model
**Status:** ✅ Used by all MCTS implementations
**Use case:** Defines state, actions, transitions, and rewards

**Key classes:**

**`OrbitalState`:**
- Encapsulates ROE (6D), VoxelGrid (20×20×20), and time
- Represents complete state of the system

**`OrbitalMCTSModel`:**
- Action space: 13 actions (no-op + 6 axes × 2 magnitudes)
- `step(state, action)`: Physics simulation + reward calculation
- `step_batch(state, actions)`: **Vectorized** version (13x faster)
- **GPU-accelerated**:
  - `batch_propagate_roe()` uses CUDA kernels
  - `simulate_observation()` uses GPU ray tracing
  - `calculate_entropy()` computed on GPU
  - Efficient grid cloning via `clone()` method

**Optimizations:**
- Batched ROE propagation for all children
- Pre-cloning grids to avoid `__init__` overhead
- Entropy calculations kept on GPU (single CPU←GPU transfer)
- Normalized rewards by initial entropy

**Key parameters:**
```python
OrbitalMCTSModel(
    a_chief, e_chief, i_chief, omega_chief, n_chief,
    rso,                    # Ground truth RSO
    camera_fn,              # Camera parameters
    grid_dims,              # Voxel grid dimensions
    lambda_dv,              # Fuel cost weight
    time_step,              # Propagation timestep
    max_depth,              # Planning horizon
    use_torch=False,        # Enable GPU
    device='cpu'            # 'cuda' or 'cpu'
)
```

---

## Comparison: Serial vs Parallel vs AlphaZero

| Feature | `mcts.py` (Serial) | `mcts_parallel.py` (Parallel) | `mcts_alphazero_controller.py` |
|---------|-------------------|-------------------------------|-------------------------------|
| **Search strategy** | UCB1 | UCB1 | PUCT |
| **Guidance** | Random rollouts | Random rollouts | Neural network |
| **Parallelization** | None | Root parallelization | Episode-level |
| **Workers** | 1 | 11 (default) | 1 per episode |
| **Speedup** | Baseline | 5-8x | N/A (different algorithm) |
| **Typical iterations** | 3000 | 3000 (distributed) | 100-500 |
| **GPU support** | Yes (via model) | Yes (via model) | Yes (via model + network) |
| **Use case** | Reference | **Pure MCTS production** | **AlphaZero production** |
| **Status** | Deprecated | ✅ **Recommended** | ✅ **Recommended** |

---

## Performance Summary

### Pure MCTS Performance

**Serial (`mcts.py`):**
- Single worker
- 3000 iterations
- Time per action: ~20-30s (CPU), ~5-10s (GPU)

**Parallel (`mcts_parallel.py`):** ✅ **Current**
- 11 workers (12-core CPU)
- 3000 total iterations (~273 per worker)
- Time per action: ~3-5s (GPU)
- **Speedup: 5-8x** over serial

### AlphaZero Performance

**Per episode:**
- 100 MCTS iterations (network-guided)
- 50 steps per episode
- Time: ~2 minutes per episode (GPU)

**Full training (65 episodes):**
- Serial: ~2 hours
- Parallel (11 workers): **~12-14 minutes**
- **Overall speedup: 9x** from parallel episodes

---

## When to Use Each Implementation

### Use `mcts_parallel.py` + `mcts_controller_parallel.py` when:
- ✅ Running pure MCTS without neural networks
- ✅ You want maximum planning quality
- ✅ You have multi-core CPU available
- ✅ You need 5-8x speedup over serial
- ✅ **This is the default for `run_pure_mcts.py`**

### Use `mcts_alphazero_controller.py` when:
- ✅ Training AlphaZero with self-play
- ✅ You have a trained neural network
- ✅ You want learning-based planning
- ✅ Lower iteration count acceptable (network guidance)
- ✅ **This is the default for `run_alphazero.py`**

### Don't use:
- ❌ `mcts.py` - Replaced by parallel version
- ❌ `mcts_controller.py` - Replaced by parallel version

---

## Configuration Examples

### Pure MCTS (`config_pure_mcts.json`)
```json
{
  "mcts": {
    "mcts_iters": 3000,
    "c_puct": 1.4,
    "gamma": 0.99,
    "num_workers": null  // Auto-detect
  }
}
```

### AlphaZero (`config.json`)
```json
{
  "training": {
    "mcts_iters": 100,   // Lower since network guides search
    "c_puct": 1.4,
    "gamma": 0.99
  }
}
```

---

## GPU Acceleration

All MCTS implementations support GPU acceleration through:

1. **Ray tracing**: CUDA kernel for camera observations (15x speedup)
2. **ROE propagation**: CUDA kernel for orbital dynamics (2x speedup)
3. **Entropy calculation**: GPU-based computations
4. **Grid operations**: Efficient cloning and belief updates on GPU

**To enable:**
```python
model = OrbitalMCTSModel(
    ...,
    use_torch=True,
    device='cuda'
)
```

**Requirements:**
- CUDA Toolkit matching PyTorch version
- Compiled CUDA kernels (see main README)
- GPU with compute capability ≥ 6.0

---

## References

- **Pure MCTS**: Coulom, R. (2006). "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search"
- **PUCT**: Rosin, C. (2011). "Multi-armed bandits with episode context"
- **AlphaZero**: Silver, D. et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- **Root parallelization**: Chaslot, G. et al. (2008). "Parallel Monte-Carlo Tree Search"
