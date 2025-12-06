# RL-Based Active Information Gathering for Non-Cooperative RSOs

This project implements reinforcement learning approaches for active information gathering using spacecraft orbital maneuvers. It compares two methods: Pure Monte Carlo Tree Search (MCTS) and AlphaZero-style learning with neural networks.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Output Structure](#output-structure)
- [Project Architecture](#project-architecture)
- [Key Parameters](#key-parameters)

## Overview

The system simulates a chaser spacecraft attempting to characterize a non-cooperative target spacecraft by:
1. Planning orbital maneuvers to optimize observation positions
2. Simulating camera observations from different viewpoints
3. Updating a 3D belief grid representing the target's shape
4. Maximizing information gain (entropy reduction) while minimizing fuel cost

**Two approaches implemented:**
- **Pure MCTS**: Classical tree search with random rollouts, no learning
- **AlphaZero**: Neural network-guided MCTS with self-play training

## Installation

### Dependencies
```bash
pip install numpy torch matplotlib pandas imageio graphviz
```

### Verify Installation
```bash
python -c "import torch; print(torch.__version__)"
```

## Quick Start

### 1. Run Pure MCTS (No Training Required)

```bash
python main.py
```

This runs a single episode using pure MCTS planning:
- Uses UCB1-based tree search with random rollouts
- No neural networks involved
- Results saved to `output/<timestamp>/`
- Generates visualization video and entropy plots

**What to expect:**
- Runtime: ~5-30 minutes depending on `mcts_iters` in `MCTSController`
- Output: Video showing spacecraft trajectory and belief evolution

### 2. Run AlphaZero Training

```bash
python run_alphazero.py
```

This trains a neural network via self-play:
- Runs multiple episodes in parallel
- Each episode uses MCTS guided by neural network predictions
- Network learns from collected trajectories
- Results saved to `output_training/run_<timestamp>/`

**What to expect:**
- Runtime: Hours to days depending on `num_episodes` in config
- Output: Training logs, checkpoints, per-episode videos, loss curves

### 3. Resume Interrupted Training

```bash
python resume_training.py --run_dir output_training/run_2025-12-04_11-08-29
```

Optional: specify additional episodes
```bash
python resume_training.py --run_dir output_training/run_2025-12-04_11-08-29 --additional_episodes 20
```

### 4. Run Baseline (No Maneuvers)

```bash
python run_baseline_no_maneuver.py
```

Spacecraft observes from fixed relative position without maneuvering (for comparison).

## Configuration

All parameters are controlled via `config.json`:

### Key Configuration Sections

#### Simulation Parameters
```json
"simulation": {
  "max_horizon": 5,              // MCTS planning depth
  "num_steps": 50,               // Steps per episode
  "time_step": 120.0,            // Orbital propagation timestep (seconds)
  "verbose": false,
  "visualize": true,
  "output_dir": "output_training"
}
```

#### Orbit Settings
```json
"orbit": {
  "a_chief_km": 7000.0,          // Target orbit semi-major axis (km)
  "e_chief": 0.001,              // Eccentricity
  "i_chief_deg": 98.0,           // Inclination (degrees)
  "omega_chief_deg": 30.0        // RAAN (degrees)
}
```

#### Camera Model
```json
"camera": {
  "fov_degrees": 10.0,           // Field of view
  "sensor_res": [64, 64],        // Resolution (pixels)
  "noise_params": {
    "p_hit_given_occupied": 0.95,  // True positive rate
    "p_hit_given_empty": 0.001     // False positive rate
  }
}
```

#### Control Parameters
```json
"control": {
  "lambda_dv": 1                 // Fuel cost weight (higher = more conservative)
}
```

#### Initial Conditions
```json
"initial_roe_meters": {          // Relative orbital elements (meters)
  "da": 0.0,                     // Semi-major axis difference
  "dl": 200.0,                   // Mean longitude difference
  "dex": 100.0, "dey": 0.0,      // Eccentricity vector differences
  "dix": 50.0, "diy": 0.0        // Inclination vector differences
}
```

#### Monte Carlo Settings
```json
"monte_carlo": {
  "num_episodes": 65,            // Number of training episodes
  "perturbation_bounds": {       // Random initial state variation
    "da": 0.0,
    "dl": 20.0,
    "dex": 10.0,
    "dey": 10.0,
    "dix": 5.0,
    "diy": 5.0
  }
}
```

#### Neural Network
```json
"network": {
  "hidden_dim": 128              // Hidden layer size
}
```

#### Training Hyperparameters
```json
"training": {
  "batch_size": 64,              // Mini-batch size for SGD
  "learning_rate": 0.0005,       // Adam learning rate
  "mcts_iters": 100,             // MCTS simulations per action
  "epochs_per_cycle": 5,         // Training epochs per batch of episodes
  "buffer_size": 20000,          // Replay buffer capacity
  "c_puct": 1.4,                 // PUCT exploration constant
  "gamma": 0.99                  // Discount factor
}
```

### Choosing Configs for Different Modes

**For Pure MCTS** (`main.py`):
- Adjust `simulation.max_horizon` (default: 5)
- In `mcts/mcts_controller.py`, the `MCTSController` init sets `mcts_iters` (default: 3000)
- Higher `mcts_iters` = better planning but slower
- Recommended: 1000-5000 iterations

**For AlphaZero** (`run_alphazero.py`):
- `training.mcts_iters`: 100-500 (faster since network guides search)
- `monte_carlo.num_episodes`: 50-200 for meaningful training
- `training.batch_size`: 32-128 depending on memory
- `training.c_puct`: 1.0-2.0 (controls exploration)
- `control.lambda_dv`: Tune to balance info gain vs fuel cost

## Running Experiments

### Experiment 1: Quick Pure MCTS Test
Edit `mcts/mcts_controller.py` line 19 to reduce iterations:
```python
self.mcts_iters = 500  # Faster for testing
```
Then:
```bash
python main.py
```

### Experiment 2: Full AlphaZero Training Run
Recommended config changes:
```json
{
  "monte_carlo": { "num_episodes": 100 },
  "training": {
    "mcts_iters": 200,
    "batch_size": 64,
    "learning_rate": 0.0003
  }
}
```
```bash
python run_alphazero.py
```

### Experiment 3: Hyperparameter Tuning

**Exploration vs Exploitation:**
- Increase `c_puct` (e.g., 2.0) for more exploration
- Decrease `c_puct` (e.g., 1.0) for more exploitation

**Fuel Efficiency:**
- Increase `lambda_dv` (e.g., 5.0) to penalize maneuvers more
- Decrease `lambda_dv` (e.g., 0.1) to prioritize info gain

**Training Speed:**
- Reduce `mcts_iters` for faster episodes (but less accurate MCTS policy)
- Increase `epochs_per_cycle` for more thorough network updates

## Output Structure

### Pure MCTS Output (`output/<timestamp>/`)
```
output/2025-12-05_10-30-00/
├── final_visualization.mp4      # Animation of spacecraft trajectory + belief
├── final_frame.png              # Last frame
├── entropy_progression.png      # Entropy reduction over time
└── replay_buffer.csv            # State-action-reward data
```

### AlphaZero Output (`output_training/run_<timestamp>/`)
```
output_training/run_2025-12-05_10-30-00/
├── run_config.json              # Configuration used
├── training.log                 # Training progress log
├── loss_history.png             # Policy and value loss curves
├── checkpoints/
│   ├── checkpoint_ep_1.pt
│   ├── checkpoint_ep_10.pt
│   ├── ...
│   └── best.pt                  # Best performing checkpoint
└── episode_01/                  # Per-episode data
    ├── episode_data.csv         # Step-by-step log
    ├── entropy.png              # Entropy curve
    ├── video.mp4                # Visualization
    └── trees/                   # MCTS tree visualizations (optional)
        ├── step_000.dot
        └── ...
```

## Project Architecture

### Directory Structure
```
CS229_Final_Project/
├── main.py                      # Entry point: Pure MCTS
├── run_alphazero.py             # Entry point: AlphaZero training
├── resume_training.py           # Resume training from checkpoint
├── run_baseline_no_maneuver.py  # Baseline comparison
├── config.json                  # Main configuration file
│
├── mcts/                        # MCTS implementations
│   ├── mcts.py                  # Pure MCTS (UCB1)
│   ├── mcts_alphazero_controller.py  # AlphaZero MCTS (PUCT)
│   ├── mcts_controller.py       # Controller wrapper
│   └── orbital_mdp_model.py     # MDP formulation
│
├── learning/                    # Neural network training
│   ├── training_loop.py         # AlphaZero self-play loop
│   ├── training.py              # Network training (SGD updates)
│   └── policy_value_network.py  # Neural network architecture
│
├── simulation/
│   └── scenario_full_mcts.py    # Pure MCTS simulation runner
│
├── roe/                         # Orbital mechanics
│   ├── propagation.py           # ROE propagation
│   └── dynamics.py              # Impulsive maneuver dynamics
│
├── camera/
│   └── camera_observations.py   # Camera model + voxel grid
│
└── output/                      # Results directories
    ├── output/                  # Pure MCTS results
    ├── output_training/         # AlphaZero results
    └── output_baseline/         # Baseline results
```

### State Representation
- **ROE (Relative Orbital Elements)**: 6D vector [Δa, Δλ, Δex, Δey, Δix, Δiy]
- **Belief Grid**: 20×20×20 voxel grid of occupancy probabilities (0-1)

### Action Space
13 discrete impulsive maneuvers in RTN (Radial-Tangential-Normal) frame:
- No-op: [0, 0, 0]
- Small/large radial: ±[0.01, 0, 0], ±[0.05, 0, 0] m/s
- Small/large tangential: ±[0, 0.01, 0], ±[0, 0.05, 0] m/s
- Small/large normal: ±[0, 0, 0.01], ±[0, 0, 0.05] m/s

### Reward Function
```
r_t = (H_before - H_after) - lambda_dv * |ΔV|
```
Where:
- H: Shannon entropy of belief grid (higher = more uncertainty)
- ΔV: Delta-v magnitude (fuel cost)
- lambda_dv: Cost coefficient (tunable in config)

### Neural Network Architecture

**Input:**
- ROE state (6D, scaled by 10000 for numerical stability)
- Voxel grid (20×20×20)

**Architecture:**
```
Grid Path:
  Conv3d(1→16) → Conv3d(16→32, stride=2) → Conv3d(32→64, stride=2)
  → Flatten → FC(8000→128)

ROE Path:
  FC(6→128)

Combined:
  Concat(256) → FC(256→128) → {Policy Head, Value Head}

Outputs:
  - Policy: FC(128→13) → Softmax [action probabilities]
  - Value: FC(128→1) → Tanh [state value estimate]
```

**Loss:**
```
L_total = L_policy + L_value
L_policy = -sum(π_MCTS * log(π_network))
L_value = (z - V(s))^2
```

## Key Parameters

### Tuning Guide

| Parameter | Location | Effect | Recommended Range |
|-----------|----------|--------|-------------------|
| `mcts_iters` | `training.mcts_iters` (AlphaZero) or `MCTSController` init (Pure) | MCTS search depth | 100-500 (AlphaZero), 1000-5000 (Pure) |
| `c_puct` | `training.c_puct` | Exploration in MCTS | 0.7-2.0 |
| `lambda_dv` | `control.lambda_dv` | Fuel cost penalty | 0.01-10 |
| `learning_rate` | `training.learning_rate` | Network training speed | 1e-4 to 1e-3 |
| `batch_size` | `training.batch_size` | SGD batch size | 32-256 |
| `num_episodes` | `monte_carlo.num_episodes` | Training episodes | 50-200 |
| `max_horizon` | `simulation.max_horizon` | MCTS tree depth | 3-10 |

### Performance vs Compute Tradeoffs

**Fast testing** (minutes):
- Pure MCTS with `mcts_iters=500`
- AlphaZero with `num_episodes=10`, `mcts_iters=50`

**Moderate quality** (hours):
- Pure MCTS with `mcts_iters=2000`
- AlphaZero with `num_episodes=50`, `mcts_iters=200`

**High quality** (days):
- Pure MCTS with `mcts_iters=5000`
- AlphaZero with `num_episodes=200`, `mcts_iters=500`

## Algorithm Comparison

| Aspect | Pure MCTS | AlphaZero |
|--------|-----------|-----------|
| **Planning** | UCB1 tree search | PUCT with network priors |
| **Evaluation** | Random rollouts | Neural network value |
| **Training** | None (ready to use) | Self-play + SGD |
| **Speed per step** | Slow (~50-500 rollouts) | Fast (network inference) |
| **Deployment** | Requires MCTS at runtime | Network inference only |
| **Sample efficiency** | N/A | Improves with training |
| **Best for** | Small-scale, interpretable | Large-scale, learned policies |

## Troubleshooting

### MCTS too slow
- Reduce `mcts_iters` in config or `MCTSController` init
- Reduce `max_horizon` to limit tree depth

### Training not improving
- Check `training.log` for loss curves
- Increase `learning_rate` if loss plateaus
- Increase `mcts_iters` for better training signal
- Ensure `buffer_size` is large enough

### Out of memory
- Reduce `batch_size`
- Reduce voxel grid resolution (hardcoded in `camera/camera_observations.py`)
- Use CPU instead of GPU (code auto-detects)

### Checkpoints not loading
- Verify checkpoint path exists
- Check `run_config.json` matches current config structure

## Citation

If you use this code, please cite:
```
CS229 Final Project: RL-Based Active Information Gathering for Non-Cooperative RSOs
Stanford University, 2025
```

## License

Academic use only. See project documentation for details.
