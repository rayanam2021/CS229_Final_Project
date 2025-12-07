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
  - **Now uses root parallelization** for 5-8x speedup
  - Runs multiple independent MCTS searches in parallel and aggregates results
- **AlphaZero**: Neural network-guided MCTS with self-play training
  - **Highly optimized** with GPU training, mixed precision, and parallel episodes
  - Achieves 100-200x speedup over baseline CPU implementation

## Installation

### Dependencies
```bash
pip install numpy torch matplotlib pandas imageio graphviz
```

### Verify Installation
```bash
python -c "import torch; print(torch.__version__)"
```

### GPU Acceleration (Optional but Recommended)

This project includes **CUDA kernels for both camera ray tracing and orbital propagation**.

**Performance with CUDA (RTX 2060):**
- **Camera ray tracing**: 15.6x faster than CPU (713ms → 46ms per observation)
- **ROE propagation**: 1.87x faster than CPU (perfect float64 accuracy)
- **Combined impact**: Significant speedup for MCTS tree search

**Without CUDA:** Automatic fallback to CPU/PyTorch implementations.

#### Quick Setup (if you already have PyTorch with CUDA):

**1. Compile Camera CUDA:**
```bash
cd camera/cuda
python setup.py install
```

**2. Compile ROE CUDA:**
```bash
cd roe/cuda
nvcc -arch=sm_75 -shared -O3 -Xcompiler -fPIC -o libroe_propagation.so roe_propagation_kernel.cu
```
*Adjust `-arch=sm_75` for your GPU: sm_60 (Pascal), sm_70 (Volta), sm_75 (Turing), sm_80/86 (Ampere), sm_89 (Ada)*

**3. Verify:**
```bash
python -c "from camera.cuda.cuda_wrapper import CUDA_AVAILABLE as CAM; from roe.dynamics import CUDA_ROE_AVAILABLE as ROE; print(f'Camera CUDA: {CAM}, ROE CUDA: {ROE}')"
```

**4. Run tests:**
```bash
python camera/cuda/test_cuda_ray_tracing.py
python roe/cuda/test_cuda_roe_propagation.py
```

#### First-Time CUDA Installation:

If you don't have CUDA Toolkit installed or get a version mismatch error:

1. **Check your PyTorch CUDA version:**
   ```bash
   python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
   ```

2. **Install matching CUDA Toolkit** (example for CUDA 12.8 on Linux):
   ```bash
   # Download and install CUDA Toolkit 12.8
   wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2004-12-8-local_12.8.1-570.124.06-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-12-8-local_12.8.1-570.124.06-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2004-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt update
   sudo apt install -y cuda-toolkit-12-8

   # Add to ~/.bashrc
   export PATH=/usr/local/cuda-12.8/bin:$PATH
   export CUDA_HOME=/usr/local/cuda-12.8
   source ~/.bashrc
   ```

3. **Compile the kernel:**
   ```bash
   cd camera/cuda
   CUDA_HOME=/usr/local/cuda-12.8 python setup.py install
   ```

**For detailed installation instructions (Windows/Linux), troubleshooting, and performance benchmarks:**

📖 **See [camera/cuda/README.md](camera/cuda/README.md)**
📖 **See [docs/Optimization_Summary.md](docs/Optimization_Summary.md)** for complete optimization details and future opportunities

**Don't have GPU?** No problem! The system automatically falls back to CPU - everything still works, just slower.

## Quick Start

### 1. Run Pure MCTS (No Training Required)

```bash
python run_pure_mcts.py
```

This runs a single episode using **parallelized** pure MCTS planning:
- Uses UCB1-based tree search with random rollouts
- **Root parallelization**: Runs multiple independent MCTS trees in parallel for 5-8x speedup
- **GPU-accelerated**: Ray tracing and ROE propagation on GPU when available
- No neural networks involved
- Results saved to `outputs/mcts/<timestamp>/`
- Generates visualization video and entropy plots

**What to expect:**
- Runtime: ~5-10 minutes with 3000 MCTS iterations (11 parallel workers)
- Output: Video showing spacecraft trajectory and belief evolution
- Checkpoints saved every 10 steps for resuming

**Resume from checkpoint:**
```bash
python resume_pure_mcts.py --checkpoint outputs/mcts/2025-12-07_XX-XX-XX/checkpoint_step_10.pkl
```

### 2. Run AlphaZero Training

```bash
python run_alphazero.py
```

This trains a neural network via self-play with **heavy optimizations**:
- **Parallel episode generation**: 11 workers run episodes simultaneously (9x speedup)
- **GPU training**: Network training on GPU with mixed precision (AMP) for 20-60x speedup
- **GPU ray tracing**: Camera observations and ROE propagation accelerated
- **Dynamic batch sizing**: 128 on GPU vs 64 on CPU
- **torch.compile**: Additional 1.5-2x inference speedup
- Each episode uses MCTS guided by neural network predictions
- Network learns from collected trajectories via replay buffer
- Results saved to `outputs/training/run_<timestamp>/`

**Performance:**
- **100-200x faster** than baseline CPU serial implementation
- GPU utilization: 80-95%
- 65 episodes complete in ~12-14 minutes (vs 2+ hours serial)

**What to expect:**
- Runtime: 15-30 minutes for 65 episodes (hours on CPU-only)
- Output: Training logs, checkpoints, per-episode videos, loss curves

### 3. Resume Interrupted Training

```bash
python resume_training.py --run_dir outputs/training/run_2025-12-04_11-08-29
```

Optional: specify additional episodes
```bash
python resume_training.py --run_dir outputs/training/run_2025-12-04_11-08-29 --additional_episodes 20
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
  "output_dir": "outputs/training"
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

**For Pure MCTS** (`run_pure_mcts.py`):
- Edit `config_pure_mcts.json`:
  - `simulation.max_horizon` (default: 5)
  - `mcts.mcts_iters` (default: 3000)
  - `mcts.num_workers` (optional, default: auto = CPU count - 1)
- Higher `mcts_iters` = better planning but slower
- Recommended: 1000-5000 iterations
- **Uses parallel MCTS** with root parallelization for 5-8x speedup

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

### Pure MCTS Output (`outputs/training/<timestamp>/`)
```
outputs/training/2025-12-05_10-30-00/
├── final_visualization.mp4      # Animation of spacecraft trajectory + belief
├── final_frame.png              # Last frame
├── entropy_progression.png      # Entropy reduction over time
└── replay_buffer.csv            # State-action-reward data
```

### AlphaZero Output (`outputs/training/run_<timestamp>/`)
```
outputs/training/run_2025-12-05_10-30-00/
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
│   ├── mcts.py                  # Pure MCTS (UCB1) - serial version
│   ├── mcts_parallel.py         # Pure MCTS with root parallelization (5-8x speedup)
│   ├── mcts_controller.py       # Serial MCTS controller wrapper
│   ├── mcts_controller_parallel.py  # Parallel MCTS controller (USED)
│   ├── mcts_alphazero_controller.py  # AlphaZero MCTS (PUCT)
│   └── orbital_mdp_model.py     # MDP formulation (GPU-accelerated)
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
├── utility_scripts/             # Utility scripts
│   ├── plot_losses.py           # Plot training losses
│   ├── plot_losses_separate.py  # Separate loss plots
│   ├── inspect_checkpoint.py    # Checkpoint inspection
│   ├── test_trained_agent.py    # Test trained models
│   ├── test_propagation_maneuver.py
│   └── generate_network_diagram.py
│
├── docs/                        # Documentation
│   ├── CS_229_Project_Active_Information_Final_Report_Full_Length.pdf
│   └── Project approach with AlphaZero.md
│
├── media/                       # Images and animations
│   ├── network_architecture.png
│   └── rso_characterization_circle.gif
│
├── tests/                       # Test and experiment scripts
│   ├── analyze_output_files.py
│   ├── sweep_experiments.py
│   └── ...
│
└── outputs/                     # All output results
    ├── mcts/                    # Pure MCTS results
    ├── training/                # AlphaZero training runs
    └── baseline/                # Baseline comparison results
```