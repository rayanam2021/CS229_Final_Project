# Pure MCTS Checkpoint & Resume Guide

## Overview

The Pure MCTS simulation now supports automatic checkpointing and resuming. This allows you to:
- **Recover from crashes** - Resume exactly where you left off
- **Save progress incrementally** - Checkpoints saved every 10 steps by default
- **Experiment safely** - Resume from any checkpoint to try different configurations

## How It Works

### Automatic Checkpointing

When you run Pure MCTS, checkpoints are automatically saved every 10 steps (configurable):

```bash
python run_pure_mcts.py
```

**Checkpoints are saved to:**
```
outputs/mcts/2025-12-07_12-30-00/
├── checkpoint_step_9.pkl     # After step 10
├── checkpoint_step_19.pkl    # After step 20
├── checkpoint_step_29.pkl    # After step 30
└── ...
```

### What's Saved in Each Checkpoint

Each checkpoint contains:
- **Current state**: ROE orbital elements
- **Current time**: Simulation time
- **Grid state**: Complete belief grid (occupancy probabilities)
- **Entropy history**: All entropy values up to this point
- **Camera trajectory**: All camera positions and view directions
- **Replay buffer**: All recorded transitions for analysis
- **Burn indices**: Maneuver timesteps

## How to Resume from a Checkpoint

### Option 1: Resume Script (Recommended)

```bash
python resume_pure_mcts.py --checkpoint outputs/mcts/2025-12-07_12-30-00/checkpoint_step_19.pkl
```

**Optional arguments:**
- `--config config_pure_mcts.json` - Use different config (default: config_pure_mcts.json)
- `--checkpoint_interval 5` - Save checkpoints every 5 steps instead of 10

**Example:**
```bash
# Resume from step 20 with checkpoints every 5 steps
python resume_pure_mcts.py \
    --checkpoint outputs/mcts/2025-12-07_12-30-00/checkpoint_step_19.pkl \
    --checkpoint_interval 5
```

### Option 2: Programmatic

```python
from simulation.scenario_full_mcts import run_orbital_camera_sim_full_mcts, load_checkpoint

# Load checkpoint
checkpoint, grid = load_checkpoint('path/to/checkpoint_step_19.pkl', use_gpu=True, device='cuda')

# Resume simulation
run_orbital_camera_sim_full_mcts(
    sim_config=config['simulation'],
    orbit_params=config['orbit'],
    camera_params=config['camera'],
    control_params=config['control'],
    initial_state_roe=initial_roe,  # Ignored when resuming
    out_folder=checkpoint_dir,
    resume_from='path/to/checkpoint_step_19.pkl',
    checkpoint_interval=10
)
```

## Common Use Cases

### Scenario 1: Crash Recovery

Your simulation crashes at step 32:

```bash
# Find latest checkpoint
ls -lt outputs/mcts/2025-12-07_12-30-00/checkpoint_*.pkl | head -1

# Resume from step 30
python resume_pure_mcts.py \
    --checkpoint outputs/mcts/2025-12-07_12-30-00/checkpoint_step_29.pkl
```

The simulation will continue from step 30 to completion.

### Scenario 2: Experiment with Different Parameters

You want to try different `lambda_dv` values after step 20:

1. **Edit `config_pure_mcts.json`** - Change `lambda_dv` to new value
2. **Resume from checkpoint:**
   ```bash
   python resume_pure_mcts.py \
       --checkpoint outputs/mcts/2025-12-07_12-30-00/checkpoint_step_19.pkl
   ```
3. **Compare results** - Old run vs new run from same starting point

### Scenario 3: More Frequent Checkpoints

For very long runs, you might want more frequent checkpoints:

**Modify `run_pure_mcts.py`:**
```python
run_orbital_camera_sim_full_mcts(
    ...
    checkpoint_interval=5  # Every 5 steps instead of 10
)
```

Or when resuming:
```bash
python resume_pure_mcts.py \
    --checkpoint path/to/checkpoint.pkl \
    --checkpoint_interval 5
```

## Performance Impact

**Checkpoint overhead:** ~0.1-0.5 seconds per checkpoint
- Saves belief grid (20×20×20 = 8,000 floats ≈ 32KB)
- Minimal impact on overall runtime

**Recommendation:** Keep default `checkpoint_interval=10` for 50-step runs.

## Troubleshooting

### "Checkpoint not found"
```bash
python resume_pure_mcts.py --checkpoint path/to/checkpoint.pkl
# ❌ Checkpoint not found: path/to/checkpoint.pkl
```
**Solution:** Check the path is correct. Use tab-completion or `ls` to verify.

### "CUDA out of memory"
If GPU memory is exhausted:
1. Checkpoints automatically save to CPU (numpy arrays)
2. When resuming, grid is transferred back to GPU
3. No manual intervention needed

### Resume adds duplicate steps
**This won't happen** - Checkpoints track the step number and resume from `step + 1`.

## Advanced: Manual Checkpoint Management

### Save Checkpoint Manually

```python
from simulation.scenario_full_mcts import save_checkpoint

save_checkpoint(
    out_folder='outputs/mcts/test',
    step=current_step,
    state=current_roe_state,
    time_sim=current_time,
    grid=voxel_grid,
    entropy_history=entropy_list,
    camera_positions=camera_pos_list,
    view_directions=view_dir_list,
    burn_indices=burn_idx_list,
    controller=mcts_controller
)
```

### Load Checkpoint Manually

```python
from simulation.scenario_full_mcts import load_checkpoint

checkpoint, grid = load_checkpoint(
    'outputs/mcts/test/checkpoint_step_10.pkl',
    use_gpu=True,
    device='cuda'
)

# Access checkpoint data
step = checkpoint['step']
state_roe = checkpoint['state_roe']
time = checkpoint['time_sim']
entropy_history = checkpoint['entropy_history']
replay_buffer = checkpoint['replay_buffer']
```

## File Size

Typical checkpoint sizes:
- **Grid (20×20×20):** ~32 KB (belief + log_odds)
- **Replay buffer (50 steps):** ~100-200 KB
- **Total:** ~200-300 KB per checkpoint

For a 50-step run with checkpoints every 10 steps:
- **5 checkpoints** × 250 KB ≈ **1.25 MB total**

Negligible storage overhead!

## Summary

✅ **Automatic checkpointing** - Every 10 steps by default
✅ **Easy resume** - Single command to continue
✅ **Crash recovery** - Never lose progress
✅ **Experiment-friendly** - Try different configs from same checkpoint
✅ **Low overhead** - <1% runtime impact, <2 MB storage

**Get started:**
```bash
# Run with checkpoints (automatic)
python run_pure_mcts.py

# Resume from checkpoint
python resume_pure_mcts.py --checkpoint outputs/mcts/<timestamp>/checkpoint_step_XX.pkl
```
