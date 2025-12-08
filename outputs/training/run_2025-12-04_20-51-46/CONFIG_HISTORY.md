# Training Configuration History

## Episodes 1-54: Original Configuration

**File:** `run_config.json` (original, saved at run start)

```json
{
    "training": {
        "batch_size": 64,
        "learning_rate": 0.001,
        "mcts_iters": 100,
        "epochs_per_cycle": 5,
        "buffer_size": 20000,
        "c_puct": 1.4,
        "gamma": 0.99,
        "device": "cuda"
    },
    "simulation": {
        "max_horizon": 5,
        "num_steps": 50,
        "time_step": 120.0
    }
}
```

**Performance:**
- Episode 54 final loss: 2.4400
- Training epochs: 35
- Fast convergence but value loss spikes

---

## Episodes 55+ (Current): Depth-Focused Configuration

**File:** `run_config.json` (currently in directory)

```json
{
    "training": {
        "batch_size": 64,
        "learning_rate": 0.0015,   // ← CHANGED: +50% more aggressive
        "mcts_iters": 80,           // ← Balanced for speed vs quality
        "epochs_per_cycle": 5,
        "buffer_size": 20000,
        "c_puct": 1.4,
        "gamma": 0.99,
        "device": "cuda"
    },
    "simulation": {
        "max_horizon": 8,           // ← KEY CHANGE: +60% deeper planning
        "num_steps": 50,
        "time_step": 120.0
    }
}
```

**Rationale:**
- Entropy analysis showed 74-76% myopic planning in episodes 50-54
- Depth 8 allows seeing 3 more steps ahead (60% deeper than baseline)
- MCTS iterations at 80 (20% reduction from baseline) for balanced speed and quality
- Tree size increases from 781 nodes (depth 5) to ~390,625 nodes (depth 8)

**Configuration evolution:**
1. Initial consideration: LR=0.0015, MCTS=200, depth=5 (too slow, doesn't fix myopic planning)
2. Depth-focused: LR=0.0015, MCTS=100, depth=8 (good, but can optimize iterations)
3. Deeper attempt: LR=0.0015, MCTS=100, depth=10 (too slow with ~9.8M nodes)
4. Speed optimization: LR=0.0015, MCTS=75, depth=10 (still too slow)
5. **Final balanced config: LR=0.0015, MCTS=80, depth=8** (best trade-off)

---

## Configuration Comparison

| Config | Episodes | LR | MCTS Iters | Depth | Episode Time | Addresses |
|--------|----------|----|-----------| ------|-------------|-----------|
| **Original** | 1-54 | 0.001 | 100 | 5 | 1x | Baseline |
| **Current** | 55+ | 0.0015 | 80 | 8 | ~2x | **Myopic planning + aggressive exploration** |

---

## Evidence for Depth Over Iterations

### Entropy Analysis Results:

```
Episode 10: 2.4% early reduction, 22% plateaus  → Good planning
Episode 30: 1.2% early reduction, 31% plateaus  → Getting myopic
Episode 50: 74% early reduction, 8% plateaus   → VERY MYOPIC!
Episode 54: 76% early reduction, 39% plateaus  → VERY MYOPIC!
```

**Diagnosis:** Policy has learned to grab immediate gains, but can't plan long-term.

**Solution:** Increase depth to see further ahead, not just explore current horizon more thoroughly.

---

## Applied Configuration (COMPLETED)

**Configuration has been updated to:**
```json
"learning_rate": 0.0015,  // Aggressive exploration (+50% vs baseline)
"mcts_iters": 80,          // Balanced for speed with deeper search
"max_horizon": 8           // Deep planning (+60% deeper than baseline)
```

**Expected outcomes:**
- ~2x slower per episode (depth 8 with 80 iterations)
- Eliminates myopic planning (front-loaded entropy reduction should decrease)
- Significantly reduces plateau steps
- More sustained entropy reduction throughout episodes
- Better final loss (~2.25-2.30 vs current 2.44)
- Tree size: ~390,625 nodes (vs 781 at depth 5, 500x larger)
- Good balance of speed and quality

**Ready to resume training from checkpoint 54 with optimized settings.**

---

## Summary of Changes from Episodes 1-54 to Episodes 55+

| Parameter | Episodes 1-54 | Episodes 55+ | Change | Reason |
|-----------|---------------|--------------|--------|--------|
| **Learning Rate** | 0.001 | 0.0015 | +50% | Aggressive exploration, faster learning |
| **MCTS Iterations** | 100 | 80 | -20% | Balance speed with deeper search |
| **Max Horizon (Depth)** | 5 | 8 | +60% | **Fix myopic planning** (74-76% early entropy reduction) |
| **Tree Size** | ~781 nodes | ~390,625 nodes | 500x | Enables long-term strategic planning |
| **Episode Time** | 1x | ~2x | 2x slower | Acceptable trade-off for eliminating myopic planning |

**Key insight:** Depth was the bottleneck, not iterations. Increasing horizon from 5 to 8 directly addresses the myopic planning problem revealed by entropy analysis.
