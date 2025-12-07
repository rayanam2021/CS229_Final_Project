# CUDA Ray-Tracing Kernel

Ultra-fast GPU-accelerated ray tracing for voxel-based spacecraft observation simulation.

**Performance:** ~46ms per observation (64×64 rays) - **15.6x faster than CPU!**

---

## Quick Start

**Already have PyTorch with CUDA 12.8?** Just compile:

```bash
cd camera/cuda
python setup.py install
```

**Don't have CUDA?** The system automatically falls back to CPU/PyTorch. See [Installation](#installation) below.

---

## Prerequisites

Check your current PyTorch CUDA version:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

**Example output:** `PyTorch: 2.8.0+cu128, CUDA: 12.8`

The CUDA Toolkit you install **must match** the CUDA version shown above (e.g., 12.8).

**Required:**
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- CUDA Toolkit (matching PyTorch's version)
- NVIDIA GPU with compute capability ≥ 7.5
- C++ compiler (g++ on Linux, MSVC on Windows)

---

## Installation

### Linux Installation

#### 1. Check NVIDIA Driver

```bash
nvidia-smi
```

**Required:** Driver ≥ 535 for CUDA 12.8

If missing or outdated:
```bash
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

#### 2. Install CUDA Toolkit 12.8

**Check which version you need:**
```bash
python -c "import torch; print(torch.version.cuda)"
```

**Install matching version (example for 12.8):**

```bash
# Download installer
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2004-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

**Add to `~/.bashrc`:**
```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
```

**Apply:**
```bash
source ~/.bashrc
```

**Verify:**
```bash
nvcc --version  # Should show CUDA 12.8
```

#### 3. Compile the Kernel

```bash
cd camera/cuda
CUDA_HOME=/usr/local/cuda-12.8 python setup.py install
```

---

### Windows Installation

#### 1. Install NVIDIA Driver

Download from [NVIDIA Downloads](https://www.nvidia.com/Download/index.aspx)

**Required:** Version ≥ 537 for CUDA 12.8

#### 2. Install Visual Studio Build Tools

Required for C++ compilation:

1. Download [Visual Studio Build Tools 2019+](https://visualstudio.microsoft.com/downloads/)
2. Install with "Desktop development with C++" workload
3. Restart

#### 3. Install CUDA Toolkit

1. Download [CUDA 12.8 for Windows](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows)
2. Run installer (default settings)
3. Installation path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`

#### 4. Set Environment Variables

**System Environment Variables:**
```
CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
```

**Add to PATH:**
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
```

**Verify in Command Prompt:**
```cmd
nvcc --version
```

#### 5. Compile the Kernel

Open **x64 Native Tools Command Prompt for VS 2019**:

```cmd
cd camera\cuda
python setup.py install
```

---

## Verification

Test if the CUDA kernel is available:

```python
from camera.cuda.cuda_wrapper import CUDA_AVAILABLE
print(f"CUDA kernel: {CUDA_AVAILABLE}")
```

**Quick test:**
```bash
python -c "
import torch
from camera.cuda.cuda_wrapper import trace_rays_cuda

origins = torch.randn(1024, 3, device='cuda') * 5.0
dirs = torch.randn(1024, 3, device='cuda')
dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)

grid = torch.zeros((20, 20, 20), dtype=torch.bool, device='cuda')
grid[5:15, 5:15, 5:15] = True

hits, misses = trace_rays_cuda(
    origins, dirs, grid,
    torch.tensor([-10., -10., -10.], device='cuda'),
    1.0, 0.9, 0.1, return_tensors=True
)
print(f'Success! {hits.shape[0]} hits, {misses.shape[0]} misses')
"
```

---

## Usage

### Automatic (Recommended)

The system automatically uses the fastest available method:

```python
from camera.camera_observations import VoxelGrid, simulate_observation

# CUDA kernel used automatically if available
grid = VoxelGrid(dims=(20, 20, 20), use_torch=True, device='cuda')
```

**Fallback chain:**
1. CUDA kernel (fastest, ~46ms)
2. PyTorch GPU (~53ms)
3. CPU sequential (~713ms)

### Manual Control

```python
# Use CUDA kernel (fastest)
grid = VoxelGrid(dims, use_torch=True, device='cuda')

# Use PyTorch only (no CUDA kernel)
grid = VoxelGrid(dims, use_torch=True, device='cpu')

# Use CPU only (most compatible)
grid = VoxelGrid(dims, use_torch=False, device='cpu')
```

---

## Troubleshooting

### "RuntimeError: CUDA version mismatch"

**Problem:** CUDA Toolkit doesn't match PyTorch's CUDA version

**Solution:**
```bash
# Check versions
python -c "import torch; print(torch.version.cuda)"  # PyTorch's CUDA
nvcc --version  # System CUDA

# They MUST match! Install correct CUDA Toolkit.
```

### "nvcc: command not found" (Linux)

**Problem:** CUDA not in PATH

**Solution:**
```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
source ~/.bashrc
```

### "CUDA_HOME not set"

**Solution:**
```bash
# Linux
export CUDA_HOME=/usr/local/cuda-12.8

# Windows (in System Environment Variables)
CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
```

### Compilation fails with "g++ not found"

**Solution (Linux):**
```bash
sudo apt install build-essential
```

---

## Performance

### Benchmark (RTX 2060, 64×64 rays, 20³ grid)

| Method | Time/Obs | Speedup |
|--------|----------|---------|
| CPU Sequential | 713ms | 1x |
| PyTorch GPU | 53ms | 13.3x |
| **CUDA Kernel** | **46ms** | **15.6x** |

### Research Impact

**Pure MCTS (750k obs/episode):**
- CPU: 148 hours (~6 days) ❌
- PyTorch GPU: 11 hours ⚠️
- CUDA: **9.6 hours** ✅

**AlphaZero (25k obs/episode):**
- CPU: 4.9 hours ❌
- PyTorch GPU: 22 minutes ⚠️
- CUDA: **19 minutes** ✅

---

## Technical Details

- **Algorithm:** DDA (Digital Differential Analyzer) ray marching
- **Parallelization:** One CUDA thread per ray
- **RNG:** cuRAND (GPU-native random numbers)
- **Memory:** Zero CPU-GPU transfers during tracing
- **Optimizations:** Fast math, vectorized extraction, preallocated buffers

**Files:**
- `ray_tracing_kernel.cu` - CUDA kernel (243 lines)
- `cuda_wrapper.py` - Python interface
- `setup.py` - Build configuration
- `test_cuda_ray_tracing.py` - Test suite and benchmarks

---

## Testing

Run the comprehensive test suite:

```bash
python camera/cuda/test_cuda_ray_tracing.py
```

This will verify correctness and benchmark performance.

---

## Options for Improved Performance

Current performance is excellent (~0.6ms), but here are additional optimization strategies:

### 1. Persistent Grid Storage

**Current:** Grid may be uploaded to GPU multiple times
**Solution:** Cache grid on GPU across observations

```python
# Keep grid on GPU permanently
grid_gpu = VoxelGrid(dims=(20, 20, 20), use_torch=True, device='cuda')
# Reuse many times without re-uploading
```

### 2. Batched Observations

**Solution:** Process multiple camera viewpoints concurrently

```python
# Process multiple MCTS nodes simultaneously
for viewpoint in viewpoints:
    with torch.cuda.stream(cuda_stream):
        simulate_observation(...)
```

**Expected gain:** 2-4x throughput for batched evaluations

### 3. Shared Memory Caching

**Solution:** Cache frequently-accessed voxels in shared memory for dense grids

**Expected gain:** 1.5-2x for very dense occupancy grids

### 4. Early Ray Termination

**Solution:** Stop on first hit if only detection matters

**Expected gain:** 2-5x for detection-only scenarios in sparse grids

### 5. Hierarchical Acceleration (Octree)

**Solution:** Skip empty regions using octree structure

**Expected gain:** 10-100x for very large sparse grids (100x100x100+)

---

## Support

**Problems?** Check:
1. CUDA version matches PyTorch: `python -c "import torch; print(torch.version.cuda)"` vs `nvcc --version`
2. NVIDIA driver is recent: `nvidia-smi`
3. C++ compiler installed: `g++ --version` (Linux) or check VS Build Tools (Windows)

**Still stuck?** File an issue with:
- Output of `nvidia-smi`
- Output of `nvcc --version`
- Output of `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
- Full error message
