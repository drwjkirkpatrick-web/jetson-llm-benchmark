# JetPack R36.4.7 → R36.5.2 Upgrade Guide

## The Bug

JetPack R36.4.7 contains a **critical regression** in the CUDA contiguous memory allocator (`NvMapMemAllocInternalTagged`) that prevents models larger than ~1.1 GB from loading into GPU memory.

### Symptoms

- 3B models (qwen2.5:3b, llama3.2:3b) load fine
- 7B models fail with `cudaMalloc failed: out of memory`
- Free RAM shows 4+ GB available, but model won't load
- `dmesg` shows: `NvMapMemAllocInternalTagged: 1075072515 error 12 (ENOMEM)`
- `/proc/meminfo` shows `CmaFree` < 50 MB despite general RAM being plentiful

### Why It Happens

Jetson uses Unified Memory Architecture (UMA) — CPU and GPU share the same physical RAM. The GPU requires **contiguous** chunks of memory allocated via `nvmap`. R36.4.7 has a bug where the IOVA (I/O Virtual Address) mapping fails for allocations > ~1.1 GB, returning ENOMEM even when sufficient total RAM exists.

This is **not** a true out-of-memory condition. It's a kernel-level contiguous allocator bug.

## The Fix

Upgrade to **JetPack R36.5.2** (or later). This includes a kernel patch that fixes the IOVA/NvMap contiguous allocation regression.

### Step 1: Update APT Sources

```bash
# Check current JetPack
cat /etc/nv_tegra_release
# Example: R36 (release), REVISION: 4.7

# Update sources to r36.5
sudo sed -i 's/r36.3/r36.5/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
sudo sed -i 's/r36.4/r36.5/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
```

### Step 2: Upgrade Packages

```bash
sudo apt update
sudo apt upgrade -y
sudo apt dist-upgrade -y
```

### Step 3: Reboot

```bash
sudo reboot
```

### Step 4: Verify

```bash
cat /etc/nv_tegra_release
# Should show: R36 (release), REVISION: 5.2

# Test with a 7B model
grep CmaFree /proc/meminfo
# Should show > 100 MB after GUI stopped
```

## Alternative: If You Cannot Upgrade

If upgrading JetPack is not possible, these workarounds may help:

### 1. Force CPU-Only Inference

```bash
# In Ollama API options
"options": {"num_gpu": 0, "num_ctx": 512}
```

This bypasses CUDA entirely. A 3B model runs in 3–9 seconds per prompt on CPU. A 7B model may be too slow for practical use.

### 2. Use Smaller Models

Stick to 3B models (qwen2.5:3b, llama3.2:3b, phi3:3.8b) which fit within the ~1.1 GB contiguous allocation limit.

### 3. Headless Mode

Stop the display manager to free CMA memory:

```bash
sudo systemctl stop gdm3
```

This helps but does **not** fix the R36.4.7 bug — it only gives you more contiguous memory to work with.

## Verification After Upgrade

Run this test to confirm the fix:

```bash
# Stop GUI
sudo systemctl stop gdm3

# Check CMA
grep CmaFree /proc/meminfo
# Expected: > 100 MB

# Try loading a 7B model with llama.cpp
~/llama.cpp/build/bin/llama-cli \
  -m ~/.ollama/models/blobs/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -n 200 -c 1024 --temp 0.3 -ngl 99 --flash-attn on \
  --no-conversation --no-display-prompt \
  -p "Test prompt"

# Should succeed and show generation speed
```

## References

- NVIDIA Jetson Orin Nano documentation: https://developer.nvidia.com/embedded/jetson
- JetPack release notes: https://docs.nvidia.com/jetson/jetpack/release-notes/
- llama.cpp CUDA support: https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#cuda
