# Jetson LLM Benchmark

> Reproducible benchmarking toolkit for running 7B parameter LLMs on NVIDIA Jetson Orin Nano 8GB — with the fixes we discovered for JetPack memory bugs, CMA fragmentation, and inference optimization.

## TL;DR Results

| Backend | Quantization | Generation tok/s | Quality |
|---------|-----------|-----------------|---------|
| **llama.cpp** | Q4_K_M | **12.4** | Best |
| **llama.cpp** | Q2_K | **11.0** | Acceptable |
| **llama.cpp** | Q3_K_M | **9.2** | Good |
| Ollama | Q4_K_M (Super Mode) | **6.4** | Best |
| Ollama | Q3_K_M | **4.8** | Good |
| Ollama | Q2_K | **5.1** | Has typos |

**Key finding:** llama.cpp is **~2× faster** than Ollama for the same GGUF models on Jetson.

## Hardware

- **Device:** NVIDIA Jetson Orin Nano 8GB (Developer Kit)
- **GPU:** 1024-core Ampere @ 625 MHz base, 1020 MHz max (Super Mode)
- **RAM:** 8 GB unified memory (CPU+GPU share)
- **Storage:** 128 GB NVMe SSD
- **JetPack:** 6.1 (L4T R36.4.7 → **R36.5.2** after upgrade)
- **OS:** Ubuntu 22.04 ARM64

## The Problem We Solved

### JetPack R36.4.7 CUDA IOVA/NvMap Bug

JetPack R36.4.7 contains a **critical regression** in the CUDA contiguous memory allocator that causes `NvMapMemAllocInternalTagged: error 12 (ENOMEM)` for models > ~1.1 GB. This means:

- 3B models load fine (they're small enough)
- 7B models **fail to load on GPU**, even with plenty of free RAM
- The error is misleading — it looks like an OOM but is actually a **CMA fragmentation bug**

**Fix:** Upgrade to **JetPack R36.5.2** (or later). This is a kernel-level fix — no workaround exists in user space.

### CMA Fragmentation (Desktop Session)

Even after the JetPack fix, running the GNOME desktop leaves CMA memory fragmented. Loading a 7B model requires contiguous GPU-addressable memory that may not be available with Xorg + GNOME Shell running.

**Fix:** Stop the display manager before benchmarking:

```bash
sudo systemctl stop gdm3
```

This frees ~600–800 MB of CMA and allows 7B models to load reliably.

## Setup Instructions

### 1. Compile llama.cpp with CUDA

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build build --config Release -j$(nproc)
```

The binary will be at `build/bin/llama-cli`.

### 2. Download Models

Download GGUF models from Hugging Face. We recommend:

| Model | Size | Use Case |
|-------|------|----------|
| Qwen2.5-7B-Instruct-Q4_K_M.gguf | ~4.7 GB | Best quality, fastest |
| Qwen2.5-7B-Instruct-Q3_K_M.gguf | ~3.8 GB | Good quality, smaller |
| Qwen2.5-7B-Instruct-Q2_K.gguf | ~3.0 GB | Fastest, some quality loss |

Place them in `~/.ollama/models/blobs/` or any directory you prefer.

### 3. Install Ollama (Optional — for Comparison)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:7b-instruct-q3_k_m
```

### 4. Stop the GUI (Critical for 7B Models)

```bash
# Free CMA memory for GPU allocation
sudo systemctl stop gdm3

# Verify
free -h | head -2
grep CmaFree /proc/meminfo
# CmaFree should be > 100 MB
```

## Running Benchmarks

### Quick Start

```bash
python3 benchmark.py \
  --models \
    ~/.ollama/models/blobs/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    ~/.ollama/models/blobs/Qwen2.5-7B-Instruct-Q3_K_M.gguf \
    ~/.ollama/models/blobs/Qwen2.5-7B-Instruct-Q2_K.gguf \
  --prompt-file prompts/clinical-hypothyroid.txt \
  --output results/benchmark-$(date +%Y%m%d).json
```

### With Ollama Models

```bash
python3 benchmark.py \
  --ollama-models qwen2.5:7b-instruct,qwen2.5:7b-instruct-q3_k_m \
  --prompt-file prompts/clinical-hypothyroid.txt \
  --output results/ollama-benchmark.json
```

### Both Together

```bash
python3 benchmark.py \
  --models \
    ~/.ollama/models/blobs/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    ~/.ollama/models/blobs/Qwen2.5-7B-Instruct-Q3_K_M.gguf \
  --ollama-models qwen2.5:7b-instruct,qwen2.5:7b-instruct-q3_k_m \
  --prompt-file prompts/clinical-hypothyroid.txt \
  --output results/full-benchmark.json
```

## Results Analysis

```bash
python3 -m scripts.analyze results/full-benchmark.json
```

Outputs a comparison table like:

```
Model                          | tok/s | Load(s) | Quality Score
-------------------------------|-------|---------|--------------
llama.cpp Q4_K_M               | 12.4  | 1.2     | 9.2/10
llama.cpp Q3_K_M               |  9.2  | 1.1     | 8.5/10
llama.cpp Q2_K                 | 11.0  | 0.9     | 7.1/10
Ollama Q4_K_M                  |  6.4  | 12.0    | 9.2/10
Ollama Q3_K_M                  |  4.8  |  8.6    | 8.5/10
Ollama Q2_K                    |  5.1  |  6.5    | 6.8/10
```

## Why llama.cpp is Faster

llama.cpp bypasses Ollama's HTTP API overhead, model management layer, and multi-model scheduling. It loads the GGUF directly into CUDA memory and runs inference with minimal indirection. For single-model benchmarking on resource-constrained devices, this 2× speedup is significant.

## Clinic Deployment Recommendation

For a naturopathic clinic assistant running on Jetson Orin Nano 8GB:

1. **Use llama.cpp with Q4_K_M** for production inference (12.4 tok/s)
2. **Keep Ollama as a fallback** for tools that expect the HTTP API
3. **Always stop gdm3** before loading 7B models
4. **Monitor CmaFree** — if < 50 MB, models won't load on GPU
5. **Use `--flash-attn on`** for ~10–15% throughput improvement

## File Structure

```
jetson-llm-benchmark/
├── README.md                          # This file
├── benchmark.py                       # Main benchmark script
├── scripts/
│   ├── analyze.py                     # Results analyzer
│   └── compare_engines.py             # llama.cpp vs Ollama comparison
├── prompts/
│   └── clinical-hypothyroid.txt       # Example clinical prompt
├── results/
│   └── README.md                      # How to interpret results
└── docs/
    └── JETPACK-FIX.md                 # Detailed JetPack upgrade guide
```

## Contributing

1. Fork the repo
2. Add your hardware specs to the results table
3. Submit a PR with your benchmark data

## License

MIT — See LICENSE file.

## Credits

Benchmark methodology and fixes developed by Walker Kirkpatrick, ND for clinic edge-AI deployment on Jetson Orin Nano.
