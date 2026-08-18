# Jetson LLM Benchmark

![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano%208GB%20Super-76B900?logo=nvidia)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia)
![llama.cpp](https://img.shields.io/badge/llama.cpp-0b1bad1-blue)
![JetPack](https://img.shields.io/badge/JetPack-6%20%28R36.5.2%29-orange)
![Models](https://img.shields.io/badge/Models%20Benchmarked-7%20%282B--7B%29-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Reproducible benchmarking toolkit for running LLMs (2B-7B) on NVIDIA Jetson Orin Nano 8GB with llama.cpp CUDA builds. Includes fixes for JetPack memory bugs, CMA fragmentation, tensor core optimization, and a 24-model selection guide.

---

## TL;DR Results

### 2B/3B Models (llama-bench, GUI off, `-ngl 99 -fa 1`)

| Model | Params | Quant | Size | Gen tok/s | Prompt tok/s |
|-------|--------|-------|------|-----------|-------------|
| **Gemma 4 E2B** | 4.63B* | Q4_0 | 2.63 GiB | **28.2** | **1007** |
| **Gemma 2 2B** | 2.61B | Q4_0 | 1.51 GiB | **26.4** | **1148** |
| **LFM 2.5 2.6B** | 2.70B | Q4_K_M | 1.55 GiB | **25.9** | **933** |
| Qwen 2.5 3B | 3.09B | Q4_K_M | 1.79 GiB | 22.0 | 805 |
| Llama 3.2 3B | 3.21B | Q4_K_M | 1.87 GiB | 21.3 | 814 |
| Hermes 3 3B | 3.21B | Q4_K_M | 1.87 GiB | 21.3 | 815 |
| Phi-3 Mini 3.8B | 3.82B | Q4_0 | 2.03 GiB | 21.6 | 690 |

*Gemma 4 E2B is a MatFormer: 5.1B total params, 2B active per token.

### 7B Models (llama.cpp vs Ollama)

| Backend | Quantization | Gen tok/s | Prompt tok/s |
|---------|-------------|-----------|-------------|
| **llama.cpp** | Q4_K_M | **12.4** | **437** |
| **llama.cpp** | Q2_K | **11.0** | -- |
| Ollama | Q4_K_M | **6.4** | 18.2 |

**Key finding:** llama.cpp is **~2x faster** than Ollama for the same GGUF models.

### Gemma 4 E2B with MTP (Multi-Token Prediction)

| Config | Prose tok/s | Code tok/s | Draft Accept |
|--------|------------|-----------|-------------|
| MTP OFF (baseline) | **31.1** | 31.1 | -- |
| MTP ON, n-max=4 | 19.2 | **42.7** | 72% (code) |

MTP helps code (37% faster), hurts prose (38% slower). Use MTP for code only.

---

## Quick-Pick: Which Model Should I Use?

| Task | Best Model | Gen tok/s | Why |
|------|-----------|-----------|-----|
| Fastest generation | Gemma 4 E2B | 28.2 | MatFormer, thinking model, newest |
| Fastest non-Google | LFM 2.5 2.6B | 25.9 | Liquid AI, novel architecture |
| Best quality 3B chat | Qwen 2.5 3B | 22.0 | Strong reasoning, tool use |
| Best agentic 3B | Hermes 3 3B | 21.3 | Nous Research, ChatML, function calling |
| Best math/logic 3B | Phi-3 3.8B | 21.6 | Microsoft, strong STEM |
| Best general 3B | Llama 3.2 3B | 21.3 | Meta baseline, broad capability |
| Best coding (with MTP) | Gemma 4 E2B | 42.7 | MTP gives 72% draft acceptance on code |
| Best 7B quality | Qwen 2.5 7B | 12.4 | Most capable, needs GUI off |

---

## Hardware

| Spec | Value |
|------|-------|
| Device | NVIDIA Jetson Orin Nano 8GB (Super) |
| GPU | 1024 CUDA cores, **32 Ampere tensor cores** (3rd-gen, sm_87) |
| GPU Clock | 306-918 MHz (Super Mode) |
| RAM | 8 GB unified (CPU+GPU), ~102.4 GB/s bandwidth |
| Storage | 128 GB NVMe SSD |
| JetPack | 6 (L4T R36.5.2) |
| CUDA | 12.6 |
| OS | Ubuntu 22.04 ARM64 |
| Power | 25W mode |

---

## Tensor Core Optimization

The Jetson Orin Nano has 8 SMs (4 TPCs x 2 SMs), each with 4 third-gen Ampere tensor cores = **32 tensor cores** supporting FP16, BF16, INT8, and INT4 matrix multiply-accumulate.

### Build flags that map to tensor cores

| CMake Flag | Effect |
|------------|--------|
| `GGML_CUDA=ON` | CUDA backend (enables GPU offload) |
| `GGML_CUDA_F16=ON` | FP16 tensor cores for matmul |
| `GGML_CUDA_FA=ON` | Tensor cores for flash attention |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | Flash attention with all quant types |
| `GGML_CUDA_GRAPHS=ON` | CUDA graphs reduce kernel launch overhead |

### MMQ vs cuBLAS: let auto-selection work

llama.cpp auto-selects between MMQ (INT8 tensor cores, for generation) and cuBLAS (FP16 tensor cores, for prompt eval). **Do NOT force either**:

| Mode | pp512 (t/s) | tg128 (t/s) |
|------|-----------|-----------|
| Default (auto) | 814 | 21.25 |
| Force MMQ | 813 | 21.26 |
| Force cuBLAS | 815 | 21.24 |

Forcing MMQ or cuBLAS can **crater prompt eval** on some models (gemma2:2b drops from 1148 to 122 t/s).

### Flash attention: mandatory

Turning flash attention OFF devastates prompt eval (the attention matrix uses FP16 tensor cores):

| Model | FA ON | FA OFF | Speedup |
|-------|-------|--------|---------|
| gemma2:2b | 1148 | 23 | **50x** |
| lfm2.5:2.6b | 933 | 56 | **17x** |
| llama3.2:3b | 814 | 129 | **6.3x** |
| phi3:3.8b | 690 | 102 | **6.8x** |

**Always use `-fa on`. Never disable it.**

---

## Build llama.cpp with CUDA

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="87" \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

Binaries at `build/bin/llama-cli`, `build/bin/llama-server`, `build/bin/llama-bench`.

---

## Pre-Inference Checklist

```bash
sudo jetson_clocks              # max GPU/CPU clocks
sudo systemctl stop gdm3        # frees ~600MB RAM (critical for 7B)
pkill -f "ollama serve"         # free GPU memory if Ollama is running
```

---

## Benchmarking

### Quick benchmark (single model)

```bash
~/llama.cpp/build/bin/llama-bench \
  -m ~/models/bench-gguf/<model>.gguf \
  -ngl 99 -fa 1 -p 512 -n 128
```

### Full inference test with timing

```bash
~/llama.cpp/build/bin/llama-cli \
  -m ~/models/bench-gguf/<model>.gguf \
  -p "Explain the difference between hypothyroidism and hyperthyroidism." \
  -n 200 -c 1024 \
  --temp 0.3 -ngl 99 -fa on \
  --no-conversation --no-display-prompt -st
```

Key flags:
- `-ngl 99`: offload all layers to GPU (uses tensor cores)
- `-fa on`: flash attention (mandatory, 6-50x speedup)
- `-st`: print timing stats
- Do NOT set `GGML_CUDA_FORCE_MMQ` or `GGML_CUDA_FORCE_CUBLAS`

---

## Gemma 4 E2B: Loading Fix

The Ollama `gemma4:e2b` blob fails in llama.cpp with `wrong number of tensors; expected 2012, got 601` because it bundles vision/audio tensors that confuse the text-only loader.

### Fix: download clean GGUFs from HuggingFace

```bash
# Text model (Q4_0, 2.7 GB)
curl -L -o ~/models/bench-gguf/gemma-4-E2B-it-Q4_0.gguf \
  "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_0.gguf"

# Multimodal projector (Q8_0, 531 MB) - for vision/audio input
curl -L -o ~/models/bench-gguf/mmproj-gemma-4-E2B-it-Q8_0.gguf \
  "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/mmproj-gemma-4-E2B-it-Q8_0.gguf"
```

### Text-only inference

```bash
~/llama.cpp/build/bin/llama-cli \
  -m ~/models/bench-gguf/gemma-4-E2B-it-Q4_0.gguf \
  -p "<prompt>" -n 200 -c 2048 \
  --temp 0.3 -ngl 99 -fa on \
  --no-conversation --no-display-prompt -st
```

### Multimodal (vision) inference

```bash
~/llama.cpp/build/bin/llama-mtmd-cli \
  -m ~/models/bench-gguf/gemma-4-E2B-it-Q4_0.gguf \
  --mmproj ~/models/bench-gguf/mmproj-gemma-4-E2B-it-Q8_0.gguf \
  --image <image.jpg> \
  -p "Describe this image." \
  -n 200 -c 2048 --temp 0.3 -ngl 99 -fa on
```

### Gemma 4 notes

- MatFormer architecture: 5.1B total params, 2B active per token
- **Thinking model**: outputs `reasoning_content` before the answer. Use `max_tokens` 800+ or the model never finishes thinking and returns empty content
- Use `llama-mtmd-cli` (not deprecated `llama-gemma3-cli`) for multimodal
- MTP (multi-token prediction) support: use `--spec-type draft-mtp --spec-draft-n-max 4` for code generation (42.7 tok/s, 72% acceptance). Turn OFF for prose.

---

## Symlink Ollama Blobs for llama.cpp

Ollama stores GGUF blobs at `~/.ollama/models/blobs/sha256-*`. Symlink them:

```bash
mkdir -p ~/models/bench-gguf
BLOB=$(python3 -c "import json; m=json.load(open('$HOME/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>')); print(m['layers'][0]['digest'].split(':')[1])")
ln -sf ~/.ollama/models/blobs/sha256-$BLOB ~/models/bench-gguf/<name>.gguf
```

---

## The Problem We Solved

### JetPack R36.4.7 CUDA IOVA/NvMap Bug

JetPack R36.4.7 has a critical regression in the CUDA contiguous memory allocator causing `NvMapMemAllocInternalTagged: error 12 (ENOMEM)` for models > ~1.1 GB. 7B models fail to load on GPU despite plenty of free RAM.

**Fix:** Upgrade to JetPack R36.5.2+. This is a kernel-level fix.

```bash
sudo sed -i 's/r36.3/r36.5/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
sudo sed -i 's/r36.4/r36.5/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
sudo apt update && sudo apt upgrade -y
sudo reboot
```

### CMA Fragmentation (Desktop Session)

Even after the JetPack fix, GNOME desktop fragments CMA memory. Stop the display manager before loading 7B models:

```bash
sudo systemctl stop gdm3    # frees ~600-800 MB CMA
```

---

## The Paradox: Why Q4_K_M Is Faster Than Q2/Q3

On our Jetson, Q4_K_M was the fastest 7B quant, outpacing both Q2_K and Q3_K_M. The reason: **tensor cores**. Jetson's Ampere GPU has dedicated hardware for 4-bit matrix math. At Q4, the model lands in a sweet spot where tensor cores work efficiently. Drop to Q2 or Q3 and you lose that hardware pathway.

---

## Recommended Launch Commands

### 7B prose (12.4 tok/s)

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/llama-cpp-models/qwen2.5-7b-instruct-q4_k_m.gguf \
  -ngl 99 -c 1024 -t 6 \
  --host 127.0.0.1 --port 8080
```

### Code with MTP (42.7 tok/s, Gemma 4 E2B)

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-4-E2B-it-Q4_K_M.gguf \
  -md ~/models/gemma-4-e2b-mtp/mtp-gemma-4-E2B-it.gguf \
  -ngl 99 -ngld 99 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -c 1024 -t 6 \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --host 127.0.0.1 --port 8080
```

### 3B chat (fastest, 28 tok/s)

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/bench-gguf/gemma-4-E2B-it-Q4_0.gguf \
  -ngl 99 -c 2048 -t 6 -fa on \
  --host 127.0.0.1 --port 8080
```

---

## File Structure

```
jetson-llm-benchmark/
+-- README.md                          # This file
+-- benchmark.py                       # Main benchmark script
+-- build_guide_pdf.py                 # Script to regenerate model guide PDF
+-- Jetson_Edge_LLM_Model_Guide.pdf    # 24-model selection guide (landscape)
+-- Hermes3_3B_50_Details.pdf          # 50 unique details about Hermes 3 3B
+-- auto_bench_results.txt             # Raw benchmark output
+-- scripts/
|   +-- analyze.py                     # Results analyzer
|   +-- compare_engines.py             # llama.cpp vs Ollama comparison
+-- prompts/
|   +-- clinical-hypothyroid.txt       # Example clinical prompt
+-- results/
|   +-- README.md                      # How to interpret results
+-- docs/
    +-- JETPACK-FIX.md                 # Detailed JetPack upgrade guide
```

---

## Memory Management

| Action | RAM Freed | When Needed |
|--------|-----------|-------------|
| Stop GUI (`stop gdm3`) | ~600 MB | Always for 7B; optional for 2B/3B |
| Kill Ollama (`pkill ollama`) | ~200 MB | When running llama.cpp |
| Free GPU (`fuser -k /dev/nvidia*`) | varies | After crashes |
| Monitor CMA | -- | `grep CmaFree /proc/meminfo` (need >50 MB) |

| Model Size | Runtime RAM | Fits with GUI? |
|-----------|------------|----------------|
| 2B Q4 | ~1.5 GB | Yes |
| 3B Q4 | ~2.0 GB | Yes |
| Gemma 4 E2B + MTP | ~4.2 GB | No (stop gdm3) |
| 7B Q4 | ~5.2 GB | No (stop gdm3) |

---

## Contributing

1. Fork the repo
2. Add your hardware specs to the results table
3. Submit a PR with your benchmark data

## License

MIT - See LICENSE file.