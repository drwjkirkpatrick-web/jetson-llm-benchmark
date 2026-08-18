# Jetson LLM Benchmark

![Platform](https://img.shields.io/badge/Platform-Jetson%20Nano%208GB-76B900?logo=nvidia)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia)
![llama.cpp](https://img.shields.io/badge/llama.cpp-0b1bad1-blue)
![JetPack](https://img.shields.io/badge/JetPack-6%20(R36.5.2)-orange)
![Models](https://img.shields.io/badge/Models%20Benchmarked-18%20(1B--4B)-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Reproducible benchmarking toolkit for running LLMs (1B-4B) on NVIDIA Jetson Nano 8GB with llama.cpp CUDA builds. Includes fixes for JetPack memory bugs, CMA fragmentation, tensor core optimization, model download troubleshooting, and two 5-prompt benchmark suites with quality scoring.

---

## Table of Contents

- [TL;DR Results](#tldr-results)
- [Quick-Pick: Which Model Should I Use?](#quick-pick-which-model-should-i-use)
- [Hardware](#hardware)
- [Tensor Core Optimization](#tensor-core-optimization)
- [Build llama.cpp with CUDA](#build-llamacpp-with-cuda)
- [Pre-Inference Checklist](#pre-inference-checklist)
- [Benchmarking](#benchmarking)
- [Model Download Guide and Fixes](#model-download-guide-and-fixes)
- [The 18 Benchmark Models](#the-18-benchmark-models)
- [Two Benchmark Suites](#two-benchmark-suites)
- [Quality Scoring](#quality-scoring)
- [Gemma 4 E2B: Loading and OOM Fix](#gemma-4-e2b-loading-and-oom-fix)
- [Thinking Models: --jinja Flag](#thinking-models---jinja-flag)
- [Memory Management](#memory-management)
- [JetPack Fix](#jetpack-fix)
- [File Structure](#file-structure)
- [Agent Instructions](#agent-instructions)

---

## TL;DR Results

### General 5-Prompt Benchmark (18 models, GUI off, `-ngl 99 -fa on --jinja`)

| Model | Params | Quant | Size | Gen tok/s | Avg Quality | QS |
|-------|--------|-------|------|-----------|-------------|-----|
| **Gemma 3 1B** | 1.0B | Q4_K_M | 0.76 GiB | **38.1** | 6.0 | 22.9 |
| **CodeGemma 2B** | 2.51B | Q4_0 | 1.44 GiB | 30.7 | 2.6 | 8.0 |
| **Granite 3.0 2B** | 2.63B | Q4_K_M | 1.49 GiB | 27.2 | 6.4 | 17.4 |
| **Granite 3.2 2B** | 2.63B | Q4_K_M | 1.44 GiB | 27.2 | 7.0 | 19.0 |
| **StableLM Zephyr** | 1.6B | Q4_K_M | 1.5 GiB | 27.5 | 7.0 | 19.3 |
| **Gemma 2 2B** | 2.61B | Q4_0 | 1.51 GiB | 25.2 | **8.2** | **20.7** |
| **Gemma 4 E2B** | 4.63B* | Q4_0 | 2.63 GiB | 26.7 | 5.8 | 15.5 |
| **LFM 2.5 2.6B** | 2.70B | Q4_K_M | 1.55 GiB | 25.3 | 6.0 | 15.2 |
| **StarCoder2 3B** | 3.3B | Q4_K_M | 1.6 GiB | 28.6 | 1.8 | 5.2 |
| **Qwen 2.5 3B** | 3.09B | Q4_K_M | 1.79 GiB | 21.5 | **7.6** | 16.3 |
| **Qwen2.5-Coder 3B** | 3.09B | Q4_K_M | 1.8 GiB | 21.5 | 7.4 | 15.9 |
| **Llama 3.2 3B** | 3.21B | Q4_K_M | 1.87 GiB | 20.8 | 7.2 | 15.0 |
| **Hermes 3 3B** | 3.21B | Q4_K_M | 1.87 GiB | 20.8 | 7.2 | 15.0 |
| **Phi-3 3.8B** | 3.82B | Q4_0 | 2.03 GiB | 20.8 | 6.8 | 14.1 |
| **Granite 4 3B** | 3.3B | Q4_K_M | 2.1 GiB | 19.6 | 6.6 | 12.9 |
| **Granite 4.1 3B** | 3.3B | Q4_K_M | 2.0 GiB | 19.6 | 7.0 | 13.7 |
| **SmallThinker 3B** | 3.4B | Q4_K_M | 3.36 GiB | 19.2 | 6.4 | 12.3 |
| **Orca-Mini 3B** | 3.0B | Q4_K_M | 1.9 GiB | 7.8 | 2.8 | 2.2 |

*Gemma 4 E2B is a MatFormer: 5.1B total params, 2B active per token.

QS = Quality-Speed = (avg_quality x avg_gen_tps) / 10. Higher is better.

### Coding 5-Prompt Benchmark (18 models, 5 languages)

| Model | Gen tok/s | Avg Quality | QS | HTML | Python | C | BASIC | Julia |
|-------|-----------|-------------|-----|------|--------|---|-------|-------|
| **Gemma 3 1B** | **38.1** | 7.4 | **28.2** | 8 | 8 | 7 | 7 | 7 |
| **Gemma 2 2B** | 25.0 | **8.6** | 21.5 | 9 | 9 | 9 | 7 | 9 |
| **Qwen 2.5 3B** | 21.4 | **8.6** | 18.4 | 9 | 9 | 9 | 8 | 8 |
| **Granite 3.2 2B** | 27.0 | 8.4 | **22.7** | 9 | 8 | 8 | 8 | 9 |
| **Granite 3.0 2B** | 27.1 | 7.8 | 21.1 | 8 | 8 | 8 | 7 | 8 |
| **Granite 4.1 3B** | 19.6 | 7.8 | 15.3 | 8 | 8 | 8 | 7 | 8 |
| **Qwen2.5-Coder 3B** | 21.4 | 7.6 | 16.3 | 8 | 7 | 8 | 8 | 7 |
| **Llama 3.2 3B** | 20.6 | 7.6 | 15.7 | 8 | 8 | 8 | 7 | 7 |
| **Phi-3 3.8B** | 18.7 | 7.4 | 13.9 | 8 | 8 | 8 | 6 | 7 |
| **Gemma 3 1B** | 38.1 | 7.4 | **28.2** | 8 | 8 | 7 | 7 | 7 |
| **StableLM Zephyr** | 25.4 | 7.0 | 17.8 | 7 | 7 | 8 | 7 | 6 |
| **LFM 2.5 2.6B** | 25.3 | 7.0 | 17.7 | 7 | 7 | 7 | 7 | 7 |
| **Hermes 3 3B** | 20.7 | 7.0 | 14.5 | 6 | 7 | 8 | 7 | 7 |
| **SmallThinker 3B** | 19.2 | 7.0 | 13.4 | 7 | 7 | 7 | 7 | 7 |
| **Granite 4 3B** | 19.6 | 7.0 | 13.7 | 7 | 8 | 7 | 6 | 7 |
| **Gemma 4 E2B** | 26.8 | 5.8 | 15.5 | 6 | 6 | 6 | 5 | 6 |
| **CodeGemma 2B** | 30.7 | 4.4 | 13.5 | 3 | 6 | 7 | 1 | 5 |
| **Orca-Mini 3B** | 12.5 | 2.6 | 3.3 | 1 | 3 | 5 | 1 | 3 |
| **StarCoder2 3B** | 28.7 | 2.0 | 5.7 | 2 | 2 | 2 | 2 | 2 |

### 7B Models (llama.cpp vs Ollama, from earlier testing)

| Backend | Quantization | Gen tok/s | Prompt tok/s |
|---------|-------------|-----------|-------------|
| **llama.cpp** | Q4_K_M | **12.4** | **437** |
| **llama.cpp** | Q2_K | **11.0** | -- |
| Ollama | Q4_K_M | **6.4** | 18.2 |

**Key finding:** llama.cpp is ~2x faster than Ollama for the same GGUF models.

---

## Quick-Pick: Which Model Should I Use?

| Task | Best Model | Gen tok/s | Why |
|------|-----------|-----------|-----|
| Fastest generation | Gemma 3 1B | 38.1 | Smallest model, blazing fast, decent quality |
| Best quality overall | Gemma 2 2B | 25.2 | Highest avg quality (8.2/10), excellent code |
| Best quality-speed | Granite 3.2 2B | 27.2 | QS 22.7 on coding, fast AND high quality |
| Best quality 3B | Qwen 2.5 3B | 21.5 | Ties Gemma 2 on code quality (8.6), good at everything |
| Best coding 3B | Qwen2.5-Coder 3B | 21.5 | Code specialist, ties Gemma 2 on coding |
| Best agentic 3B | Hermes 3 3B | 20.8 | Nous Research, ChatML, function calling |
| Best math/logic | Phi-3 3.8B | 20.8 | Microsoft, strong STEM |
| Best general 3B | Llama 3.2 3B | 20.8 | Meta baseline, broad capability |
| Best value 2B | Granite 3.2 2B | 27.2 | IBM, consistently good across all tasks |
| Surprise performer | StableLM Zephyr | 27.5 | 1.6B params but 7.0/10 quality |
| Best coding (with MTP) | Gemma 4 E2B | 42.7 | MTP gives 72% draft acceptance on code |
| Best 7B quality | Qwen 2.5 7B | 12.4 | Most capable, needs GUI off |

**Models to avoid:**
- **Orca-Mini 3B**: Refuses tasks, wrong math, very slow (7.8 tok/s)
- **StarCoder2 3B**: Base code model, not instruction-tuned, echoes prompts
- **CodeGemma 2B**: Code completion model, echoes prompts, low quality on non-code tasks

---

## Hardware

| Spec | Value |
|------|-------|
| Device | NVIDIA Jetson Nano 8GB |
| GPU | 1024 CUDA cores, 32 Ampere tensor cores (3rd-gen, sm_87) |
| GPU Clock | 306-918 MHz (Super Mode) |
| RAM | 8 GB unified (CPU+GPU), ~102.4 GB/s bandwidth |
| Storage | 128 GB NVMe SSD |
| JetPack | 6 (L4T R36.5.2) |
| CUDA | 12.6 |
| OS | Ubuntu 22.04 ARM64 |
| Power | 25W mode |

---

## Tensor Core Optimization

The Jetson has 8 SMs (4 TPCs x 2 SMs), each with 4 third-gen Ampere tensor cores = 32 tensor cores supporting FP16, BF16, INT8, and INT4 matrix multiply-accumulate.

### Build flags that map to tensor cores

| CMake Flag | Effect |
|------------|--------|
| `GGML_CUDA=ON` | CUDA backend (enables GPU offload) |
| `GGML_CUDA_F16=ON` | FP16 tensor cores for matmul |
| `GGML_CUDA_FA=ON` | Tensor cores for flash attention |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | Flash attention with all quant types |
| `GGML_CUDA_GRAPHS=ON` | CUDA graphs reduce kernel launch overhead |

### MMQ vs cuBLAS: let auto-selection work

llama.cpp auto-selects between MMQ (INT8 tensor cores, for generation) and cuBLAS (FP16 tensor cores, for prompt eval). Do NOT force either:

| Mode | pp512 (t/s) | tg128 (t/s) |
|------|-----------|-----------|
| Default (auto) | 814 | 21.25 |
| Force MMQ | 813 | 21.26 |
| Force cuBLAS | 815 | 21.24 |

Forcing MMQ or cuBLAS can crater prompt eval on some models (gemma2:2b drops from 1148 to 122 t/s).

### Flash attention: mandatory

Turning flash attention OFF devastates prompt eval:

| Model | FA ON | FA OFF | Speedup |
|-------|-------|--------|---------|
| gemma2:2b | 1148 | 23 | 50x |
| lfm2.5:2.6b | 933 | 56 | 17x |
| llama3.2:3b | 814 | 129 | 6.3x |
| phi3:3.8b | 690 | 102 | 6.8x |

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
sudo systemctl stop gdm3        # frees ~600MB RAM (critical for 3B+)
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
  -n 2000 -c 4096 \
  --temp 0.3 -ngl 99 -fa on \
  --no-conversation --no-display-prompt \
  --jinja -st
```

Key flags:
- `-ngl 99`: offload all layers to GPU (uses tensor cores)
- `-fa on`: flash attention (mandatory, 6-50x speedup)
- `--jinja`: process chat template (required for thinking models)
- `-st`: print timing stats
- `--no-conversation --no-display-prompt`: clean output for benchmarking
- Do NOT set `GGML_CUDA_FORCE_MMQ` or `GGML_CUDA_FORCE_CUBLAS`
- For models that OOM: reduce to `-c 2048`

---

## Model Download Guide and Fixes

### Method 1: Ollama pull + symlink (standard models)

```bash
ollama pull <model>:<tag>

# Symlink the blob for llama.cpp:
BLOB=$(python3 -c "import json; m=json.load(open('$HOME/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>')); print(m['layers'][0]['digest'].split(':')[1])")
ln -sf ~/.ollama/models/blobs/sha256-$BLOB ~/models/bench-gguf/<name>.gguf
```

**Tag note:** Some models use `:latest` (stablelm-zephyr) while others use `:3b` or `:2b`. Check `ollama list` to find the correct tag. granite4.1, orca-mini, and starcoder2 use `:3b` not `:latest`.

### Method 2: Direct GGUF download (broken Ollama blobs)

Some Ollama blobs fail in llama.cpp because they bundle multimodal (vision/audio) tensors. Download clean text-only GGUFs from HuggingFace instead:

```bash
curl -L -o ~/models/bench-gguf/<name>.gguf \
  "https://huggingface.co/ggml-org/<model>-GGUF/resolve/main/<model>-Q4_K_M.gguf"
```

### Models that need alternative downloads

| Model | Problem | Fix |
|-------|---------|-----|
| **Gemma 4 E2B** | Ollama blob: `wrong number of tensors; expected 2012, got 601` | Download from `huggingface.co/ggml-org/gemma-4-E2B-it-GGUF` |
| **gemma3n:e2b** | Same multimodal tensor issue | Download clean text-only GGUF from `huggingface.co/ggml-org` |
| **SmallThinker 3B** | Ollama blob produces only 49 tokens | Download from `huggingface.co/ggml-org/SmallThinker-3B-GGUF` |
| **ministral-3** | Wrong tensor count (multimodal) | No clean GGUF available; skip |
| **qwen3.5:2b** | `rope.dimension_sections wrong array length` | No clean GGUF available; skip |

### Models with expected low scores (not bugs)

| Model | Why it scores low |
|-------|-------------------|
| **StarCoder2 3B** | Base code model, not instruction-tuned. Echoes prompts, generates completions. Expected behavior. |
| **CodeGemma 2B** | Code completion model. Echoes prompts, generates long code blocks. High on code, very low on creative/prose. |
| **Orca-Mini 3B** | Refuses creative tasks ("I cannot create HTML"), wrong math, chat loops, very slow. Model is unsuitable for these tasks. |

---

## The 18 Benchmark Models

| # | Model | Params | Quant | Size | Source | Notes |
|---|-------|--------|-------|------|--------|-------|
| 1 | Gemma 3 1B | 1.0B | Q4_K_M | 0.76 GiB | Ollama | Fastest model tested (38 tok/s) |
| 2 | Gemma 2 2B | 2.61B | Q4_0 | 1.51 GiB | Ollama | Quality champion (8.2/10 general, 8.6 coding) |
| 3 | CodeGemma 2B | 2.51B | Q4_0 | 1.44 GiB | Ollama | Code completion model, not instruction-tuned |
| 4 | Granite 3.0 2B | 2.63B | Q4_K_M | 1.49 GiB | Ollama | IBM dense 2B, solid all-rounder |
| 5 | Granite 3.2 2B | 2.63B | Q4_K_M | 1.44 GiB | Ollama | Best quality-speed on coding (QS 22.7) |
| 6 | LFM 2.5 2.6B | 2.70B | Q4_K_M | 1.55 GiB | Ollama | Liquid AI, novel hybrid architecture |
| 7 | StableLM Zephyr | 1.6B | Q4_K_M | 1.5 GiB | Ollama | Surprise performer (7.0/10 at 1.6B) |
| 8 | Gemma 4 E2B | 4.63B* | Q4_0 | 2.63 GiB | HuggingFace | MatFormer, thinking model, needs --jinja |
| 9 | Qwen 2.5 3B | 3.09B | Q4_K_M | 1.79 GiB | Ollama | Ties Gemma 2 on code quality (8.6) |
| 10 | Qwen2.5-Coder 3B | 3.09B | Q4_K_M | 1.8 GiB | Ollama | Code specialist, good clinical prose too |
| 11 | Llama 3.2 3B | 3.21B | Q4_K_M | 1.87 GiB | Ollama | Meta baseline, broad capability |
| 12 | Hermes 3 3B | 3.21B | Q4_K_M | 1.87 GiB | Ollama | Nous Research, ChatML, function calling |
| 13 | Phi-3 3.8B | 3.82B | Q4_0 | 2.03 GiB | Ollama | Microsoft, strong STEM |
| 14 | Granite 4 3B | 3.3B | Q4_K_M | 2.1 GiB | Ollama | IBM Granite 4 |
| 15 | Granite 4.1 3B | 3.3B | Q4_K_M | 2.0 GiB | Ollama | IBM Granite 4.1, good code+math |
| 16 | SmallThinker 3B | 3.4B | Q4_K_M | 3.36 GiB | HuggingFace | Thinking model, needs --jinja + 2000 tokens |
| 17 | Orca-Mini 3B | 3.0B | Q4_K_M | 1.9 GiB | Ollama | Refuses tasks, very slow, unsuitable |
| 18 | StarCoder2 3B | 3.3B | Q4_K_M | 1.6 GiB | Ollama | Base code model, not instruction-tuned |

*Gemma 4 E2B is a MatFormer: 5.1B total params, 2B active per token.

---

## Two Benchmark Suites

### General 5-Prompt Suite

Tests how models handle different styles of content:

| Prompt | ID | Description |
|--------|----|-------------|
| Code Generation | `code` | Write a Python function with type hints and docstring |
| Iambic Pentameter | `iambic` | Write 14-line poem in strict iambic pentameter |
| Clinical Prose | `prose` | Write a naturopathic clinical case writeup |
| Creative Writing | `creative` | Write a short story with specific sensory details |
| Mathematical Proof | `math` | Prove an algebraic identity step by step |

### Coding 5-Prompt Suite

Tests language breadth across 5 programming languages:

| Prompt | ID | Description |
|--------|----|-------------|
| HTML/CSS | `html` | Portfolio page with flexbox, responsive design, color scheme |
| Python | `python` | DataProcessor class with CSV/JSON, type hints, error handling |
| C | `c` | Thread-safe queue with pthread mutex, linked list, 5 functions |
| TRS-80 BASIC | `basic` | Text adventure game with 3 rooms, line numbers, INPUT/GOTO |
| Julia | `julia` | Newton-Raphson method with type annotations and docstring |

---

## Quality Scoring

Scores are 1-10 per model per prompt, based on:

- **Correctness**: Does the output solve the requested task?
- **Completeness**: Is the output complete or truncated?
- **Language accuracy**: Does code use correct syntax for the target language?
- **Instruction following**: Does the model follow all parts of the prompt?
- **Creativity/Style**: For creative tasks, is the output engaging and well-written?

Quality-Speed (QS) = (avg_quality x avg_gen_tps) / 10. Higher is better.

Typical score ranges:

| Score | Meaning |
|-------|---------|
| 9-10 | Excellent, complete, correct, could ship as-is |
| 7-8 | Good, mostly correct, minor issues |
| 5-6 | Acceptable but incomplete or with errors |
| 3-4 | Poor, major issues, mostly wrong |
| 1-2 | Refused, echoed prompt, or completely wrong |

---

## Gemma 4 E2B: Loading and OOM Fix

### The Ollama blob problem

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

### OOM during coding benchmark

At 2.63 GB with 4096 context, Gemma 4 E2B can OOM on longer prompts. The symptom is `the server exited before becoming ready`.

**Fix:** Reduce context to `-c 2048`:

```bash
~/llama.cpp/build/bin/llama-cli \
  -m ~/models/bench-gguf/gemma-4-E2B-it-Q4_0.gguf \
  -p "<prompt>" -n 2000 -c 2048 \
  --temp 0.3 -ngl 99 -fa on \
  --no-conversation --no-display-prompt \
  --jinja -st
```

### Gemma 4 notes

- MatFormer architecture: 5.1B total params, 2B active per token
- Thinking model: outputs `[Start thinking]` before the answer. Use `max_tokens` 800+ or the model never finishes thinking and returns empty content
- Use `llama-mtmd-cli` (not deprecated `llama-gemma3-cli`) for multimodal
- MTP (multi-token prediction): use `--spec-type draft-mtp --spec-draft-n-max 4` for code generation (42.7 tok/s, 72% acceptance). Turn OFF for prose.

---

## Thinking Models: --jinja Flag

**Thinking models** (Gemma 4 E2B, SmallThinker 3B) MUST use the `--jinja` flag. Without it, they produce fewer than 50 tokens because the chat template engine is not loaded, and the model cannot properly separate its reasoning content from the answer.

```bash
# CORRECT: --jinja enables chat template processing
~/llama.cpp/build/bin/llama-cli \
  -m <model> -p "<prompt>" \
  -n 2000 -c 2048 \
  --jinja -st

# WRONG: without --jinja, thinking models output <50 tokens
~/llama.cpp/build/bin/llama-cli \
  -m <model> -p "<prompt>" \
  -n 2000 -c 2048 -st
```

SmallThinker 3B also requires a clean GGUF from HuggingFace. The Ollama blob produces only 49 tokens regardless of the `-n` setting.

---

## Memory Management

| Action | RAM Freed | When Needed |
|--------|-----------|-------------|
| Stop GUI (`stop gdm3`) | ~600 MB | Always for 3B+; optional for 1B-2B |
| Kill Ollama (`pkill ollama`) | ~200 MB | When running llama.cpp |
| Free GPU (`fuser -k /dev/nvidia*`) | varies | After crashes |
| Monitor CMA | -- | `grep CmaFree /proc/meminfo` (need >50 MB) |

| Model Size | Context | Runtime RAM | Fits with GUI? |
|-----------|---------|------------|----------------|
| 1B Q4 | 4096 | ~1.0 GB | Yes |
| 2B Q4 | 4096 | ~1.5 GB | Yes |
| 3B Q4 | 4096 | ~2.0 GB | Yes |
| 3.4B Q4 | 4096 | ~3.5 GB | Marginal |
| 4B+ Q4 | 4096 | ~4.5 GB | No (stop gdm3) |
| 4B+ Q4 | 2048 | ~3.5 GB | Yes (reduced context) |

### If a model OOMs

1. Stop GUI: `sudo systemctl stop gdm3`
2. Kill Ollama: `pkill -f "ollama serve"`
3. Reduce context: `-c 2048` instead of `-c 4096`
4. Check free RAM: `free -h` (need >1 GB free after model load)

---

## JetPack Fix

### JetPack R36.4.7 CUDA IOVA/NvMap Bug

JetPack R36.4.7 has a critical regression in the CUDA contiguous memory allocator causing `NvMapMemAllocInternalTagged: error 12 (ENOMEM)` for models larger than ~1.1 GB. 7B models fail to load on GPU despite plenty of free RAM.

**Fix:** Upgrade to JetPack R36.5.2+. This is a kernel-level fix.

```bash
sudo sed -i 's/r36.3/r36.5/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
sudo sed -i 's/r36.4/r36.5/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
sudo apt update && sudo apt upgrade -y
sudo reboot
```

### CMA Fragmentation (Desktop Session)

Even after the JetPack fix, GNOME desktop fragments CMA memory. Stop the display manager before loading 3B+ models:

```bash
sudo systemctl stop gdm3    # frees ~600-800 MB CMA
```

---

## The Paradox: Why Q4_K_M Is Faster Than Q2/Q3

On our Jetson, Q4_K_M was the fastest 7B quant, outpacing both Q2_K and Q3_K_M. The reason: tensor cores. Jetson's Ampere GPU has dedicated hardware for 4-bit matrix math. At Q4, the model lands in a sweet spot where tensor cores work efficiently. Drop to Q2 or Q3 and you lose that hardware pathway.

---

## File Structure

```
jetson-llm-benchmark/
+-- README.md                          # This file
+-- AGENTS.md                          # Agent instruction set (read this before running benchmarks)
+-- benchmark.py                       # Original benchmark script
+-- multiprompt_bench.py               # General 5-prompt benchmark harness
+-- bench_coding.py                    # Coding 5-prompt benchmark harness
+-- bench_new_models.py                # Standalone benchmark for new models
+-- bench_new3.py                      # Benchmark newest models on both suites
+-- rebench_fixed.py                   # Re-benchmark with --jinja fix
+-- quality_scoring.py                 # General quality scoring script
+-- score_coding.py                    # Coding quality scoring script
+-- build_multiprompt_pdf.py           # General benchmark PDF builder
+-- build_coding_pdf.py                # Coding benchmark PDF builder
+-- build_guide_pdf.py                 # Model selection guide PDF builder
+-- Multi_Prompt_Benchmark_Report.pdf  # 8-page general benchmark report
+-- Coding_Benchmark_Report.pdf        # 5-page coding benchmark report
+-- Jetson_Edge_LLM_Model_Guide.pdf    # 6-page model selection guide
+-- Hermes3_3B_50_Details.pdf          # 4-page Hermes 3 deep dive
+-- multiprompt_results.json           # General benchmark raw results
+-- coding_benchmark_results.json      # Coding benchmark raw results
+-- quality_scores.json                # General quality scores (flat format)
+-- coding_quality_scores.json         # Coding quality scores (flat format)
+-- auto_bench_results.txt             # Raw auto-benchmark output
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

## Agent Instructions

If you are an AI agent working on this project, read `AGENTS.md` first. It contains:
- Critical rules for benchmarking, scoring, and file handling
- Step-by-step guides for adding new models and rebuilding PDFs
- Banner stripping details for clean output capture
- OOM troubleshooting matrix
- Known issues and fixes for all problematic models
- JSON data file format specifications
- Git workflow with SSH key configuration

---

## Recommended Launch Commands

### Fastest chat (38 tok/s, Gemma 3 1B)

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/bench-gguf/gemma3-1b.gguf \
  -ngl 99 -c 2048 -t 6 -fa on \
  --host 127.0.0.1 --port 8080
```

### Best quality-speed (27 tok/s, Granite 3.2 2B)

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/bench-gguf/granite3.2-2b.gguf \
  -ngl 99 -c 4096 -t 6 -fa on \
  --host 127.0.0.1 --port 8080
```

### Best quality (25 tok/s, Gemma 2 2B)

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/bench-gguf/gemma2-2b.gguf \
  -ngl 99 -c 4096 -t 6 -fa on \
  --host 127.0.0.1 --port 8080
```

### Code with MTP (42.7 tok/s, Gemma 4 E2B)

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-4-E2B-it-Q4_0.gguf \
  -md ~/models/gemma-4-e2b-mtp/mtp-gemma-4-E2B-it.gguf \
  -ngl 99 -ngld 99 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -c 1024 -t 6 \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --host 127.0.0.1 --port 8080
```

### Thinking model (Gemma 4 E2B, reduced context)

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/bench-gguf/gemma-4-E2B-it-Q4_0.gguf \
  -ngl 99 -c 2048 -t 6 -fa on \
  --jinja \
  --host 127.0.0.1 --port 8080
```

---

## Contributing

1. Fork the repo
2. Add your hardware specs to the results table
3. Submit a PR with your benchmark data

## License

MIT - See LICENSE file.