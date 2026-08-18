# Agent Instructions: Jetson LLM Benchmark Project

> Operational guide for AI agents working on this project. Read this before running benchmarks, scoring outputs, building PDFs, or downloading models.

## Project Overview

Benchmark 1B-4B LLMs on NVIDIA Jetson Orin Nano 8GB using llama.cpp with CUDA. Two benchmark suites:
1. **General 5-prompt**: code, iambic pentameter, clinical prose, creative writing, math proof
2. **Coding 5-prompt**: HTML/CSS, Python, C, TRS-80 BASIC, Julia

Results stored as JSON, rendered to landscape A4 PDFs via reportlab, pushed to GitHub.

## Critical Rules

1. **Never have two subagents patch the same file.** Parent handles all shared-file edits sequentially after subagents finish.
2. **One subagent per file** when doing batch development.
3. **Always use `--jinja` flag** for thinking models (Gemma 4 E2B, SmallThinker 3B). Without it, thinking models produce fewer than 50 tokens.
4. **Always use `-fa on`** (flash attention). Turning it off causes 6-50x slowdown on prompt eval.
5. **Never force MMQ or cuBLAS.** Auto-selection is optimal on Jetson.
6. **Use `--no-conversation --no-display-prompt`** for clean output capture.
7. **Use `-st`** (stats) to get tok/s numbers in stderr.
8. **Strip the llama.cpp banner** from output before scoring. See `strip_banner()` in bench scripts.
9. **Quality scores are 1-10**. Quality-Speed = (avg_quality x avg_gen_tps) / 10.
10. **Do not commit junk keys** in JSON files. Quality score files should be flat: `{model: {prompt: {score, notes}}}`.

## llama.cpp Build

Commit `0b1bad1`. Build at `~/llama.cpp/build/bin/`.

```bash
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

Binaries: `build/bin/llama-cli`, `build/bin/llama-bench`, `build/bin/llama-server`.

## Benchmark Commands

### llama-bench (speed only)

```bash
~/llama.cpp/build/bin/llama-bench \
  -m ~/models/bench-gguf/<model>.gguf \
  -ngl 99 -fa 1 -p 512 -n 128
```

### Generation with quality capture

```bash
~/llama.cpp/build/bin/llama-cli \
  -m ~/models/bench-gguf/<model>.gguf \
  -p "<prompt>" \
  -n 2000 -c 4096 \
  --temp 0.3 -ngl 99 -fa on \
  --no-conversation --no-display-prompt \
  --jinja -st
```

**For models that OOM** (Gemma 4 E2B, models larger than 2.5GB): reduce context to `-c 2048`.

### Thinking models (Gemma 4 E2B, SmallThinker 3B)

MUST use `--jinja` and `-n 2000` or higher. Without `--jinja`, thinking models output fewer than 50 tokens. The `--jinja` flag enables the Jinja template engine that processes the model's chat template, which is required for thinking models to produce their reasoning content before the answer.

### Pre-inference checklist

```bash
sudo jetson_clocks              # max GPU/CPU clocks
sudo systemctl stop gdm3        # frees ~600MB RAM (required for 3B+ models)
pkill -f "ollama serve"         # free GPU memory if Ollama running
```

## Model Download Guide

### Method 1: Ollama pull + symlink (preferred for standard models)

```bash
ollama pull <model>:<tag>
# Then symlink the blob:
BLOB=$(python3 -c "import json; m=json.load(open('$HOME/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>')); print(m['layers'][0]['digest'].split(':')[1])")
ln -sf ~/.ollama/models/blobs/sha256-$BLOB ~/models/bench-gguf/<name>.gguf
```

### Method 2: Direct GGUF download (for broken Ollama blobs)

Some Ollama blobs fail in llama.cpp. Download clean GGUFs from HuggingFace instead.

## Models That Need Alternative Download Sources

### Gemma 4 E2B (MatFormer, multimodal)

**Problem:** Ollama `gemma4:e2b` blob fails with `wrong number of tensors; expected 2012, got 601`. The blob bundles vision/audio tensors that confuse the text-only loader.

**Fix:** Download clean text-only GGUF from ggml-org on HuggingFace:

```bash
curl -L -o ~/models/bench-gguf/gemma-4-E2B-it-Q4_0.gguf \
  "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_0.gguf"
```

**OOM during coding benchmark:** At 2.63GB with 4096 context, Gemma 4 E2B can OOM on longer prompts. Reduce context to `-c 2048`. This is safe because the model's thinking overhead means it rarely uses the full context window.

**Additional notes:**
- MatFormer: 5.1B total params, 2B active per token
- Thinking model: outputs `[Start thinking]` before the answer. Use `--jinja` + 2000+ tokens.
- MTP support: `--spec-type draft-mtp --spec-draft-n-max 4` for code (42.7 tok/s, 72% acceptance). Turn OFF for prose.

### gemma3n:e2b (MatFormer, multimodal)

**Problem:** Same tensor count issue as Gemma 4 E2B. Ollama blob has multimodal tensors that fail in llama.cpp text-only loading.

**Fix:** Download clean text-only GGUF from ggml-org on HuggingFace (when available). Check `https://huggingface.co/ggml-org` for the model name.

### SmallThinker 3B

**Problem:** Ollama blob produces only 49 tokens regardless of `-n` setting.

**Fix:** Download clean GGUF from ggml-org on HuggingFace:

```bash
# Search for the correct repo name
curl -L -o ~/models/bench-gguf/smallthinker-3b.gguf \
  "https://huggingface.co/ggml-org/SmallThinker-3B-GGUF/resolve/main/SmallThinker-3B-Q4_K_M.gguf"
```

Must use `--jinja` flag. Thinking model: outputs reasoning before answer.

### ministral-3:latest

**Problem:** Wrong tensor count. Multimodal model, not compatible with text-only llama.cpp loader.

**Fix:** No clean text-only GGUF available at time of testing. Skip this model.

### qwen3.5:2b

**Problem:** `rope.dimension_sections wrong array length` error. The GGUF metadata has incorrect rope scaling parameters.

**Fix:** No clean GGUF available. Skip this model.

### StarCoder2 3B

**Not an error, but important:** StarCoder2 is a **base code model**, not instruction-tuned. It echoes prompts and generates code completions rather than following instructions. Quality scores will be very low (1-2/10) for all prompt types. This is expected behavior, not a bug.

### CodeGemma 2B

**Similar to StarCoder2:** CodeGemma is primarily a code completion model. It generates very long outputs (2000+ tokens) but often echoes the prompt and produces code-completion-style output rather than following instructions. Quality scores reflect this: high on code, very low on creative/prose tasks.

### Orca-Mini 3B

**Problem:** Refuses creative writing tasks ("I cannot create HTML"), generates chat loops, produces wrong math. Also very slow (7.3 tok/s vs 20+ for other 3B models).

**Fix:** No fix needed. This model is unsuitable for the benchmark tasks. Document low scores and move on.

## Model Symlink Reference

All benchmarked models are symlinked in `~/models/bench-gguf/`. Here are the 18 working models:

| Symlink Name | Ollama Tag | Size | Source |
|---|---|---|---|
| `gemma3-1b.gguf` | gemma3:1b | 0.76 GiB | Ollama blob |
| `gemma2-2b.gguf` | gemma2:2b | 1.51 GiB | Ollama blob |
| `codegemma-2b.gguf` | codegemma:2b | 1.44 GiB | Ollama blob |
| `granite3-dense-2b.gguf` | granite3-dense:2b | 1.49 GiB | Ollama blob |
| `granite3.2-2b.gguf` | granite3.2:2b | 1.44 GiB | Ollama blob |
| `lfm2.5-2.6b.gguf` | lfm2.5:2.6b | 1.55 GiB | Ollama blob |
| `stablelm-zephyr.gguf` | stablelm-zephyr | 1.5 GiB | Ollama blob |
| `gemma4-e2b.gguf` | gemma4:e2b | 2.63 GiB | HuggingFace (clean GGUF) |
| `gemma-4-E2B-it-Q4_0.gguf` | (direct) | 2.63 GiB | HuggingFace (clean GGUF) |
| `granite4-3b.gguf` | granite4:3b | 2.1 GiB | Ollama blob |
| `granite4.1-3b.gguf` | granite4.1:3b | 2.0 GiB | Ollama blob |
| `qwen2.5-3b.gguf` | qwen2.5:3b | 1.79 GiB | Ollama blob |
| `qwen2.5-coder-3b.gguf` | qwen2.5-coder:3b | 1.8 GiB | Ollama blob |
| `llama3.2-3b.gguf` | llama3.2:3b | 1.87 GiB | Ollama blob |
| `hermes3-3b.gguf` | hermes3:3b | 1.87 GiB | Ollama blob |
| `phi3-3.8b.gguf` | phi3:3.8b | 2.03 GiB | Ollama blob |
| `smallthinker-3b.gguf` | smallthinker:3b | 3.36 GiB | HuggingFace (clean GGUF) |
| `orca-mini-3b.gguf` | orca-mini:3b | 1.9 GiB | Ollama blob |
| `starcoder2-3b.gguf` | starcoder2:3b | 1.6 GiB | Ollama blob |

**Ollama tag note:** Some models use `:latest` as their tag (stablelm-zephyr) while others use `:3b` or `:2b`. Check `ollama list` to find the correct tag. granite4.1, orca-mini, and starcoder2 use `:3b` not `:latest`.

## Benchmark Scripts

### General 5-prompt suite

- `multiprompt_bench.py` — original 12-model harness (227 lines)
- `rebench_fixed.py` — re-benchmark specific models with `--jinja` fix
- `bench_new_models.py` — standalone benchmark for 4 new models
- `bench_new3.py` — benchmark 2 newest models (gemma3:1b, qwen2.5-coder:3b) on both suites

### Coding 5-prompt suite

- `bench_coding.py` — 5 coding prompts, 16+ models, banner stripping, `--jinja` (245 lines)

### Quality scoring

- `quality_scoring.py` — general 5-prompt quality scores (original, uses `{"scores": {...}}` format)
- `score_coding.py` — coding quality scores

### PDF builders

- `build_multiprompt_pdf.py` — general benchmark PDF (8 pages, landscape A4)
- `build_coding_pdf.py` — coding benchmark PDF (5 pages, landscape A4)
- `build_guide_pdf.py` — model selection guide PDF (6 pages)

**Important:** Both PDF builders now support both flat `{model: {prompt: {score}}}` and nested `{"scores": {model: ...}}` JSON formats. The `get_quality()` and `get_metric()` functions check for `"scores"` key and fall back to flat format.

## JSON Data Files

### multiprompt_results.json

Structure: `{"prompts": {id: label}, "models": {model: {prompt: {gen_tps, prompt_tps, tokens_generated, wall_time_s, output_preview}}}}`

### coding_benchmark_results.json

Same structure as above but with coding prompts (html, python, c, basic, julia).

### quality_scores.json (general)

**Format: flat** `{model: {prompt_id: {score: int, notes: str}}}`

Prompt IDs: `code`, `iambic`, `prose`, `creative`, `math`

### coding_quality_scores.json

**Format: flat** `{model: {prompt_id: {score: int, notes: str}}}`

Prompt IDs: `html`, `python`, `c`, `basic`, `julia`

**Warning:** Previous versions of scoring scripts wrote these files with extra keys (`"scores"`, `"summary"`, `"metrics"`) wrapping the actual data. These junk keys break PDF builders and must be removed. The PDF builders have been patched to handle both formats, but always use flat format for new files.

## Banner Stripping

llama-cli outputs a startup banner with build info, model metadata, and system prompt. This must be stripped before scoring quality. The `strip_banner()` function in each bench script handles this:

1. Remove `Loading model...` progress lines
2. Remove build/model/ftype/modality metadata lines
3. Remove `available commands:` block
4. Remove `system:` block
5. Find the `>` prompt marker and keep only text after it
6. Remove `<|im_start|>` / `<|im_end|>` chat template tokens

## OOM Troubleshooting

| Model Size | Context | RAM Needed | Fits with GUI? |
|---|---|---|---|
| 1B Q4 | 4096 | ~1.0 GB | Yes |
| 2B Q4 | 4096 | ~1.5 GB | Yes |
| 3B Q4 | 4096 | ~2.0 GB | Yes |
| 3.4B Q4 | 4096 | ~3.5 GB | Marginal |
| 4B+ Q4 | 4096 | ~4.5 GB | No (stop gdm3) |
| 4B+ Q4 | 2048 | ~3.5 GB | Yes (reduced context) |

**If a model OOMs with "the server exited before becoming ready":**
1. Stop GUI: `sudo systemctl stop gdm3`
2. Kill Ollama: `pkill -f "ollama serve"`
3. Reduce context: `-c 2048` instead of `-c 4096`
4. Check free RAM: `free -h` (need >1GB free after model load)

## Git Workflow

```bash
# Commit
cd ~/projects/jetson-llm-benchmark
git add -A
git commit -m "Descriptive message"

# Push (SSH key required)
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519_hermes -o IdentitiesOnly=yes" \
  git push origin main
```

- Default branch: `main` (not `master`)
- GitHub: `drwjkirkpatrick-web/jetson-llm-benchmark`
- SSH key: `~/.ssh/id_ed25519_hermes`
- `gh auth` token may be expired; use SSH for push, do not rely on `gh` CLI for repo edits.

## Step-by-Step: Adding a New Model

1. Download the model (Ollama pull or direct GGUF from HuggingFace)
2. Symlink to `~/models/bench-gguf/`:
   ```bash
   BLOB=$(python3 -c "import json; m=json.load(open('$HOME/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>')); print(m['layers'][0]['digest'].split(':')[1])")
   ln -sf ~/.ollama/models/blobs/sha256-$BLOB ~/models/bench-gguf/<name>.gguf
   ```
3. Run llama-bench to verify it loads:
   ```bash
   ~/llama.cpp/build/bin/llama-bench -m ~/models/bench-gguf/<name>.gguf -ngl 99 -fa 1 -p 512 -n 128
   ```
4. If it fails with tensor count errors, download clean GGUF from `huggingface.co/ggml-org`
5. If it OOMs, reduce context to `-c 2048`
6. Run the benchmark script (modify REBENCH list or write a new one)
7. Quality-score the outputs (1-10 per prompt)
8. Update the PDF builder's MODELS list
9. Rebuild both PDFs
10. Commit and push

## Step-by-Step: Rebuilding PDFs After Quality Score Updates

1. Update `quality_scores.json` (general) and/or `coding_quality_scores.json` (coding) with new scores
2. Ensure JSON is flat format: `{model: {prompt: {score, notes}}}`
3. Rebuild:
   ```bash
   cd ~/projects/jetson-llm-benchmark
   python3 build_multiprompt_pdf.py
   python3 build_coding_pdf.py
   ```
4. Verify all models appear:
   ```python
   from pypdf import PdfReader
   r = PdfReader('Multi_Prompt_Benchmark_Report.pdf')
   text = ''.join(p.extract_text() for p in r.pages)
   # Check model names appear
   ```
5. Commit and push

## Quality Scoring Methodology

Scores are 1-10 based on:

- **Correctness**: Does the output solve the requested task?
- **Completeness**: Is the output complete or truncated?
- **Language accuracy**: Does code use correct syntax for the target language?
- **Instruction following**: Does the model follow all parts of the prompt?
- **Creativity/Style**: For creative tasks, is the output engaging and well-written?

Typical score ranges:
- 9-10: Excellent, complete, correct, could ship as-is
- 7-8: Good, mostly correct, minor issues
- 5-6: Acceptable but incomplete or with errors
- 3-4: Poor, major issues, mostly wrong
- 1-2: Refused, echoed prompt, or completely wrong

## Known Issues and Fixes Summary

| Issue | Symptom | Fix |
|---|---|---|
| Gemma 4 E2B Ollama blob | `wrong number of tensors; expected 2012, got 601` | Download clean GGUF from huggingface.co/ggml-org |
| Gemma 4 E2B OOM in coding bench | `the server exited before becoming ready` | Reduce context to `-c 2048` |
| gemma3n:e2b | Same tensor count error as Gemma 4 | Download clean text-only GGUF (when available) |
| SmallThinker 49 tokens | Model outputs only 49 tokens | Use clean GGUF from ggml-org + `--jinja` + `-n 2000` |
| Thinking models low output | Fewer than 50 tokens | Add `--jinja` flag (enables chat template processing) |
| ministral-3:latest | Wrong tensor count (multimodal) | No fix; skip model |
| qwen3.5:2b | `rope.dimension_sections wrong array length` | No fix; skip model |
| StarCoder2 low scores | Echoes prompts, no instruction following | Expected: base model, not instruction-tuned |
| CodeGemma low scores | Echoes prompts, code completion behavior | Expected: code completion model, not chat-tuned |
| Orca-Mini refuses tasks | "I cannot create HTML" | No fix; model is unsuitable for these tasks |
| Junk keys in quality JSON | `{"scores": ..., "summary": ...}` | Use flat format; PDF builders patched to handle both |
| `gh auth` 401 | Cannot update repo description | Token expired; use SSH for git push instead |
| SSH to Jetson by hostname | `ssh jetson` fails | Use IP address or work locally |
| PYTHONPATH polluted | pip install skips packages | Use `.venv/bin/python -m pip install --ignore-installed` |