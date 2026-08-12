# Results

This directory stores benchmark JSON output files.

Each file is named `benchmark-YYYYMMDD.json` and contains:

- `metadata`: timestamp, device info, context length
- `results`: array of per-model results with tok/s, RAM, CMA, temperature

## Interpreting Results

| Field | Meaning |
|-------|---------|
| `tok_per_sec` | Generation throughput — higher is better |
| `prompt_tok_per_sec` | Prompt processing speed |
| `load_time_s` | Time to load model into GPU memory |
| `ram_after_mb` | Available system RAM after inference |
| `cma_after_mb` | Free contiguous GPU memory — critical on Jetson |
| `temp_after_c` | GPU temperature after inference |

## What Makes a Good Result

- **tok/s > 10**: Usable for interactive use
- **tok/s 5–10**: Acceptable for background/draft generation
- **tok/s < 5**: Too slow for most workflows
- **cma_after_mb > 50**: Safe headroom for next model load
- **temp_after_c < 75°C**: No thermal throttling