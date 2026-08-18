#!/usr/bin/env python3
"""
Multi-prompt benchmark for Jetson LLM models.
Tests each model with 5 prompt styles: code, iambic pentameter, prose, creative writing, math proof.
Collects generation tok/s, prompt tok/s, and output quality metrics.
"""

import subprocess
import re
import json
import sys
import os
import time

MODELS_DIR = os.path.expanduser("~/models/bench-gguf")
LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
RESULTS_FILE = os.path.expanduser("~/projects/jetson-llm-benchmark/multiprompt_results.json")

PROMPTS = [
    {
        "id": "code",
        "name": "Code Generation",
        "prompt": "Write a Python function called `merge_sort` that takes a list of integers and returns a new list sorted in ascending order. Include type hints, a docstring, and handle edge cases (empty list, single element). Then show an example usage.",
        "max_tokens": 400,
    },
    {
        "id": "iambic",
        "name": "Iambic Pentameter",
        "prompt": "Write a poem about the changing of seasons from autumn to winter, strictly in iambic pentameter (10 syllables per line, alternating stress). Write exactly 8 lines with an ABAB CDCD rhyme scheme.",
        "max_tokens": 300,
    },
    {
        "id": "prose",
        "name": "Clinical Prose",
        "prompt": "Explain the pathophysiology of Hashimoto's thyroiditis in detail. Cover the autoimmune mechanism, the role of anti-TPO antibodies, the progression from euthyroid to hypothyroid, and the typical lab findings at each stage. Write for a medical student audience.",
        "max_tokens": 400,
    },
    {
        "id": "creative",
        "name": "Creative Writing",
        "prompt": "Write a short scene (200-300 words) set in a lighthouse during a storm. The lighthouse keeper is an old woman who has been alone for thirty years. A stranger arrives at the door, drenched and shivering. Show, don't tell — use sensory details and subtext.",
        "max_tokens": 400,
    },
    {
        "id": "math",
        "name": "Mathematical Proof",
        "prompt": "Prove that the square root of 2 is irrational. Use a proof by contradiction. State each step clearly with justification. Begin with: 'Theorem: sqrt(2) is irrational.'",
        "max_tokens": 400,
    },
]

# Models to benchmark — must have GGUF symlinks in ~/models/bench-gguf/
MODELS = [
    ("codegemma:2b",      "codegemma-2b.gguf",         "--no-conversation --no-display-prompt"),
    ("gemma2:2b",         "gemma2-2b.gguf",            "--no-conversation --no-display-prompt"),
    ("gemma4 E2B",        "gemma-4-E2B-it-Q4_0.gguf",  "--no-conversation --no-display-prompt --jinja"),
    ("lfm2.5:2.6b",       "lfm2.5-2.6b-q4km.gguf",     "--no-conversation --no-display-prompt"),
    ("qwen2.5:3b",        "qwen2.5-3b.gguf",           "--no-conversation --no-display-prompt"),
    ("hermes3:3b",        "hermes3-3b.gguf",           "--no-conversation --no-display-prompt"),
    ("llama3.2:3b",       "llama3.2-3b.gguf",          "--no-conversation --no-display-prompt"),
    ("phi3:3.8b",         "phi3-3.8b.gguf",            "--no-conversation --no-display-prompt"),
]


def run_benchmark(model_name, gguf_file, extra_flags, prompt_data):
    """Run a single benchmark and parse results."""
    model_path = os.path.join(MODELS_DIR, gguf_file)
    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}

    cmd = [
        LLAMA_CLI,
        "-m", model_path,
        "-p", prompt_data["prompt"],
        "-n", str(prompt_data["max_tokens"]),
        "-c", "4096",
        "--temp", "0.3",
        "-ngl", "99",
        "-fa", "on",
        "-st",
    ]
    # Add extra flags
    for flag in extra_flags.split():
        cmd.append(flag)

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"error": "Timeout (300s)"}
    except Exception as e:
        return {"error": str(e)}

    wall_time = time.time() - start_time
    output = result.stdout + result.stderr

    # Parse timing stats from -st output
    # Format: [ Prompt: 60.4 t/s | Generation: 26.7 t/s ]
    gen_tps = None
    prompt_tps = None
    tokens_generated = 0

    # Try multiple patterns
    gen_match = re.search(r"Generation:\s*([\d.]+)\s*t/s", output)
    prompt_match = re.search(r"Prompt:\s*([\d.]+)\s*t/s", output)

    if gen_match:
        gen_tps = float(gen_match.group(1))
    if prompt_match:
        prompt_tps = float(prompt_match.group(1))

    # Count actual generated tokens (lines after prompt, before timing)
    # Extract the generated text between the prompt and the timing line
    text_lines = []
    capturing = False
    for line in output.split("\n"):
        if "Prompt:" in line and "Generation:" in line:
            break
        if capturing:
            text_lines.append(line)
        # Start capturing after we see the prompt echoed (or after model loading)
        if prompt_data["prompt"][:50] in line:
            capturing = True
        elif "system_info" in line.lower() or "main: interactive" in line.lower():
            capturing = True

    generated_text = "\n".join(text_lines).strip()
    tokens_generated = len(generated_text.split())

    return {
        "gen_tps": gen_tps,
        "prompt_tps": prompt_tps,
        "wall_time_s": round(wall_time, 1),
        "tokens_generated": tokens_generated,
        "output_preview": generated_text[:500],
        "raw_tail": output[-300:] if not gen_tps else "",
    }


def main():
    results = {}
    total_runs = len(MODELS) * len(PROMPTS)
    run_num = 0

    # Load existing results if any (for resuming)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results.get('models', {}))} existing model results")

    if "models" not in results:
        results["models"] = {}
    if "prompts" not in results:
        results["prompts"] = {p["id"]: p["name"] for p in PROMPTS}

    for model_name, gguf_file, extra_flags in MODELS:
        if model_name not in results["models"]:
            results["models"][model_name] = {}

        for prompt_data in PROMPTS:
            pid = prompt_data["id"]
            run_num += 1

            # Skip if already done (resume support)
            if pid in results["models"][model_name] and "error" not in results["models"][model_name][pid]:
                print(f"[{run_num}/{total_runs}] SKIP {model_name} / {pid} (already done)")
                continue

            print(f"[{run_num}/{total_runs}] BENCH {model_name} / {pid} ({prompt_data['name']})...")
            sys.stdout.flush()

            res = run_benchmark(model_name, gguf_file, extra_flags, prompt_data)
            results["models"][model_name][pid] = res

            if "error" in res:
                print(f"  -> ERROR: {res['error']}")
            else:
                print(f"  -> gen={res['gen_tps']} t/s, prompt={res['prompt_tps']} t/s, {res['wall_time_s']}s wall, {res['tokens_generated']} tokens")

            # Save after each run (crash recovery)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 120)
    print("MULTI-PROMPT BENCHMARK RESULTS")
    print("=" * 120)

    # Header
    prompt_ids = [p["id"] for p in PROMPTS]
    header = f"{'Model':<20}"
    for pid in prompt_ids:
        header += f" | {pid:>10}"
    header += f" | {'AVERAGE':>10}"
    print(header)
    print("-" * len(header))

    for model_name, gguf_file, _ in MODELS:
        if model_name not in results["models"]:
            continue
        row = f"{model_name:<20}"
        speeds = []
        for pid in prompt_ids:
            data = results["models"][model_name].get(pid, {})
            if "gen_tps" in data and data["gen_tps"]:
                row += f" | {data['gen_tps']:>10.1f}"
                speeds.append(data["gen_tps"])
            else:
                row += f" | {'ERR':>10}"
        if speeds:
            avg = sum(speeds) / len(speeds)
            row += f" | {avg:>10.1f}"
        else:
            row += f" | {'N/A':>10}"
        print(row)

    print("=" * len(header))
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()