#!/usr/bin/env python3
"""
Benchmark DeepSeek R1-Distill models on both suites.
All are thinking models: --jinja + -n 2000.
7B and 8B use -c 2048 (tight RAM), 1.5B uses -c 4096.
"""

import json, os, re, subprocess, time, sys

LLAMA_CLI = os.path.expanduser('~/llama.cpp/build/bin/llama-cli')
BENCH_DIR = os.path.expanduser('~/models/bench-gguf')
RESULTS_DIR = os.path.expanduser('~/projects/jetson-llm-benchmark')

# ── Models ──────────────────────────────────────────────────────────────────
MODELS = [
    {
        'name': 'deepseek-r1-qwen-1.5b',
        'path': os.path.expanduser('~/models/deepseek/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf'),
        'context': 4096,
        'thinking': True,
        'label': 'DeepSeek R1 Distill Qwen 1.5B',
    },
    {
        'name': 'deepseek-r1-qwen-7b',
        'path': os.path.expanduser('~/models/deepseek/DeepSeek-R1-Distill-Qwen-7B-Q2_K.gguf'),
        'context': 2048,
        'thinking': True,
        'label': 'DeepSeek R1 Distill Qwen 7B',
    },
    {
        'name': 'deepseek-r1-llama-8b',
        'path': os.path.expanduser('~/models/deepseek/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf'),
        'context': 2048,
        'thinking': True,
        'label': 'DeepSeek R1 Distill Llama 8B',
    },
]

# ── General 5-Prompt Suite ───────────────────────────────────────────────────
GENERAL_PROMPTS = {
    'code': 'Write a Python function called `merge_sorted_lists` that takes two sorted lists of integers and returns a single merged sorted list. Include type hints, a docstring, and handle edge cases like empty lists.',
    'iambic': 'Write a 14-line poem in strict iambic pentameter about the passage of seasons. Each line must have exactly ten syllables with the stress pattern: da-DUM da-DUM da-DUM da-DUM da-DUM.',
    'prose': 'Write a concise clinical case writeup (approximately 200 words) for a 45-year-old female presenting with fatigue, weight gain, cold intolerance, and brittle nails. Include history, physical exam findings, assessment, and naturopathic treatment plan.',
    'creative': 'Write a short story (300-500 words) about an old lighthouse keeper who discovers a message in a bottle that changes everything. Use vivid sensory details about the sea, the light, and the keeper\'s isolation.',
    'math': 'Prove that for any positive integer n, the sum 1 + 2 + 3 + ... + n equals n(n+1)/2. Provide a clear, step-by-step mathematical proof with proper notation.',
}

# ── Coding 5-Prompt Suite ────────────────────────────────────────────────────
CODING_PROMPTS = {
    'html': 'Create a complete HTML page for a personal portfolio website. Include a header with navigation, a hero section with name and tagline, an about section, a projects section with 3 sample project cards using flexbox, and a footer. Add inline CSS with a modern color scheme, responsive design with a media query for mobile, and Google Fonts link.',
    'python': 'Write a Python class called `DataProcessor` that reads a CSV file, filters rows based on a column condition, computes summary statistics (mean, median, std dev) for numeric columns, and exports results to JSON. Include type hints, docstrings, proper error handling with try/except, and a `__main__` block with example usage.',
    'c': 'Write a complete C program that implements a thread-safe queue using a linked list. Include: a struct for queue nodes, a struct for the queue with a pthread mutex, functions for enqueue, dequeue, peek, size, and free_queue. Add a main function that demonstrates usage with 3 producer threads and 1 consumer thread.',
    'basic': 'Write a complete TRS-80 Level II BASIC program that implements a simple text adventure game. Use line numbers (starting at 10, incrementing by 10). The game should have at least 3 rooms with descriptions, items the player can pick up with a GET command, and a goal to find a treasure. Use INPUT for player commands, PRINT for output, and GOTO for room navigation. Include CLEAR to reset the screen.',
    'julia': 'Write a Julia function called `newton_raphson` that finds the root of a function using the Newton-Raphson method. Include: type annotations for all arguments, a docstring with usage examples, proper error handling for non-convergence, a maximum iteration parameter with default value, and a return type annotation. The function should take f (the function), f_prime (its derivative), x0 (initial guess), and optional parameters for tolerance and max iterations.',
}

# ── Banner Stripping ────────────────────────────────────────────────────────
def strip_banner(raw_output):
    """Remove llama.cpp startup banner and metadata, keep only generated text."""
    lines = raw_output.split('\n')
    # Find the first line that's just ">" or starts with the actual output
    output_started = False
    clean_lines = []
    for line in lines:
        # Skip known banner patterns
        if any(line.strip().startswith(p) for p in [
            'build:', 'system:', 'main:', 'llama_model_loader:', 'llama_model:',
            'load_backend:', 'load_tensors:', 'model_loader:', 'llama_kv_cache:',
            'llama_compute:', 'llama_new', 'llama_init:', 'llama_print:',
            'llama_backend', 'ggml_', '| ', 'sampling:', 'print_info:',
            'available', 'XXX:', '___', '===', '---', 'load_',
            'type_', 'n_ctx', 'n_batch', 'n_ubatch', 'n_seq', 'n_thread',
            'n_gpu', 'n_mmap', 'llama_decode', 'session:',
            'clip_', 'image:', 'filename:', 'init_', 'device:',
            'CUDA', 'cuBLAS', 'ggml_cuda', 'BLAS', 'Neo', 'NPU',
        ]):
            continue
        # Skip empty lines before output starts
        if not output_started and not line.strip():
            continue
        # Skip chat template tokens
        if '<|im_start|>' in line or '<|im_end|>' in line:
            continue
        if '<|begin_of_text|>' in line or '<|end_of_text|>' in line:
            continue
        # Mark start of actual output (first substantive line after banner)
        if not output_started and line.strip():
            output_started = True
        if output_started:
            clean_lines.append(line)
    return '\n'.join(clean_lines).strip()

# ── Run a single prompt ──────────────────────────────────────────────────────
def run_prompt(model, prompt_id, prompt_text, suite_name):
    """Run a single prompt against a model and capture results."""
    cmd = [
        LLAMA_CLI,
        '-m', model['path'],
        '-p', prompt_text,
        '-n', '2000',
        '-c', str(model['context']),
        '--temp', '0.3',
        '-ngl', '99',
        '-fa', 'on',
        '--no-conversation',
        '--no-display-prompt',
        '-st',
    ]
    if model.get('thinking'):
        cmd.append('--jinja')

    print(f"\n{'='*60}")
    print(f"  {suite_name}: {model['name']} / {prompt_id}")
    print(f"  ctx={model['context']} thinking={model.get('thinking', False)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    wall_time = time.time() - t0

    # Parse stderr for timing stats
    stderr = result.stderr
    stdout = result.stdout

    # Extract tokens per second from stderr
    gen_tps = 0.0
    prompt_tps = 0.0
    tokens_generated = 0

    # Look for: "        2000 ->  85.23 tokens/sec" or similar
    for line in stderr.split('\n'):
        # Generation speed: "n_eval = 2000, ... tokens/sec"
        m = re.search(r'(\d+\.\d+)\s*tokens/sec', line)
        if m:
            tps = float(m.group(1))
            # First match is usually prompt eval, second is generation
            if 'prompt' in line.lower() or 'pp' in line.lower():
                prompt_tps = tps
            else:
                gen_tps = tps

    # Also check for the llama-cli specific format
    if gen_tps == 0.0:
        m = re.search(r'eval time.*=\s*(\d+\.\d+)\s*ms', stderr)
        if m:
            eval_ms = float(m.group(1))
            # tokens = 2000, time in ms
            # Actually look for n_eval
            m2 = re.search(r'n_eval\s*=\s*(\d+)', stderr)
            if m2:
                n_eval = int(m2.group(1))
                tokens_generated = n_eval
                if eval_ms > 0:
                    gen_tps = n_eval / (eval_ms / 1000.0)

    # Try to get tokens from output length
    clean_output = strip_banner(stdout)
    if tokens_generated == 0:
        # Rough estimate from output
        tokens_generated = len(clean_output.split())

    # Get prompt eval stats
    m = re.search(r'prompt eval time.*=\s*(\d+\.\d+)\s*ms', stderr)
    if m:
        pp_ms = float(m.group(1))
        m2 = re.search(r'n_prompt\s*=\s*(\d+)', stderr)
        if m2:
            n_prompt = int(m2.group(1))
            if pp_ms > 0:
                prompt_tps = n_prompt / (pp_ms / 1000.0)

    print(f"  gen_tps={gen_tps:.1f} prompt_tps={prompt_tps:.1f} tokens={tokens_generated} wall={wall_time:.1f}s")
    print(f"  Output preview: {clean_output[:200]}...")

    return {
        'gen_tps': round(gen_tps, 2),
        'prompt_tps': round(prompt_tps, 2),
        'tokens_generated': tokens_generated,
        'wall_time_s': round(wall_time, 1),
        'output_preview': clean_output[:500],
        'full_output': clean_output[:2000],
    }

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Verify model files exist
    for m in MODELS:
        if not os.path.exists(m['path']):
            print(f"ERROR: Model file not found: {m['path']}")
            sys.exit(1)
        size_gb = os.path.getsize(m['path']) / (1024**3)
        print(f"  {m['name']}: {size_gb:.2f} GB, ctx={m['context']}")

    # Symlink to bench-gguf for consistency
    for m in MODELS:
        link_path = os.path.join(BENCH_DIR, f"{m['name']}.gguf")
        if os.path.exists(link_path) or os.path.islink(link_path):
            os.remove(link_path)
        os.symlink(m['path'], link_path)
        print(f"  Symlinked: {link_path} -> {m['path']}")

    general_results = {}
    coding_results = {}

    for model in MODELS:
        key = model['name']
        general_results[key] = {}
        coding_results[key] = {}

        # General suite
        for pid, prompt_text in GENERAL_PROMPTS.items():
            r = run_prompt(model, pid, prompt_text, 'GENERAL')
            general_results[key][pid] = r
            # Save incrementally
            with open(os.path.join(RESULTS_DIR, 'deepseek_general_results.json'), 'w') as f:
                json.dump({'prompts': GENERAL_PROMPTS, 'models': general_results}, f, indent=2)

        # Coding suite
        for pid, prompt_text in CODING_PROMPTS.items():
            r = run_prompt(model, pid, prompt_text, 'CODING')
            coding_results[key][pid] = r
            with open(os.path.join(RESULTS_DIR, 'deepseek_coding_results.json'), 'w') as f:
                json.dump({'prompts': CODING_PROMPTS, 'models': coding_results}, f, indent=2)

    print("\n\n=== ALL DONE ===")
    print(f"General results: {os.path.join(RESULTS_DIR, 'deepseek_general_results.json')}")
    print(f"Coding results: {os.path.join(RESULTS_DIR, 'deepseek_coding_results.json')}")


if __name__ == '__main__':
    main()