#!/usr/bin/env python3
"""
Coding-focused benchmark: 5 language-specific prompts to test code generation.
Tests HTML, Python, C, TRS-80 BASIC, and Julia across all models.
Uses the fixed harness: --jinja, 2000 tokens, banner stripping.
"""
import json, os, re, subprocess, time

RESULTS = os.path.expanduser('~/projects/jetson-llm-benchmark/coding_benchmark_results.json')
LLAMA_CLI = os.path.expanduser('~/llama.cpp/build/bin/llama-cli')
MODELS_DIR = os.path.expanduser('~/models/bench-gguf')

# 5 coding prompts covering different languages and paradigms
PROMPTS = {
    'html': (
        'HTML/CSS Web Page',
        'Create a complete HTML page with embedded CSS for a personal portfolio website. '
        'Include a header with navigation, a hero section with name and tagline, '
        'an about section, a projects grid with 3 sample project cards, '
        'and a footer with contact links. Use modern CSS with flexbox, '
        'a color scheme of dark blue and gold, and responsive design with a media query for mobile.'
    ),
    'python': (
        'Python Data Processing',
        'Write a Python class called DataProcessor that loads a CSV file of employee records '
        '(columns: name, department, salary, hire_date), filters employees hired after 2020, '
        'calculates average salary by department, finds the top 3 highest paid employees, '
        'and exports results to a JSON file. Include type hints, docstrings, error handling, '
        'and use the csv and json standard library modules. Show example usage.'
    ),
    'c': (
        'C System Programming',
        'Write a C program that implements a simple thread-safe queue using a linked list. '
        'Include functions: queue_init, queue_push, queue_pop, queue_size, and queue_destroy. '
        'Use pthread mutex for thread safety. Include a main function that demonstrates '
        'creating the queue, pushing 5 integers, and popping them all. '
        'Add proper error handling and memory management (free on destroy).'
    ),
    'basic': (
        'TRS-80 BASIC Retro Game',
        'Write a TRS-80 Model III BASIC program for a simple text adventure game. '
        'The player explores 3 rooms (Entrance Hall, Library, Treasure Room) looking for a golden key. '
        'Use PRINT for room descriptions, INPUT for player commands (GO NORTH, GO SOUTH, TAKE KEY, EXAMINE), '
        'string variables for room descriptions, and GOTO for navigation. '
        'Include inventory tracking and a win condition when the player takes the key '
        'and reaches the Treasure Room. Number lines starting at 10 with increments of 10.'
    ),
    'julia': (
        'Julia Numerical Computing',
        'Write a Julia program that implements the Newton-Raphson method for finding roots '
        'of a function. Create a function newton_raphson(f, df, x0, tol, max_iter) that takes '
        'a function, its derivative, initial guess, tolerance, and max iterations. '
        'Use it to find the root of x^3 - 2x - 5 (starting at x0=2.0). '
        'Print each iteration showing the step number, current x, f(x), and the error. '
        'Include type annotations and a docstring. Show the final result.'
    ),
}

# All 16 models
MODELS = [
    ('codegemma:2b',       'codegemma-2b.gguf'),
    ('granite3-dense:2b',  'granite3-dense-2b.gguf'),
    ('granite3.2:2b',      'granite3.2-2b.gguf'),
    ('gemma4 E2B',         'gemma4-e2b.gguf'),
    ('gemma2:2b',          'gemma2-2b.gguf'),
    ('lfm2.5:2.6b',        'lfm2.5-2.6b-q4km.gguf'),
    ('qwen2.5:3b',         'qwen2.5-3b.gguf'),
    ('hermes3:3b',         'hermes3-3b.gguf'),
    ('llama3.2:3b',        'llama3.2-3b.gguf'),
    ('granite4:3b',        'granite4-3b.gguf'),
    ('granite4.1:3b',      'granite4.1-3b.gguf'),
    ('phi3:3.8b',          'phi3-3.8b.gguf'),
    ('stablelm-zephyr',    'stablelm-zephyr.gguf'),
    ('smallthinker:3b',    'smallthinker-3b.gguf'),
    ('orca-mini:3b',       'orca-mini-3b.gguf'),
    ('starcoder2:3b',      'starcoder2-3b.gguf'),
]

def strip_banner(text):
    """Remove the llama.cpp startup banner and UI chrome from output."""
    text = re.sub(r'Loading model\.\.\..*?(?=\n\n|\n[▄█])', '', text, flags=re.DOTALL)
    text = re.sub(r'▄▄ ▄▄.*?(?=build|available|>)', '', text, flags=re.DOTALL)
    text = re.sub(r'^build\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^model\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^ftype\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^modalities\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^available commands:.*?(?=\n\n|\n>|\Z)', '', text, flags=re.DOTALL | re.MULTILINE)
    # Remove system prompt
    text = re.sub(r'^system\s*:.*?(?=\n\n|\nuser|\Z)', '', text, flags=re.DOTALL | re.MULTILINE)

    lines = text.split('\n')
    prompt_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('> ') or stripped == '>':
            prompt_idx = i
    if prompt_idx >= 0:
        lines = lines[prompt_idx + 1:]

    text = '\n'.join(lines)
    text = text.replace('<|im_start|>', '').replace('<|im_end|>', '')
    text = text.replace('<|channel|>', '').replace('<|start_header_id|>', '').replace('<|end_header_id|>', '')
    text = text.strip()
    return text

def parse_stats(raw_output):
    """Parse gen_tps, prompt_tps, tokens from llama-cli -st output."""
    # Format: [ Prompt: 92.5 t/s | Generation: 24.5 t/s ]
    match = re.search(r'Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s', raw_output)
    if match:
        prompt_tps = float(match.group(1))
        gen_tps = float(match.group(2))
    else:
        prompt_tps = 0
        gen_tps = 0

    # Count generated tokens: look for "n_eval = N" or count from generation time
    tok_match = re.search(r'n_eval\s*=\s*(\d+)', raw_output)
    if tok_match:
        tokens = int(tok_match.group(1))
    else:
        # Estimate from gen_tps and wall time
        tokens = 0

    return gen_tps, prompt_tps, tokens

def run_benchmark(model_key, gguf_file, prompt_id, prompt_text):
    model_path = os.path.join(MODELS_DIR, gguf_file)
    if not os.path.exists(model_path):
        print(f"  SKIP {model_key} - model file not found")
        return None

    cmd = [
        LLAMA_CLI,
        '-m', model_path,
        '-p', prompt_text,
        '-n', '2000',
        '-c', '4096',
        '--temp', '0.3',
        '-ngl', '99',
        '-fa', 'on',
        '--no-conversation',
        '--no-display-prompt',
        '--jinja',
        '-st',
    ]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        wall_time = time.time() - start
        raw = result.stdout + result.stderr

        gen_tps, prompt_tps, tokens = parse_stats(raw)

        # If tokens not found from n_eval, estimate from gen_tps * wall_time
        if tokens == 0 and gen_tps > 0:
            # Try to find token count from output
            tok_match2 = re.search(r'(\d+)\s*tokens?\s*', raw)
            if tok_match2:
                tokens = int(tok_match2.group(1))
            else:
                tokens = int(gen_tps * (wall_time - 1))  # Rough estimate

        # Strip banner and extract clean output
        clean = strip_banner(raw)

        # Also try to count tokens from the clean output length
        if tokens == 0:
            tokens = len(clean.split())

        return {
            'gen_tps': round(gen_tps, 1),
            'prompt_tps': round(prompt_tps, 1),
            'tokens_generated': tokens,
            'wall_time_s': round(wall_time, 1),
            'output_preview': clean[:500],
        }
    except subprocess.TimeoutExpired:
        return {'gen_tps': 0, 'prompt_tps': 0, 'tokens_generated': 0, 'wall_time_s': 180, 'output_preview': 'TIMEOUT'}
    except Exception as e:
        return {'gen_tps': 0, 'prompt_tps': 0, 'tokens_generated': 0, 'wall_time_s': 0, 'output_preview': f'ERROR: {e}'}

# Main
data = {'models': {}}

# Load existing results if any
if os.path.exists(RESULTS):
    with open(RESULTS) as f:
        data = json.load(f)

total = len(MODELS) * len(PROMPTS)
done = 0

for model_key, gguf_file in MODELS:
    print(f"\n{'='*70}")
    print(f"MODEL: {model_key} ({gguf_file})")
    print(f"{'='*70}")

    if model_key not in data['models']:
        data['models'][model_key] = {}

    for pid, (label, prompt_text) in PROMPTS.items():
        done += 1

        # Skip if already done
        if pid in data['models'].get(model_key, {}):
            print(f"[{done}/{total}] SKIP {model_key} / {pid} (already done)")
            continue

        print(f"[{done}/{total}] {model_key} / {pid}...", end=' ', flush=True)
        result = run_benchmark(model_key, gguf_file, pid, prompt_text)
        if result:
            data['models'][model_key][pid] = result
            preview = result['output_preview'][:100].replace('\n', ' ')
            print(f"-> gen={result['gen_tps']} t/s, {result['tokens_generated']} tokens, {result['wall_time_s']}s")
            print(f"   Preview: {preview}")
        else:
            print("FAILED")

    # Save after each model
    with open(RESULTS, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {model_key}")

# Summary table
print(f"\n{'='*80}")
print("CODING BENCHMARK RESULTS")
print(f"{'='*80}")
print(f"{'Model':<22} {'HTML':>8} {'Python':>8} {'C':>8} {'BASIC':>8} {'Julia':>8} {'Avg':>8}")
print(f"{'-'*80}")
for model_key, _ in MODELS:
    m = data['models'].get(model_key, {})
    speeds = []
    row = f"{model_key:<22}"
    for pid in ['html', 'python', 'c', 'basic', 'julia']:
        r = m.get(pid, {})
        gen = r.get('gen_tps', 0)
        speeds.append(gen)
        row += f" {gen:>7.1f}t"
    avg = sum(speeds) / len(speeds) if speeds else 0
    row += f" {avg:>7.1f}t"
    print(row)

print(f"\nResults saved to {RESULTS}")