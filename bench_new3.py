#!/usr/bin/env python3
"""
Benchmark new models on both the general 5-prompt and coding 5-prompt suites.
Uses the fixed harness: --jinja, 2000 tokens, banner stripping.
Merges results into the existing JSON files.
"""
import json, os, re, subprocess, time

LLAMA_CLI = os.path.expanduser('~/llama.cpp/build/bin/llama-cli')
MODELS_DIR = os.path.expanduser('~/models/bench-gguf')

NEW_MODELS = [
    ('gemma3:1b',         'gemma3-1b.gguf'),
    ('qwen2.5-coder:3b',  'qwen2.5-coder-3b.gguf'),
]

# General 5-prompt suite (same as rebench_fixed.py)
GENERAL_PROMPTS = {
    'code': 'Write a Python function called "calculate_bmi" that takes weight in kilograms and height in meters, calculates BMI, returns the value rounded to 1 decimal place, and includes a docstring with type hints.',
    'iambic': 'Write a poem of exactly 14 lines in iambic pentameter about the changing seasons. Each line must have exactly 10 syllables with the stress pattern da-DUM da-DUM da-DUM da-DUM da-DUM.',
    'prose': 'Write a 3-paragraph clinical case discussion about a 45-year-old patient presenting with chronic fatigue, considering differential diagnosis, lab workup, and treatment approach. Use professional medical language.',
    'creative': 'Write a short story (300-500 words) about a lighthouse keeper who discovers a message in a bottle that changes everything. Include vivid sensory details and an unexpected twist.',
    'math': 'Prove that for any positive integers a and b, the square of (a + b) equals a squared plus 2ab plus b squared. Write a clear, formal proof with each step justified.',
}

# Coding 5-prompt suite (same as bench_coding.py)
CODING_PROMPTS = {
    'html': (
        'Create a complete HTML page with embedded CSS for a personal portfolio website. '
        'Include a header with navigation, a hero section with name and tagline, '
        'an about section, a projects grid with 3 sample project cards, '
        'and a footer with contact links. Use modern CSS with flexbox, '
        'a color scheme of dark blue and gold, and responsive design with a media query for mobile.'
    ),
    'python': (
        'Write a Python class called DataProcessor that loads a CSV file of employee records '
        '(columns: name, department, salary, hire_date), filters employees hired after 2020, '
        'calculates average salary by department, finds the top 3 highest paid employees, '
        'and exports results to a JSON file. Include type hints, docstrings, error handling, '
        'and use the csv and json standard library modules. Show example usage.'
    ),
    'c': (
        'Write a C program that implements a simple thread-safe queue using a linked list. '
        'Include functions: queue_init, queue_push, queue_pop, queue_size, and queue_destroy. '
        'Use pthread mutex for thread safety. Include a main function that demonstrates '
        'creating the queue, pushing 5 integers, and popping them all. '
        'Add proper error handling and memory management (free on destroy).'
    ),
    'basic': (
        'Write a TRS-80 Model III BASIC program for a simple text adventure game. '
        'The player explores 3 rooms (Entrance Hall, Library, Treasure Room) looking for a golden key. '
        'Use PRINT for room descriptions, INPUT for player commands (GO NORTH, GO SOUTH, TAKE KEY, EXAMINE), '
        'string variables for room descriptions, and GOTO for navigation. '
        'Include inventory tracking and a win condition when the player takes the key '
        'and reaches the Treasure Room. Number lines starting at 10 with increments of 10.'
    ),
    'julia': (
        'Write a Julia program that implements the Newton-Raphson method for finding roots '
        'of a function. Create a function newton_raphson(f, df, x0, tol, max_iter) that takes '
        'a function, its derivative, initial guess, tolerance, and max iterations. '
        'Use it to find the root of x^3 - 2x - 5 (starting at x0=2.0). '
        'Print each iteration showing the step number, current x, f(x), and the error. '
        'Include type annotations and a docstring. Show the final result.'
    ),
}

def strip_banner(text):
    """Remove the llama.cpp startup banner and UI chrome from output."""
    text = re.sub(r'Loading model\.\.\..*?(?=\n\n|\n[▄█])', '', text, flags=re.DOTALL)
    text = re.sub(r'▄▄ ▄▄.*?(?=build|available|>)', '', text, flags=re.DOTALL)
    text = re.sub(r'^build\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^model\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^ftype\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^modalities\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^available commands:.*?(?=\n\n|\n>|\Z)', '', text, flags=re.DOTALL | re.MULTILINE)
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
    match = re.search(r'Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s', raw_output)
    if match:
        prompt_tps = float(match.group(1))
        gen_tps = float(match.group(2))
    else:
        prompt_tps = 0
        gen_tps = 0

    tok_match = re.search(r'n_eval\s*=\s*(\d+)', raw_output)
    if tok_match:
        tokens = int(tok_match.group(1))
    else:
        tokens = 0

    return gen_tps, prompt_tps, tokens

def run_one(model_key, gguf_file, prompt_text, timeout=180):
    model_path = os.path.join(MODELS_DIR, gguf_file)
    if not os.path.exists(model_path):
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        wall_time = time.time() - start
        raw = result.stdout + result.stderr

        gen_tps, prompt_tps, tokens = parse_stats(raw)

        if tokens == 0:
            tok_match2 = re.search(r'(\d+)\s*tokens?\s*', raw)
            if tok_match2:
                tokens = int(tok_match2.group(1))
            elif gen_tps > 0:
                tokens = int(gen_tps * (wall_time - 1))

        clean = strip_banner(raw)

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

# ---- Run general 5-prompt suite ----
GENERAL_RESULTS = os.path.expanduser('~/projects/jetson-llm-benchmark/multiprompt_results.json')
with open(GENERAL_RESULTS) as f:
    general_data = json.load(f)

print("=" * 70)
print("GENERAL 5-PROMPT SUITE (code, iambic, prose, creative, math)")
print("=" * 70)

for model_key, gguf_file in NEW_MODELS:
    print(f"\nMODEL: {model_key}")
    if model_key not in general_data['models']:
        general_data['models'][model_key] = {}

    for pid, prompt_text in GENERAL_PROMPTS.items():
        if pid in general_data['models'].get(model_key, {}):
            print(f"  SKIP {pid} (already done)")
            continue
        print(f"  {pid}...", end=' ', flush=True)
        r = run_one(model_key, gguf_file, prompt_text)
        if r:
            general_data['models'][model_key][pid] = r
            preview = r['output_preview'][:80].replace('\n', ' ')
            print(f"gen={r['gen_tps']} t/s, {r['tokens_generated']} tokens, {r['wall_time_s']}s")
            print(f"    {preview}")
        else:
            print("FAILED")

    with open(GENERAL_RESULTS, 'w') as f:
        json.dump(general_data, f, indent=2)
    print(f"  Saved {model_key}")

# ---- Run coding 5-prompt suite ----
CODING_RESULTS = os.path.expanduser('~/projects/jetson-llm-benchmark/coding_benchmark_results.json')
with open(CODING_RESULTS) as f:
    coding_data = json.load(f)

print("\n" + "=" * 70)
print("CODING 5-PROMPT SUITE (html, python, c, basic, julia)")
print("=" * 70)

for model_key, gguf_file in NEW_MODELS:
    print(f"\nMODEL: {model_key}")
    if model_key not in coding_data['models']:
        coding_data['models'][model_key] = {}

    for pid, prompt_text in CODING_PROMPTS.items():
        if pid in coding_data['models'].get(model_key, {}):
            print(f"  SKIP {pid} (already done)")
            continue
        print(f"  {pid}...", end=' ', flush=True)
        r = run_one(model_key, gguf_file, prompt_text)
        if r:
            coding_data['models'][model_key][pid] = r
            preview = r['output_preview'][:80].replace('\n', ' ')
            print(f"gen={r['gen_tps']} t/s, {r['tokens_generated']} tokens, {r['wall_time_s']}s")
            print(f"    {preview}")
        else:
            print("FAILED")

    with open(CODING_RESULTS, 'w') as f:
        json.dump(coding_data, f, indent=2)
    print(f"  Saved {model_key}")

# ---- Summary ----
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for model_key, gguf_file in NEW_MODELS:
    g = general_data['models'].get(model_key, {})
    c = coding_data['models'].get(model_key, {})

    g_speeds = [g.get(p, {}).get('gen_tps', 0) for p in GENERAL_PROMPTS]
    c_speeds = [c.get(p, {}).get('gen_tps', 0) for p in CODING_PROMPTS]
    g_avg = sum(g_speeds) / len(g_speeds) if g_speeds else 0
    c_avg = sum(c_speeds) / len(c_speeds) if c_speeds else 0

    print(f"\n{model_key}:")
    print(f"  General: avg gen={g_avg:.1f} t/s")
    print(f"  Coding:  avg gen={c_avg:.1f} t/s")
    for pid in GENERAL_PROMPTS:
        r = g.get(pid, {})
        print(f"    {pid:10s}: gen={r.get('gen_tps',0):.1f} t/s, {r.get('tokens_generated',0)} tokens")
    for pid in CODING_PROMPTS:
        r = c.get(pid, {})
        print(f"    {pid:10s}: gen={r.get('gen_tps',0):.1f} t/s, {r.get('tokens_generated',0)} tokens")