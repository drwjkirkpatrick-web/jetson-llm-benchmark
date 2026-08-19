#!/usr/bin/env python3
"""
Re-run DeepSeek 7B and 8B benchmarks with full output capture.
Both are thinking models: --jinja + 4000 tokens, ctx=2048.
"""
import json, os, re, subprocess, time, sys

LLAMA_CLI = os.path.expanduser('~/llama.cpp/build/bin/llama-cli')

MODELS = {
    'deepseek-r1-qwen-7b': {
        'path': os.path.expanduser('~/models/bench-gguf/deepseek-r1-qwen-7b.gguf'),
        'ctx': 2048,
        'max_tokens': 4000,
        'timeout': 600,
    },
    'deepseek-r1-llama-8b': {
        'path': os.path.expanduser('~/models/bench-gguf/deepseek-r1-llama-8b.gguf'),
        'ctx': 2048,
        'max_tokens': 4000,
        'timeout': 900,  # 15 min per prompt for 8B
    },
}

GENERAL_PROMPTS = {
    'code': 'Write a Python function called `merge_sorted_lists` that takes two sorted lists of integers and returns a single merged sorted list. Include type hints, a docstring, and handle edge cases like empty lists.',
    'iambic': 'Write a 14-line poem in strict iambic pentameter about the passage of seasons. Each line must have exactly ten syllables with the stress pattern: da-DUM da-DUM da-DUM da-DUM da-DUM.',
    'prose': 'Write a concise clinical case writeup (approximately 200 words) for a 45-year-old female presenting with fatigue, weight gain, cold intolerance, and brittle nails. Include history, physical exam findings, assessment, and naturopathic treatment plan.',
    'creative': 'Write a short story (300-500 words) about an old lighthouse keeper who discovers a message in a bottle that changes everything. Use vivid sensory details about the sea, the light, and the keeper`s isolation.',
    'math': 'Prove that for any positive integer n, the sum 1 + 2 + 3 + ... + n equals n(n+1)/2. Provide a clear, step-by-step mathematical proof with proper notation.',
}

CODING_PROMPTS = {
    'html': 'Create a complete HTML page for a personal portfolio website. Include a header with navigation, a hero section with name and tagline, an about section, a projects section with 3 sample project cards using flexbox, and a footer. Add inline CSS with a modern color scheme, responsive design with a media query for mobile, and Google Fonts link.',
    'python': 'Write a Python class called `DataProcessor` that reads a CSV file, filters rows based on a column condition, computes summary statistics (mean, median, std dev) for numeric columns, and exports results to JSON. Include type hints, docstrings, proper error handling with try/except, and a `__main__` block with example usage.',
    'c': 'Write a complete C program that implements a thread-safe queue using a linked list. Include: a struct for queue nodes, a struct for the queue with a pthread mutex, functions for enqueue, dequeue, peek, size, and free_queue. Add a main function that demonstrates usage with 3 producer threads and 1 consumer thread.',
    'basic': 'Write a complete TRS-80 Level II BASIC program that implements a simple text adventure game. Use line numbers (starting at 10, incrementing by 10). The game should have at least 3 rooms with descriptions, items the player can pick up with a GET command, and a goal to find a treasure. Use INPUT for player commands, PRINT for output, and GOTO for room navigation. Include CLEAR to reset the screen.',
    'julia': 'Write a Julia function called `newton_raphson` that finds the root of a function using the Newton-Raphson method. Include: type annotations for all arguments, a docstring with usage examples, proper error handling for non-convergence, a maximum iteration parameter with default value, and a return type annotation. The function should take f (the function), f_prime (its derivative), x0 (initial guess), and optional parameters for tolerance and max iterations.',
}

def strip_banner(raw):
    """Remove llama.cpp startup banner, keep only model response."""
    text = raw.replace('\x08', '')  # Remove backspace chars from spinner
    # Find the prompt echo line (starts with '> ')
    idx = text.find('> ')
    if idx >= 0:
        rest = text[idx:]
        first_nl = rest.find('\n')
        if first_nl >= 0:
            text = rest[first_nl+1:]
        else:
            text = ''
    # Remove trailing llama-cli stats
    text = re.sub(r'\[ Prompt:.*?\].*$', '', text, flags=re.DOTALL)
    # Remove chat template tokens
    for tok in ['<|im_start|>', '<|im_end|>', '<|begin_of_text|>', '<|end_of_text|>']:
        text = text.replace(tok, '')
    # Remove "Exiting..." and "stop or exit" type lines at the end
    text = re.sub(r'\nExiting\.\.\..*$', '', text)
    return text.strip()

def run_prompt(model_key, cfg, pid, prompt_text, suite):
    cmd = [
        LLAMA_CLI, '-m', cfg['path'],
        '-p', prompt_text,
        '-n', str(cfg['max_tokens']),
        '-c', str(cfg['ctx']),
        '--temp', '0.3', '-ngl', '99', '-fa', 'on',
        '--no-conversation', '--no-display-prompt',
        '--jinja', '-st'
    ]
    print(f'  [{model_key}] [{suite}] {pid}...', flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg['timeout'])
        wall = time.time() - t0
        clean = strip_banner(r.stdout)
        tokens = len(clean.split())
        # Get actual gen tps from stderr
        gen_tps = 0.0
        for line in r.stderr.split('\n'):
            m = re.search(r'Generation:\s*(\d+\.\d+)\s*t/s', line)
            if m:
                gen_tps = float(m.group(1))
                break
            m2 = re.search(r'eval time.*?=\s*(\d+\.\d+)\s*ms.*?n_eval\s*=\s*(\d+)', line)
            if m2:
                eval_ms = float(m2.group(1))
                n_eval = int(m2.group(2))
                if eval_ms > 0:
                    gen_tps = n_eval / (eval_ms / 1000.0)
        if gen_tps == 0.0 and wall > 0:
            gen_tps = round(tokens / wall, 2)
        print(f'    gen={gen_tps:.1f} t/s, {tokens} tokens, {wall:.1f}s', flush=True)
        print(f'    First 150: {clean[:150]}', flush=True)
        return {
            'gen_tps': round(gen_tps, 2),
            'prompt_tps': 0,
            'tokens_generated': tokens,
            'wall_time_s': round(wall, 1),
            'output_full': clean
        }
    except subprocess.TimeoutExpired:
        print(f'    TIMEOUT after {cfg["timeout"]}s', flush=True)
        return {
            'gen_tps': 0, 'prompt_tps': 0, 'tokens_generated': 0,
            'wall_time_s': cfg['timeout'], 'output_full': 'TIMEOUT'
        }

# Process models specified on command line, or all
model_keys = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

for model_key in model_keys:
    cfg = MODELS[model_key]
    print(f"\n{'='*60}", flush=True)
    print(f"  Benchmarking: {model_key}", flush=True)
    print(f"  ctx={cfg['ctx']}, max_tokens={cfg['max_tokens']}", flush=True)
    print(f"{'='*60}", flush=True)

    # General suite
    general = {}
    gen_file = f'deepseek_{model_key}_general_full.json'
    for pid, pt in GENERAL_PROMPTS.items():
        general[pid] = run_prompt(model_key, cfg, pid, pt, 'GEN')
        # Save incrementally
        with open(gen_file, 'w') as f:
            json.dump({'prompts': GENERAL_PROMPTS, 'models': {model_key: general}}, f, indent=2)
    print(f'General done for {model_key}', flush=True)

    # Coding suite
    coding = {}
    cod_file = f'deepseek_{model_key}_coding_full.json'
    for pid, pt in CODING_PROMPTS.items():
        coding[pid] = run_prompt(model_key, cfg, pid, pt, 'COD')
        with open(cod_file, 'w') as f:
            json.dump({'prompts': CODING_PROMPTS, 'models': {model_key: coding}}, f, indent=2)
    print(f'ALL DONE for {model_key}', flush=True)