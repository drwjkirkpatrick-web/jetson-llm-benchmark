#!/usr/bin/env python3
"""
Re-benchmark SmallThinker and Granite models with fixed harness:
- --jinja flag for proper chat template handling
- 2000 token limit (thinking models need room)
- Banner/output stripping to capture only the model's response
- Separates timing info from generated text
"""
import json, os, re, subprocess, time, sys

RESULTS = os.path.expanduser('~/projects/jetson-llm-benchmark/multiprompt_results.json')
LLAMA_CLI = os.path.expanduser('~/llama.cpp/build/bin/llama-cli')
MODELS_DIR = os.path.expanduser('~/models/bench-gguf')

PROMPTS = {
    'code': ('Code Generation',
        'Write a Python function that takes a list of integers and returns a new list with only the even numbers, sorted in ascending order. Include a docstring and type hints.'),
    'iambic': ('Iambic Pentameter',
        'Write a poem about the changing of seasons from autumn to winter. The poem must be strictly in iambic pentameter (10 syllables per line, alternating stress) and have exactly 8 lines.'),
    'prose': ('Clinical Prose',
        'Explain the difference between hypothyroidism and hyperthyroidism in 3 paragraphs. Include common symptoms, diagnostic approaches, and treatment considerations. Write in clear clinical prose.'),
    'creative': ('Creative Writing',
        'Write a vivid opening paragraph for a mystery novel set in a remote lighthouse during a storm. Use sensory details and create atmosphere. 200-300 words.'),
    'math': ('Mathematical Proof',
        'Prove that the square root of 2 is irrational. Start with "Theorem:" and use a proof by contradiction. Show each step clearly.'),
}

# Models that need re-benchmarking (banner pollution or thinking model issues)
REBENCH = [
    ('granite3-dense:2b', 'granite3-dense-2b.gguf'),
    ('granite3.2:2b',     'granite3.2-2b.gguf'),
    ('granite4:3b',       'granite4-3b.gguf'),
    ('smallthinker:3b',   'smallthinker-3b.gguf'),
]

def strip_banner(text):
    """Remove the llama.cpp startup banner and UI chrome from output."""
    # Remove the spinner + banner block
    # The banner ends with something like "> prompt_text\n"
    # or we look for the prompt echo and take everything after it

    # Remove loading spinner
    text = re.sub(r'Loading model\.\.\..*?(?=\n\n|\n[▄█])', '', text, flags=re.DOTALL)

    # Remove the ASCII art logo block
    text = re.sub(r'▄▄ ▄▄.*?(?=build|available|>)', '', text, flags=re.DOTALL)

    # Remove build info lines
    text = re.sub(r'^build\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^model\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^ftype\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^modalities\s+:.*$', '', text, flags=re.MULTILINE)

    # Remove available commands block
    text = re.sub(r'^available commands:.*?(?=\n\n|\n>|\Z)', '', text, flags=re.DOTALL | re.MULTILINE)

    # Remove the prompt echo (line starting with > )
    # Find the LAST occurrence of "> " which is the prompt echo
    lines = text.split('\n')
    prompt_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('> '):
            prompt_idx = i

    if prompt_idx >= 0:
        # Take everything after the prompt echo line
        lines = lines[prompt_idx + 1:]

    text = '\n'.join(lines)

    # Remove timing/stats lines at the end
    text = re.sub(r'\[ Prompt:.*Generation:.*\]', '', text)
    text = re.sub(r'^prompt eval time.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^generation time.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^system prompt.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^all time.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*$', '', text, flags=re.MULTILINE)  # Remove blank lines

    # Strip chat template tokens
    text = text.replace('<|im_start|>', '').replace('<|im_end|>', '')
    text = re.sub(r'<\|channel\|>.*?<\|end\|>', '', text, flags=re.DOTALL)
    text = text.replace('</s>', '')
    text = text.replace('<s>', '')

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def run_bench(model_name, gguf_file, prompt_id, prompt_text):
    """Run a single benchmark and return parsed results."""
    model_path = os.path.join(MODELS_DIR, gguf_file)
    cmd = [
        LLAMA_CLI, '-m', model_path,
        '-p', prompt_text,
        '-n', '2000',  # 2000 tokens — local LLM, power only cost
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
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start
        raw_output = r.stdout + r.stderr

        # Extract timing
        gen_tps = None
        prompt_tps = None
        for line in raw_output.split('\n'):
            m = re.search(r'Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s', line)
            if m:
                prompt_tps = float(m.group(1))
                gen_tps = float(m.group(2))

        # Strip banner and get clean response
        clean = strip_banner(raw_output)

        # Count tokens (approximate: words in clean output)
        token_count = len(clean.split()) if clean else 0

        return {
            'label': PROMPTS[prompt_id][0],
            'gen_tps': gen_tps,
            'prompt_tps': prompt_tps,
            'tokens_generated': token_count,
            'wall_time_s': round(elapsed, 1),
            'output_preview': clean[:800],
            'output_full_len': len(clean),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            'label': PROMPTS[prompt_id][0],
            'gen_tps': None, 'prompt_tps': None,
            'tokens_generated': 0,
            'wall_time_s': round(elapsed, 1),
            'output_preview': 'TIMEOUT after 300s',
            'output_full_len': 0,
        }
    except Exception as e:
        return {
            'label': PROMPTS[prompt_id][0],
            'gen_tps': None, 'prompt_tps': None,
            'tokens_generated': 0,
            'wall_time_s': 0,
            'output_preview': f'ERROR: {e}',
            'output_full_len': 0,
        }


def main():
    # Load existing results
    with open(RESULTS) as f:
        data = json.load(f)

    total = len(REBENCH) * len(PROMPTS)
    done = 0

    for model_name, gguf_file in REBENCH:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name} ({gguf_file})")
        print(f"{'='*70}")

        if model_name not in data['models']:
            data['models'][model_name] = {}

        for pid, (label, prompt) in PROMPTS.items():
            done += 1
            print(f"\n[{done}/{total}] {model_name} / {pid}...", end=' ', flush=True)
            result = run_bench(model_name, gguf_file, pid, prompt)
            data['models'][model_name][pid] = result

            gen = result.get('gen_tps', 'N/A')
            tok = result.get('tokens_generated', 0)
            wall = result.get('wall_time_s', 0)
            print(f"-> gen={gen} t/s, {tok} tokens, {wall}s")

            # Show first 100 chars of output
            preview = result.get('output_preview', '')[:120]
            print(f"   Preview: {preview}...")

        # Save after each model
        with open(RESULTS, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {model_name}")

    print(f"\n\nDone! Re-benchmarked {len(REBENCH)} models x {len(PROMPTS)} prompts = {total} runs")
    print(f"Total models in results: {len(data['models'])}")


if __name__ == '__main__':
    main()