#!/usr/bin/env python3
"""Benchmark new models and merge into existing results JSON."""
import json, os, re, subprocess, time, sys

RESULTS_FILE = os.path.expanduser("~/projects/jetson-llm-benchmark/multiprompt_results.json")
LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
MODELS_DIR = os.path.expanduser("~/models/bench-gguf")

PROMPTS = {
    "code": {
        "label": "Code Generation",
        "prompt": "Write a Python function that takes a list of integers and returns a new list with only the even numbers, sorted in ascending order. Include a docstring and type hints.",
    },
    "iambic": {
        "label": "Iambic Pentameter",
        "prompt": "Write a poem about the changing of seasons from autumn to winter. The poem must be strictly in iambic pentameter (10 syllables per line, alternating stress) and have exactly 8 lines.",
    },
    "prose": {
        "label": "Clinical Prose",
        "prompt": "Explain the difference between hypothyroidism and hyperthyroidism in 3 paragraphs. Include common symptoms, diagnostic approaches, and treatment considerations. Write in clear clinical prose.",
    },
    "creative": {
        "label": "Creative Writing",
        "prompt": "Write a vivid opening paragraph for a mystery novel set in a remote lighthouse during a storm. Use sensory details and create atmosphere. 200-300 words.",
    },
    "math": {
        "label": "Mathematical Proof",
        "prompt": "Prove that the square root of 2 is irrational. Start with 'Theorem:' and use a proof by contradiction. Show each step clearly.",
    },
}

NEW_MODELS = [
    ("granite3-dense:2b", "granite3-dense-2b.gguf"),
    ("smallthinker:3b", "smallthinker-3b.gguf"),
]

# Load existing results
with open(RESULTS_FILE) as f:
    data = json.load(f)

for model_name, gguf_file in NEW_MODELS:
    model_path = os.path.join(MODELS_DIR, gguf_file)
    if not os.path.exists(model_path):
        print(f"SKIP {model_name}: file not found at {model_path}")
        continue

    if model_name not in data["models"]:
        data["models"][model_name] = {}

    for pid, pinfo in PROMPTS.items():
        print(f"[BENCH] {model_name} / {pid} ({pinfo['label']})...", end=" ", flush=True)
        cmd = [
            LLAMA_CLI, "-m", model_path,
            "-p", pinfo["prompt"],
            "-n", "300", "-c", "2048", "--temp", "0.3",
            "-ngl", "99", "-fa", "on",
            "--no-conversation", "--no-display-prompt", "-st",
        ]
        try:
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = time.time() - start

            output = result.stdout
            # Parse timing info
            gen_tps = None
            prompt_tps = None
            tokens_gen = 0

            # Extract from llama.cpp -st output: [ Prompt: 92.5 t/s | Generation: 24.5 t/s ]
            for line in output.split("\n"):
                m = re.search(r"Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s", line)
                if m:
                    prompt_tps = float(m.group(1))
                    gen_tps = float(m.group(2))

            # Count generated tokens
            gen_section = output.split("</s>")[-1] if "</s>" in output else output
            # Strip timing info
            gen_section = re.sub(r"(prompt eval|generation|system prompt|all).*", "", gen_section, flags=re.DOTALL).strip()
            tokens_gen = len(gen_section.split())

            # Get preview
            preview = gen_section[:500] if gen_section else ""

            entry = {
                "label": pinfo["label"],
                "gen_tps": gen_tps,
                "prompt_tps": prompt_tps,
                "tokens_generated": tokens_gen,
                "wall_time_s": round(elapsed, 1),
                "output_preview": preview,
            }
            data["models"][model_name][pid] = entry
            print(f"-> gen={gen_tps} t/s, prompt={prompt_tps} t/s, {elapsed:.1f}s, {tokens_gen} tokens")
        except Exception as e:
            print(f"-> ERROR: {e}")
            data["models"][model_name][pid] = {"error": str(e)}

    # Save after each model
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved results for {model_name}")

print(f"\nDone! Results saved to {RESULTS_FILE}")
print(f"Total models: {len(data['models'])}")