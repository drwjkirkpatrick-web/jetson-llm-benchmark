#!/usr/bin/env python3
"""
Compare llama.cpp vs Ollama for the same model/quantization.
Generates a side-by-side bar chart (ASCII) of throughput.
"""

import json
import sys
from pathlib import Path


def draw_bar(label: str, value: float, max_val: float, width: int = 40):
    if max_val <= 0:
        filled = 0
    else:
        filled = int((value / max_val) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:<20} {bar} {value:.1f}"


def compare(results_path: Path):
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", [])

    # Build quant -> {backend: tok/s} mapping
    quant_map = {}
    for r in results:
        if "error" in r:
            continue
        model = r.get("model", "")
        be = r.get("backend", "")
        tok = r.get("tok_per_sec", 0)

        quant = None
        for q in ["Q4_K_M", "Q3_K_M", "Q2_K", "Q4_K_S", "Q5_K_M"]:
            if q in model:
                quant = q
                break
        if not quant:
            continue

        quant_map.setdefault(quant, {})[be] = tok

    if not quant_map:
        print("No comparable results found.")
        return

    max_tok = max(
        max(v.values()) for v in quant_map.values() if v
    )

    print("=" * 60)
    print("  LLAMA.CPP vs OLLAMA THROUGHPUT COMPARISON")
    print("=" * 60)
    print()
    print(f"  {'Quant':<8} {'Backend':<12} {'tok/s':<8} {'Chart':<{40+8}}")
    print(f"  {'-'*8} {'-'*12} {'-'*8}")

    for quant in sorted(quant_map.keys()):
        backends = quant_map[quant]
        print()
        for be in ["llama.cpp", "ollama"]:
            tok = backends.get(be, 0)
            if tok:
                print(draw_bar(f"{quant} {be}", tok, max_tok))
            else:
                print(f"  {quant} {be:<12} N/A")

        if "llama.cpp" in backends and "ollama" in backends:
            llama_tok = backends["llama.cpp"]
            ollama_tok = backends["ollama"]
            speedup = llama_tok / ollama_tok if ollama_tok else 0
            print(f"  {'':<20} Speedup: {speedup:.1f}x")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m scripts.compare_engines results/benchmark.json")
        sys.exit(1)

    compare(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
