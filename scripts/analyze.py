#!/usr/bin/env python3
"""
Analyze benchmark results JSON and print comparison tables.
"""

import json
import sys
from pathlib import Path


def analyze_results(data: dict):
    results = data.get("results", [])
    if not results:
        print("No results found.")
        return

    print("=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Device: {data['metadata']['device']}")
    print(f"  Date:   {data['metadata']['timestamp']}")
    print(f"  num_ctx: {data['metadata']['num_ctx']}")
    print()

    # Group by backend
    backends = {}
    for r in results:
        be = r.get("backend", "unknown")
        backends.setdefault(be, []).append(r)

    for backend, items in backends.items():
        print(f"\n  --- {backend.upper()} ---")
        print(f"  {'Model':<35} {'tok/s':>6} {'Load(s)':>8} {'RAM(MB)':>8} {'CMA(MB)':>8} {'Status':>8}")
        print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for r in items:
            model = r.get("model", "?")[:34]
            status = "OK" if "error" not in r else "FAIL"
            tok = f"{r.get('tok_per_sec', 0):.1f}" if r.get("tok_per_sec") else "N/A"
            load = f"{r.get('load_time_s', 0):.1f}" if r.get("load_time_s") else "N/A"
            ram = str(r.get("ram_after_mb", "N/A"))
            cma = str(r.get("cma_after_mb", "N/A"))
            print(f"  {model:<35} {tok:>6} {load:>8} {ram:>8} {cma:>8} {status:>8}")

    # Cross-backend comparison for same quant
    print("\n  --- CROSS-BACKEND COMPARISON ---")
    print(f"  {'Quantization':<20} {'llama.cpp tok/s':>15} {'Ollama tok/s':>15} {'Speedup':>10}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*10}")

    quant_map = {}
    for r in results:
        if "error" in r:
            continue
        model_name = r.get("model", "")
        be = r.get("backend", "")
        tok = r.get("tok_per_sec", 0)

        # Extract quant from model name
        quant = None
        for q in ["Q4_K_M", "Q3_K_M", "Q2_K", "Q4_K_S", "Q5_K_M"]:
            if q in model_name:
                quant = q
                break
        if not quant:
            continue

        if quant not in quant_map:
            quant_map[quant] = {}
        quant_map[quant][be] = tok

    for quant, backends in quant_map.items():
        llama_tok = backends.get("llama.cpp", 0)
        ollama_tok = backends.get("ollama", 0)
        if llama_tok and ollama_tok:
            speedup = llama_tok / ollama_tok if ollama_tok else 0
            print(f"  {quant:<20} {llama_tok:>14.1f} {ollama_tok:>14.1f} {speedup:>9.1f}x")
        elif llama_tok:
            print(f"  {quant:<20} {llama_tok:>14.1f} {'N/A':>15} {'N/A':>10}")
        elif ollama_tok:
            print(f"  {quant:<20} {'N/A':>15} {ollama_tok:>14.1f} {'N/A':>10}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m scripts.analyze results/benchmark.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    analyze_results(data)


if __name__ == "__main__":
    main()
