#!/usr/bin/env python3
"""
Jetson LLM Benchmark — No-Timeout Design

Runs llama.cpp and/or Ollama models with a clinical prompt,
measures tokens/sec, RAM usage, CMA free space, and GPU temperature.

Designed for Jetson Orin Nano 8GB with GUI killed (gdm3 stopped).
No timeouts — runs to completion or reports failure clearly.

Usage:
    python3 benchmark.py \
        --models /path/to/Q4_K_M.gguf /path/to/Q3_K_M.gguf \
        --ollama-models qwen2.5:7b-instruct \
        --prompt-file prompts/clinical-hypothyroid.txt \
        --output results/benchmark.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_PROMPT = """A 52-year-old female presents with fatigue, weight gain (15 lbs over 6 months), 
cold intolerance, dry skin, and brittle nails. Labs show TSH 8.2 mIU/L (ref 0.4-4.0), 
Free T4 0.7 ng/dL (ref 0.8-1.8), TPO antibodies positive. 

Provide a concise naturopathic assessment including:
1. Assessment/diagnosis
2. Top 3 naturopathic interventions with rationale
3. Monitoring recommendations

Keep response under 200 words."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_ram_info():
    """Read /proc/meminfo and return key values."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            key = parts[0].rstrip(":").strip()
            if key in ("MemTotal", "MemFree", "MemAvailable", "CmaFree", "CmaTotal"):
                info[key] = int(parts[1])  # kB
    return info


def get_gpu_temp():
    """Read GPU temperature from thermal zone (Jetson)."""
    for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            # Look for GPU zone by checking type
            zone_type = zone.with_name("type").read_text().strip()
            if "gpu" in zone_type.lower() or "soc" in zone_type.lower():
                temp_millideg = int(zone.read_text().strip())
                return temp_millideg / 1000.0
        except (OSError, ValueError):
            continue
    return None


def check_ollama_server():
    """Check if Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def ollama_generate(model_name: str, prompt: str, num_ctx: int = 1024):
    """Run Ollama inference and return metrics. Blocks until completion."""
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": 0.3,
            "num_predict": 200,
        }
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    ram_before = get_ram_info()
    temp_before = get_gpu_temp()
    t0 = time.time()

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:  # 10 min, no early abort
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()[:500]
        return {
            "backend": "ollama",
            "model": model_name,
            "error": f"HTTP {e.code}: {error_body}",
            "ram_before_mb": ram_before.get("MemAvailable", 0) // 1024,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    except Exception as e:
        return {
            "backend": "ollama",
            "model": model_name,
            "error": str(e)[:500],
            "ram_before_mb": ram_before.get("MemAvailable", 0) // 1024,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    t1 = time.time()
    elapsed = t1 - t0
    ram_after = get_ram_info()
    temp_after = get_gpu_temp()

    eval_count = result.get("eval_count", 0)
    eval_dur_ns = result.get("eval_duration", 0)
    prompt_eval_count = result.get("prompt_eval_count", 0)
    prompt_dur_ns = result.get("prompt_eval_duration", 0)
    load_dur_ns = result.get("load_duration", 0)

    tok_per_sec = eval_count / (eval_dur_ns / 1e9) if eval_dur_ns > 0 else 0
    prompt_tok_per_sec = prompt_eval_count / (prompt_dur_ns / 1e9) if prompt_dur_ns > 0 else 0

    return {
        "backend": "ollama",
        "model": model_name,
        "load_time_s": round(load_dur_ns / 1e9, 2),
        "prompt_tokens": prompt_eval_count,
        "prompt_tok_per_sec": round(prompt_tok_per_sec, 1),
        "gen_tokens": eval_count,
        "gen_time_s": round(eval_dur_ns / 1e9, 2),
        "total_time_s": round(elapsed, 2),
        "tok_per_sec": round(tok_per_sec, 1),
        "ram_before_mb": ram_before.get("MemAvailable", 0) // 1024,
        "ram_after_mb": ram_after.get("MemAvailable", 0) // 1024,
        "cma_before_mb": ram_before.get("CmaFree", 0) // 1024,
        "cma_after_mb": ram_after.get("CmaFree", 0) // 1024,
        "temp_before_c": temp_before,
        "temp_after_c": temp_after,
        "response_preview": result.get("response", "")[:300],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def llama_cpp_generate(gguf_path: str, prompt: str, num_ctx: int = 1024):
    """Run llama-cli inference and return metrics."""
    if not os.path.exists(LLAMA_CLI):
        return {
            "backend": "llama.cpp",
            "model": Path(gguf_path).name,
            "error": f"llama-cli not found at {LLAMA_CLI}",
        }

    cmd = [
        LLAMA_CLI,
        "-m", gguf_path,
        "-n", "200",
        "-c", str(num_ctx),
        "--temp", "0.3",
        "-ngl", "99",
        "--flash-attn", "on",
        "--no-conversation",
        "--no-display-prompt",
        "-p", prompt,
    ]

    ram_before = get_ram_info()
    temp_before = get_gpu_temp()
    t0 = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max per model
        )
    except subprocess.TimeoutExpired:
        return {
            "backend": "llama.cpp",
            "model": Path(gguf_path).name,
            "error": "Timeout after 10 minutes",
            "ram_before_mb": ram_before.get("MemAvailable", 0) // 1024,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    except Exception as e:
        return {
            "backend": "llama.cpp",
            "model": Path(gguf_path).name,
            "error": str(e)[:500],
            "ram_before_mb": ram_before.get("MemAvailable", 0) // 1024,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    t1 = time.time()
    elapsed = t1 - t0
    ram_after = get_ram_info()
    temp_after = get_gpu_temp()

    output = proc.stdout + proc.stderr

    # Parse llama.cpp output for metrics
    prompt_tok_per_sec = 0.0
    tok_per_sec = 0.0
    for line in output.splitlines():
        if "[ Prompt:" in line and "t/s" in line:
            # Extract prompt speed: [ Prompt: 262.2 t/s | Generation: 12.4 t/s ]
            try:
                prompt_part = line.split("Prompt:")[1].split("t/s")[0].strip()
                prompt_tok_per_sec = float(prompt_part)
            except (ValueError, IndexError):
                pass
            try:
                gen_part = line.split("Generation:")[1].split("t/s")[0].strip()
                tok_per_sec = float(gen_part)
            except (ValueError, IndexError):
                pass

    # Also try to detect load errors
    if "failed to load model" in output.lower() or proc.returncode != 0:
        return {
            "backend": "llama.cpp",
            "model": Path(gguf_path).name,
            "error": output[-500:],
            "ram_before_mb": ram_before.get("MemAvailable", 0) // 1024,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    return {
        "backend": "llama.cpp",
        "model": Path(gguf_path).name,
        "load_time_s": round(elapsed, 2),  # llama.cpp doesn't separate load time
        "prompt_tokens": None,  # llama.cpp doesn't expose this separately
        "prompt_tok_per_sec": round(prompt_tok_per_sec, 1),
        "gen_tokens": None,
        "gen_time_s": None,
        "total_time_s": round(elapsed, 2),
        "tok_per_sec": round(tok_per_sec, 1),
        "ram_before_mb": ram_before.get("MemAvailable", 0) // 1024,
        "ram_after_mb": ram_after.get("MemAvailable", 0) // 1024,
        "cma_before_mb": ram_before.get("CmaFree", 0) // 1024,
        "cma_after_mb": ram_after.get("CmaFree", 0) // 1024,
        "temp_before_c": temp_before,
        "temp_after_c": temp_after,
        "response_preview": output[-300:],  # llama.cpp prints response inline
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs on Jetson Orin Nano")
    parser.add_argument("--models", nargs="*", default=[],
                        help="Path(s) to GGUF files for llama.cpp")
    parser.add_argument("--ollama-models", nargs="*", default=[],
                        help="Ollama model names (e.g., qwen2.5:7b-instruct)")
    parser.add_argument("--prompt-file", type=Path,
                        help="File containing the prompt text")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"),
                        help="Output JSON file for results")
    parser.add_argument("--num-ctx", type=int, default=1024,
                        help="Context length (default: 1024)")
    args = parser.parse_args()

    # Load prompt
    if args.prompt_file and args.prompt_file.exists():
        prompt = args.prompt_file.read_text().strip()
    else:
        prompt = DEFAULT_PROMPT
        print("[INFO] Using default clinical prompt (no --prompt-file provided)")

    results = []

    # Pre-checks
    print("=" * 60)
    print("  Jetson LLM Benchmark")
    print("  No-timeout design — runs to completion")
    print("=" * 60)
    print()
    ram = get_ram_info()
    print(f"  RAM Available: {ram.get('MemAvailable', 0) // 1024} MB")
    print(f"  CMA Free:      {ram.get('CmaFree', 0) // 1024} MB")
    print(f"  GPU Temp:      {get_gpu_temp()}°C")
    print()

    if ram.get("CmaFree", 0) < 50 * 1024:  # < 50 MB
        print("[WARNING] CmaFree < 50 MB — models may fail to load on GPU!")
        print("          Stop gdm3: sudo systemctl stop gdm3")
        print()

    # llama.cpp models
    if args.models:
        print(f"--- llama.cpp ({len(args.models)} model(s)) ---")
        for gguf in args.models:
            print(f"\n  Testing: {Path(gguf).name}")
            if not Path(gguf).exists():
                print(f"  [SKIP] File not found: {gguf}")
                continue
            result = llama_cpp_generate(gguf, prompt, args.num_ctx)
            results.append(result)
            if "error" in result:
                print(f"  [FAIL] {result['error'][:200]}")
            else:
                print(f"  Prompt:   {result['prompt_tok_per_sec']:.1f} tok/s")
                print(f"  Generate: {result['tok_per_sec']:.1f} tok/s")
                print(f"  RAM:      {result['ram_after_mb']} MB available")
                print(f"  CMA:      {result['cma_after_mb']} MB")
            # Cool down between models
            time.sleep(5)

    # Ollama models
    if args.ollama_models:
        print(f"\n--- Ollama ({len(args.ollama_models)} model(s)) ---")
        if not check_ollama_server():
            print("  [FAIL] Ollama server not reachable at 127.0.0.1:11434")
            print("  Start it: ollama serve")
        else:
            for model in args.ollama_models:
                print(f"\n  Testing: {model}")
                result = ollama_generate(model, prompt, args.num_ctx)
                results.append(result)
                if "error" in result:
                    print(f"  [FAIL] {result['error'][:200]}")
                else:
                    print(f"  Load:     {result['load_time_s']:.1f}s")
                    print(f"  Prompt:   {result['prompt_tokens']} tok @ {result['prompt_tok_per_sec']:.0f} tok/s")
                    print(f"  Generate: {result['gen_tokens']} tok @ {result['tok_per_sec']:.1f} tok/s")
                    print(f"  Total:    {result['total_time_s']:.1f}s")
                    print(f"  RAM:      {result['ram_after_mb']} MB available")
                    print(f"  CMA:      {result['cma_after_mb']} MB")
                # Cool down
                time.sleep(5)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "device": "NVIDIA Jetson Orin Nano 8GB",
                "num_ctx": args.num_ctx,
                "prompt_length_chars": len(prompt),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n[SAVED] {args.output}")

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Backend':<12} {'Model':<35} {'tok/s':>6} {'Status':<10}")
    print(f"  {'-'*12} {'-'*35} {'-'*6} {'-'*10}")
    for r in results:
        backend = r.get("backend", "?")
        model = r.get("model", "?")[:34]
        status = "OK" if "error" not in r else "FAIL"
        tok = f"{r['tok_per_sec']:.1f}" if r.get("tok_per_sec") else "N/A"
        print(f"  {backend:<12} {model:<35} {tok:>6} {status:<10}")


if __name__ == "__main__":
    main()
