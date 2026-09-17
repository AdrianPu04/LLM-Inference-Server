"""
Run this FIRST, before writing any scheduler/batching code.
Verifies CUDA, VRAM, and whether vLLM installs/imports cleanly on a
consumer (Ada) GPU. If vLLM fails here, fall back to the PyTorch-native
path noted at the bottom — that's a legitimate, arguably more impressive
route, not a downgrade.

Usage:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    python check_env.py --skip-vllm      # just check CUDA/VRAM first
    pip install vllm
    python check_env.py                  # full check including vLLM
"""

import argparse
import sys


def check_cuda():
    print("=" * 60)
    print("CUDA / GPU check")
    print("=" * 60)
    try:
        import torch
    except ImportError:
        print("[FAIL] torch not installed. Run:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    print(f"torch version:      {torch.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available. Check nvidia-smi and driver install.")
        sys.exit(1)

    print(f"CUDA version:       {torch.version.cuda}")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    print(f"GPU name:            {props.name}")
    print(f"Total VRAM:          {vram_gb:.1f} GB")
    print(f"Compute capability:  {props.major}.{props.minor}  (Ada = 8.9)")

    print()
    print("Rough model sizing guidance for this GPU:")
    usable = vram_gb - 2.0  # leave headroom for OS/driver/activations
    print(f"  Usable VRAM (rough): ~{usable:.1f} GB")
    print(f"  fp16 1.5B model:  ~3 GB weights  -> {'OK' if usable > 3 else 'TIGHT'}")
    print(f"  fp16 3B model:    ~6 GB weights  -> {'OK' if usable > 6 else 'TIGHT'}")
    print(f"  fp16 7-8B model:  ~16 GB weights -> {'OK' if usable > 16 else 'WILL NOT FIT'}")
    print(f"  AWQ/GPTQ 4-bit 7-8B: ~5-6 GB     -> {'OK' if usable > 6 else 'TIGHT'}")
    print()
    return vram_gb


def check_vllm():
    print("=" * 60)
    print("vLLM import check")
    print("=" * 60)
    try:
        import vllm
        print(f"[OK] vLLM imported successfully. Version: {vllm.__version__}")
    except ImportError as e:
        print(f"[FAIL] vLLM import failed: {e}")
        print()
        print("This is common on consumer/Ada GPUs and not a dead end.")
        print("Fallback plan: build the server on raw PyTorch + HF generate(),")
        print("and implement your own batching/scheduling loop directly.")
        print("This is MORE of your own code, not less impressive.")
        sys.exit(1)
    except Exception as e:
        print(f"[FAIL] vLLM imported but raised an error: {e}")
        sys.exit(1)


def try_load_tiny_model():
    print("=" * 60)
    print("Smoke test: loading a tiny model with vLLM's LLMEngine")
    print("=" * 60)
    try:
        from vllm import LLM, SamplingParams
        # Smallest reasonable test — swap for your real model once this passes
        llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.5)
        params = SamplingParams(max_tokens=20)
        out = llm.generate(["Hello, my name is"], params)
        print("[OK] Generation succeeded:")
        print(f"  {out[0].outputs[0].text!r}")
    except Exception as e:
        print(f"[FAIL] vLLM engine smoke test failed: {e}")
        print("If this fails but the import succeeded, check CUDA/driver version")
        print("mismatch — Ada support needs a reasonably recent vLLM + CUDA 12.1+.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-vllm", action="store_true",
                         help="Only check CUDA/VRAM, skip vLLM import/smoke test")
    args = parser.parse_args()

    vram_gb = check_cuda()

    if not args.skip_vllm:
        check_vllm()
        try_load_tiny_model()

    print("=" * 60)
    print("All checks passed.")
    print("=" * 60)
