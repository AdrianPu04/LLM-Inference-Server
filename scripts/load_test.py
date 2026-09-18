"""
Load generator for benchmarking any of the servers (naive, batched,
continuous-batched) against the same interface: POST /generate.

This is what produces the throughput/latency numbers for your results
table and the "collapse curve" graph for Phase 1.

Usage:
    python load_test.py --url http://localhost:8000/generate \
        --concurrency 1 2 4 8 16 32 --requests-per-level 20 \
        --out results/naive_1.5b.json

    # Mixed lengths (cycles 32 / 128 / 256 across requests):
    python load_test.py --url http://localhost:8000/generate \
        --concurrency 1 2 4 8 16 32 --max-new-tokens 32 128 256 \
        --out results/continuous_mixed.json
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

PROMPTS = [
    "Explain the difference between a stack and a queue.",
    "Write a short story about a robot learning to paint.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What are the main causes of the French Revolution?",
    "Describe how photosynthesis works at a high level.",
]


async def single_request(client: httpx.AsyncClient, url: str, prompt: str, max_new_tokens: int):
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens}
    start = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        wall_latency = time.perf_counter() - start
        return {
            "success": True,
            "wall_latency_s": wall_latency,
            "completion_tokens": data.get("completion_tokens", 0),
        }
    except Exception as e:
        wall_latency = time.perf_counter() - start
        return {"success": False, "wall_latency_s": wall_latency, "error": str(e)}


async def run_level(url: str, concurrency: int, n_requests: int, max_new_tokens: list[int]):
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(n_requests):
            prompt = PROMPTS[i % len(PROMPTS)]
            tokens = max_new_tokens[i % len(max_new_tokens)]
            tasks.append(single_request(client, url, prompt, tokens))

        sem = asyncio.Semaphore(concurrency)

        async def bounded(task_coro):
            async with sem:
                return await task_coro

        start = time.perf_counter()
        results = await asyncio.gather(*[bounded(t) for t in tasks])
        wall_time = time.perf_counter() - start

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = sorted(r["wall_latency_s"] for r in successes)
    total_completion_tokens = sum(r["completion_tokens"] for r in successes)

    def pct(p):
        if not latencies:
            return None
        idx = min(int(len(latencies) * p), len(latencies) - 1)
        return latencies[idx]

    summary = {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "n_success": len(successes),
        "n_failure": len(failures),
        "wall_time_s": wall_time,
        "throughput_req_per_s": len(successes) / wall_time if wall_time > 0 else 0,
        "throughput_tokens_per_s": total_completion_tokens / wall_time if wall_time > 0 else 0,
        "p50_latency_s": pct(0.50),
        "p90_latency_s": pct(0.90),
        "p99_latency_s": pct(0.99),
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "max_new_tokens": max_new_tokens,
    }
    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        nargs="+",
        default=[128],
        help="One value = fixed length; several values cycle per request (mixed length)",
    )
    parser.add_argument("--out", default=None, help="Optional path to dump JSON results")
    args = parser.parse_args()

    all_results = []
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"{'concurrency':>12} {'req/s':>8} {'tok/s':>10} {'p50':>8} {'p90':>8} {'p99':>8} {'fail':>6}")
    for c in args.concurrency:
        summary = await run_level(args.url, c, args.requests_per_level, args.max_new_tokens)
        all_results.append(summary)
        print(f"{summary['concurrency']:>12} "
              f"{summary['throughput_req_per_s']:>8.2f} "
              f"{summary['throughput_tokens_per_s']:>10.2f} "
              f"{(summary['p50_latency_s'] or 0):>8.2f} "
              f"{(summary['p90_latency_s'] or 0):>8.2f} "
              f"{(summary['p99_latency_s'] or 0):>8.2f} "
              f"{summary['n_failure']:>6}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
