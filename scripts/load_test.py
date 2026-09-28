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

    # vLLM / any OpenAI-compatible server (greedy, raw prompt — same work as /generate):
    python load_test.py --api openai --url http://localhost:8001/v1/completions \
        --model Qwen/Qwen2.5-1.5B-Instruct --out results/vllm.json

    # Every 8th request gets a ~1500-token prompt; short requests' latency is reported separately
    # (shows how much a long prefill stalls everyone else):
    python load_test.py --url http://localhost:8000/generate --concurrency 16 \
        --long-prompt-every 8 --long-prompt-words 1100 --out results/long_prompts.json
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

import httpx

PROMPTS = [
    "Explain the difference between a stack and a queue.",
    "Write a short story about a robot learning to paint.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What are the main causes of the French Revolution?",
    "Describe how photosynthesis works at a high level.",
]

PASSAGE = (
    "The history of computing spans mechanical calculators, vacuum tube machines, transistors, "
    "integrated circuits and the microprocessor, each generation making machines smaller, faster "
    "and cheaper while programming languages climbed from wiring panels to assembly to compilers. "
)


def long_prompt(words: int) -> str:
    base = PASSAGE.split()
    body = " ".join(base[j % len(base)] for j in range(words))
    # Unique leading nonce so no server can serve the prompt from a prefix cache (vLLM's is on by default).
    return f"Request {uuid.uuid4().hex}. Read the following text. {body}\nSummarize the text above in one paragraph."


def build_payload(api: str, model: str, prompt: str, max_new_tokens: int) -> dict:
    if api == "openai":
        return {"model": model, "prompt": prompt, "max_tokens": max_new_tokens, "temperature": 0}
    return {"prompt": prompt, "max_new_tokens": max_new_tokens}


def completion_tokens(api: str, data: dict) -> int:
    if api == "openai":
        return data.get("usage", {}).get("completion_tokens", 0)
    return data.get("completion_tokens", 0)


async def single_request(
    client: httpx.AsyncClient, url: str, api: str, model: str, prompt: str, max_new_tokens: int, long: bool
):
    payload = build_payload(api, model, prompt, max_new_tokens)
    start = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        wall_latency = time.perf_counter() - start
        return {
            "success": True,
            "long": long,
            "wall_latency_s": wall_latency,
            "completion_tokens": completion_tokens(api, data),
        }
    except Exception as e:
        wall_latency = time.perf_counter() - start
        return {"success": False, "long": long, "wall_latency_s": wall_latency, "error": str(e)}


async def run_level(
    url: str,
    api: str,
    model: str,
    concurrency: int,
    n_requests: int,
    max_new_tokens: list[int],
    long_every: int = 0,
    long_words: int = 0,
):
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(n_requests):
            long = long_every > 0 and i % long_every == long_every - 1
            prompt = long_prompt(long_words) if long else PROMPTS[i % len(PROMPTS)]
            tokens = max_new_tokens[i % len(max_new_tokens)]
            tasks.append(single_request(client, url, api, model, prompt, tokens, long))

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

    def pct(p, values=latencies):
        if not values:
            return None
        idx = min(int(len(values) * p), len(values) - 1)
        return values[idx]

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
        "api": api,
    }
    if long_every:
        short = sorted(r["wall_latency_s"] for r in successes if not r["long"])
        longs = sorted(r["wall_latency_s"] for r in successes if r["long"])
        summary.update({
            "long_prompt_every": long_every,
            "long_prompt_words": long_words,
            "short_p50_latency_s": pct(0.50, short),
            "short_p90_latency_s": pct(0.90, short),
            "short_p99_latency_s": pct(0.99, short),
            "long_p50_latency_s": pct(0.50, longs),
        })
    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--api",
        choices=["native", "openai"],
        default="native",
        help="native = this repo's POST /generate; openai = /v1/completions (vLLM etc.)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name sent in OpenAI requests (must match the served model)",
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        nargs="+",
        default=[128],
        help="One value = fixed length; several values cycle per request (mixed length)",
    )
    parser.add_argument(
        "--long-prompt-every",
        type=int,
        default=0,
        help="Make every Nth request a long prompt (0 = off); short-request latency is reported separately",
    )
    parser.add_argument("--long-prompt-words", type=int, default=1100, help="Words in each long prompt")
    parser.add_argument("--out", default=None, help="Optional path to dump JSON results")
    args = parser.parse_args()

    all_results = []
    print(f"api={args.api} max_new_tokens={args.max_new_tokens}")
    header = f"{'concurrency':>12} {'req/s':>8} {'tok/s':>10} {'p50':>8} {'p90':>8} {'p99':>8} {'fail':>6}"
    if args.long_prompt_every:
        header += f" {'short p50':>10} {'short p90':>10} {'long p50':>9}"
    print(header)
    for c in args.concurrency:
        summary = await run_level(
            args.url, args.api, args.model, c, args.requests_per_level, args.max_new_tokens,
            args.long_prompt_every, args.long_prompt_words,
        )
        all_results.append(summary)
        line = (f"{summary['concurrency']:>12} "
                f"{summary['throughput_req_per_s']:>8.2f} "
                f"{summary['throughput_tokens_per_s']:>10.2f} "
                f"{(summary['p50_latency_s'] or 0):>8.2f} "
                f"{(summary['p90_latency_s'] or 0):>8.2f} "
                f"{(summary['p99_latency_s'] or 0):>8.2f} "
                f"{summary['n_failure']:>6}")
        if args.long_prompt_every:
            line += (f" {(summary['short_p50_latency_s'] or 0):>10.2f}"
                     f" {(summary['short_p90_latency_s'] or 0):>10.2f}"
                     f" {(summary['long_p50_latency_s'] or 0):>9.2f}")
        print(line)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
