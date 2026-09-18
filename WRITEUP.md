# Writeup: Building an LLM Inference Server

This project implements and benchmarks three serving strategies for the same
HuggingFace causal LM behind a shared FastAPI `POST /generate` API:

1. **Naive** — one request at a time  
2. **Static batching** — fixed batch for a whole `generate()` call  
3. **Continuous batching** — iteration-level scheduling with in-place KV  

Hardware: NVIDIA RTX 4070 (12 GB). Model: `Qwen/Qwen2.5-1.5B-Instruct` (fp16).  
Load test: 20 requests per concurrency level `{1,2,4,8,16,32}` via `scripts/load_test.py`.

Raw JSON: `results/`. How to run: [README.md](README.md).

---

## Motivation

Serving LLMs is less about “call `.generate()`” and more about **how the GPU is
kept busy** when many clients arrive at once. A lock around a single generate
call is easy to write and easy to measure — and it produces a collapse curve
that motivates better schedulers.

The goal was to climb that ladder on consumer hardware, measure honestly, and
decide when to stop chasing infrastructure (vLLM) vs build it myself.

---

## Environment decision: no vLLM on Windows

`scripts/check_env.py` confirmed CUDA and VRAM, then failed on:

```text
No module named 'vllm._C_stable_libtorch'
```

The installed Windows wheel lacked the native CUDA extension. vLLM is primarily
a Linux stack; rather than sink days into WSL/driver archaeology, Phase 3 used a
**PyTorch-native** continuous batcher. That is a deliberate engineering call, not
a downgrade — more of the scheduler is our code. A fair vLLM baseline belongs on
Linux/GKE later.

---

## Phase 1 — Naive baseline

`server/naive_server.py` loads the model once, then serves each request under an
`asyncio.Lock` with `model.generate()` in a thread pool. Concurrent HTTP
requests queue; the GPU never sees more than one sequence at a time.

| concurrency | req/s | p50 (s) | p99 (s) |
|------------:|------:|--------:|--------:|
| 1 | 0.36 | 2.95 | 3.65 |
| 4 | 0.38 | 10.26 | 11.71 |
| 8 | 0.37 | 20.49 | 22.90 |
| 16 | 0.38 | 28.19 | 42.78 |
| 32 | 0.25 | 38.65 | 80.08 |

Throughput is flat; tail latency grows roughly with queue depth. That is the
control curve everything else is measured against.

---

## Phase 2 — Static batching

`server/static_batch_server.py` enqueues requests as futures, waits up to
`BATCH_TIMEOUT_MS` (or `MAX_BATCH_SIZE`), left-pads prompts, and runs **one**
`model.generate()` for the batch.

| concurrency | req/s | p50 (s) | p99 (s) |
|------------:|------:|--------:|--------:|
| 1 | 0.33 | 3.12 | 3.97 |
| 4 | 1.23 | 3.10 | 3.79 |
| 8 | 1.60 | 3.53 | 5.91 |
| 16 | 2.27 | 5.70 | 6.00 |
| 32 | 2.27 | 5.95 | 8.79 |

Peak throughput jumps to ~**2.3 req/s** (~6× naive). At concurrency 1, static is
slightly worse (timeout wait with nothing to batch) — expected.

**Limitation:** the batch is frozen for the whole generate. Short completions
stuck with long ones cannot leave until the call finishes. Under uniform
`max_new_tokens=128` that barely shows; under mixed lengths it does.

---

## Phase 3 — Continuous batching

`server/continuous_batch_server.py` runs a **GPU worker thread** that:

1. Prefills new admits into the running set  
2. Decodes one token for everyone with a **shared in-place batched KV cache**  
3. Drops finished sequences and admits newcomers between steps  

`DECODE_BURST` lets many decode steps run before polling the queue again, so
asyncio is not on the per-token path. Early versions that split/stacked KV every
step (or gather/scattered a paged pool in Python) were much slower than static;
the in-place cache + worker thread closed that gap.

### Uniform load (`max_new_tokens=128`)

| concurrency | static req/s | continuous req/s | static p99 | continuous p99 |
|------------:|-------------:|-----------------:|-----------:|---------------:|
| 4 | 1.23 | 1.31 | 3.79 | 3.25 |
| 8 | 1.60 | **2.04** | 5.91 | **3.53** |
| 16 | **2.27** | 1.67 | **6.00** | 8.07 |
| 32 | **2.27** | 1.56 | **8.79** | 12.78 |

On uniform traffic, continuous is competitive and even leads around concurrency 8;
static’s fused `generate()` still edges peak throughput at 16–32.

### Mixed load (`max_new_tokens` cycling 32 / 128 / 256)

This is where iteration-level scheduling should win.

| concurrency | static req/s | continuous req/s | static p50 | continuous p50 | static p99 | continuous p99 |
|------------:|-------------:|-----------------:|-----------:|---------------:|-----------:|---------------:|
| 8 | 0.99 | **1.79** | 6.74 | **3.20** | 6.82 | 6.75 |
| 16 | 1.01 | **1.79** | 12.74 | **6.35** | 13.47 | **9.91** |
| 32 | 1.02 | **1.90** | 13.26 | **6.80** | 19.59 | **10.50** |

Continuous delivers roughly **1.8–1.9×** static request throughput and about
**half** the median latency. Short requests leave when they hit their token cap;
static holds the whole batch inside one `generate()` (and uses `max()` of the
batch’s `max_new_tokens`, which further couples shorts to longs).

---

## What I learned

1. **Batching dominates locking.** Naive → static was the largest single jump.  
2. **Scheduler shape ≠ free performance.** Continuous is only faster when the
   decode path is cheap enough; a Python-heavy per-token loop lost to static
   until the GPU-thread / in-place KV rewrite.  
3. **Benchmark design matters.** Uniform lengths hide continuous’s advantage;
   mixed lengths reveal it.  
4. **Platform constraints are part of the story.** Skipping broken Windows vLLM
   and shipping a custom scheduler was the right call for this machine.

---

## Next steps

- **Quantization (AWQ/GPTQ 4-bit):** same API and harness; compare VRAM and
  throughput, optionally larger models on 12 GB.  
- **GKE / Linux:** deploy and add a real vLLM baseline on identical hardware.  
- Optional: latency broken out by length bucket for mixed tests.
