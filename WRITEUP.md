# Writeup: Building an LLM Inference Server

This project implements and benchmarks serving strategies for HuggingFace
causal LMs behind a shared FastAPI `POST /generate` API:

1. **Naive** — one request at a time  
2. **Static batching** — fixed batch for a whole `generate()` call  
3. **Continuous batching** — iteration-level scheduling with in-place KV  
4. **Quantized continuous** — same continuous scheduler with 4-bit weights (BnB NF4)

Hardware: NVIDIA RTX 4070 (12 GB). Primary model: `Qwen/Qwen2.5-1.5B-Instruct`
(fp16), plus BnB runs on 1.5B and `Qwen/Qwen2.5-7B-Instruct`.  
Load test: 20 requests per concurrency level via `scripts/load_test.py`
(typically `{1,2,4,8,16,32}`; 7B used up to 16).

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

## Phase 4 — Quantization (bitsandbytes NF4)

`server/quantized_server.py` reuses the continuous GPU-worker scheduler but loads
weights in 4-bit. The plan called for AWQ/GPTQ; on Windows `autoawq` could not
install (Triton dependency), so the default path is **bitsandbytes NF4** on the
full Instruct checkpoint at load time. AWQ/GPTQ remain wired for Linux/GKE.

BnB packs weights as NF4 (with double quantization); matmuls still run in fp16
after on-the-fly dequant. The KV cache is **not** 4-bit — VRAM savings are mostly
on weights.

### 1.5B: fp16 continuous vs BnB continuous

Same model size and scheduler; only the weight format changes.

| concurrency | fp16 req/s | BnB req/s | fp16 p50 | BnB p50 | fp16 p99 | BnB p99 |
|------------:|-----------:|----------:|---------:|--------:|---------:|--------:|
| 4 | 1.31 | 0.92 | 2.99 | 4.31 | 3.25 | 4.50 |
| 8 | **2.04** | 1.36 | 3.27 | 5.16 | 3.53 | 5.20 |
| 16 | 1.67 | 1.40 | 8.01 | 9.01 | 8.07 | 10.26 |
| 32 | 1.56 | 1.39 | 8.69 | 10.10 | 12.78 | 14.38 |

| | fp16 continuous | BnB 1.5B |
|--|-----------------|----------|
| Peak req/s | ~2.0 | ~1.4 |
| Peak tok/s | ~261 | ~179 |
| Load VRAM (approx.) | ~3 GiB weights | **~1.1 GiB** |

On a 1.5B model that already fit in fp16, 4-bit is a **memory win and a speed
loss** (~30% slower at the sweet spot) because of dequant overhead.

### 7B: what quantization is actually for

fp16 ~7B does not fit comfortably on 12 GB. With BnB:

```text
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct  QUANT_METHOD=bnb
peak_vram ≈ 5.30 GiB at load
```

Uniform load (`max_new_tokens=128`, concurrency through 16):

| concurrency | req/s | tok/s | p50 (s) | p99 (s) |
|------------:|------:|------:|--------:|--------:|
| 1 | 0.19 | 24.3 | 5.16 | 6.30 |
| 2 | 0.40 | 50.9 | 5.10 | 5.35 |
| 4 | 0.69 | 87.9 | 5.90 | 5.96 |
| 8 | **1.08** | **137.6** | 6.28 | 6.51 |
| 16 | 1.06 | 135.7 | 12.19 | 12.98 |

The 7B BnB server is slower than 1.5B fp16 (expected: more compute per token),
but it **runs at all** on the same card, peaking ~1.1 req/s with the continuous
scheduler intact and zero failures through c=16. That is the Phase 4 payoff:
quantization unlocks a larger model, not a faster tiny one.

---

## What I learned

1. **Batching dominates locking.** Naive → static was the largest single jump.  
2. **Scheduler shape ≠ free performance.** Continuous is only faster when the
   decode path is cheap enough; a Python-heavy per-token loop lost to static
   until the GPU-thread / in-place KV rewrite.  
3. **Benchmark design matters.** Uniform lengths hide continuous’s advantage;
   mixed lengths reveal it.  
4. **Platform constraints are part of the story.** Skipping broken Windows vLLM
   and preferring BnB over AWQ on this machine were the right calls.  
5. **Quantization is about fit, not free speed.** On 1.5B, BnB traded ~30%
   throughput for ~⅓ the weight VRAM; on 7B, the same path made a previously
   impractical model serveable (~5.3 GiB load).

---

## Next steps

- **GKE / Linux:** deploy and add a real vLLM (+ optional AWQ) baseline on
  identical hardware.  
- Optional: latency broken out by length bucket for mixed tests; longer 7B
  concurrency sweeps if VRAM allows.
