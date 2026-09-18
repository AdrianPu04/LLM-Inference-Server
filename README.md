# LLM Inference Server

HuggingFace inference servers with a shared `POST /generate` API and a load-test harness for throughput/latency under concurrency.

Default model: `Qwen/Qwen2.5-1.5B-Instruct` (fits a 12–16GB GPU). Override with `MODEL_NAME`.

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/check_env.py --skip-vllm
```

vLLM does not import cleanly on this Windows setup (`vllm._C_stable_libtorch` missing); servers are PyTorch-native instead.

## Servers

| Server | Command | Behavior |
|--------|---------|----------|
| Naive | `uvicorn server.naive_server:app --host 0.0.0.0 --port 8000` | One request at a time (async lock around `.generate()`) |
| Static batch | `uvicorn server.static_batch_server:app --host 0.0.0.0 --port 8000` | Collect requests for up to 30ms / batch of 8 → one padded `generate()` |
| Continuous batch | `uvicorn server.continuous_batch_server:app --host 0.0.0.0 --port 8000` | Iteration-level scheduling on a GPU worker thread; in-place batched KV |

| Knob | Default | Applies to |
|------|---------|------------|
| `MAX_BATCH_SIZE` | `8` | static, continuous |
| `BATCH_TIMEOUT_MS` | `30` | static |
| `DECODE_BURST` | `64` | continuous |
| `TORCH_COMPILE` | `0` | continuous |
| `MAX_SEQ_LEN` | `2048` | continuous |

Continuous keeps a batched KV cache across decode steps (rebuilds only when requests join/leave) so asyncio is not on the per-token hot path.

### API

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Explain recursion in one paragraph.\", \"max_new_tokens\": 64}"
```

`GET /health` reports model and server-specific knobs.

## Benchmark

```bash
# Fixed completion length
python scripts/load_test.py \
  --url http://localhost:8000/generate \
  --concurrency 1 2 4 8 16 32 \
  --requests-per-level 20 \
  --out results/continuous_batch.json

# Mixed lengths (cycles 32 / 128 / 256 across requests)
python scripts/load_test.py \
  --url http://localhost:8000/generate \
  --concurrency 1 2 4 8 16 32 \
  --max-new-tokens 32 128 256 \
  --out results/continuous_mixed.json
```

## Results

| File | Server / notes |
|------|----------------|
| `results/naive_baseline.json` | Naive, fixed 128 tokens |
| `results/static_batch.json` | Static, fixed 128 |
| `results/continuous_batch.json` | Continuous, fixed 128 |
| `results/static_mixed.json` | Static, mixed 32/128/256 |
| `results/continuous_mixed.json` | Continuous, mixed 32/128/256 |

**Headline findings (4070, Qwen2.5-1.5B):**
- Naive throughput stays ~flat (~0.37 req/s) while p99 climbs with concurrency.
- Static batching raises peak throughput to ~2.3 req/s on uniform load.
- Continuous matches/beats static around moderate concurrency on uniform load, and **clearly wins on mixed lengths** (~1.8–1.9× static req/s, much lower p50) because short requests can leave mid-decode.
