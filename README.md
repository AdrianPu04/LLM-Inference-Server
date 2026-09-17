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

# Optional: CUDA / VRAM (and later vLLM) check
python scripts/check_env.py --skip-vllm
```

## Servers

| Server | Command | Behavior |
|--------|---------|----------|
| Naive | `uvicorn server.naive_server:app --host 0.0.0.0 --port 8000` | One request at a time (async lock) |
| Static batch | `uvicorn server.static_batch_server:app --host 0.0.0.0 --port 8000` | Collects requests for up to 30ms / batch of 8, one padded `generate()` |

Static batch knobs: `MAX_BATCH_SIZE` (default 8), `BATCH_TIMEOUT_MS` (default 30).

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Explain recursion in one paragraph.\", \"max_new_tokens\": 64}"
```

## Benchmark

```bash
python scripts/load_test.py \
  --url http://localhost:8000/generate \
  --concurrency 1 2 4 8 16 32 \
  --requests-per-level 20 \
  --out results/naive_baseline.json   # or results/static_batch.json
```

Committed results: `results/naive_baseline.json`, `results/static_batch.json`.
