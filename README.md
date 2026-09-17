# LLM Inference Server

A minimal HuggingFace inference server with a load test harness for measuring throughput and latency under concurrency.

Default model: `Qwen/Qwen2.5-1.5B-Instruct` (fits a 12–16GB GPU). Override with `MODEL_NAME`.

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run the server

```bash
uvicorn server.naive_server:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Explain recursion in one paragraph.\", \"max_new_tokens\": 64}"
```

One request is served at a time (async lock around `.generate()`).

## Benchmark

```bash
python scripts/load_test.py \
  --url http://localhost:8000/generate \
  --concurrency 1 2 4 8 16 32 \
  --requests-per-level 20 \
  --out results/naive_baseline.json
```

Results are written to `results/`.
