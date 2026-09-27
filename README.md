# LLM Inference Server

HuggingFace inference servers with a shared `POST /generate` API and a load-test harness for throughput/latency under concurrency.

Design notes and benchmark discussion: [WRITEUP.md](WRITEUP.md).

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
| Quantized continuous | `uvicorn server.quantized_server:app --host 0.0.0.0 --port 8000` | Same continuous scheduler with AWQ / BnB / GPTQ 4-bit weights |

| Knob | Default | Applies to |
|------|---------|------------|
| `MAX_BATCH_SIZE` | `8` | static, continuous, quantized |
| `BATCH_TIMEOUT_MS` | `30` | static |
| `DECODE_BURST` | `64` | continuous, quantized |
| `TORCH_COMPILE` | `0` | continuous, quantized |
| `MAX_SEQ_LEN` | `2048` | continuous, quantized |
| `QUANT_METHOD` | `bnb` | quantized (`bnb` \| `awq` \| `gptq`) |
| `MODEL_NAME` | (see below) | all |

Quantized defaults: `bnb` → base Instruct + NF4 (Windows-friendly); `awq` / `gptq` → pre-quantized HF checkpoints (typically Linux; `autoawq` needs Triton).

```powershell
pip install bitsandbytes
uvicorn server.quantized_server:app --host 0.0.0.0 --port 8000

# On Linux, AWQ checkpoint instead:
# pip install autoawq
# $env:QUANT_METHOD="awq"
# uvicorn server.quantized_server:app --host 0.0.0.0 --port 8000
```

### Docker

One image serves every server; pick one with `SERVER` (module name under `server/`). Requires the NVIDIA driver plus NVIDIA Container Toolkit (on Windows: Docker Desktop with the WSL2 backend).

```bash
docker build -t llm-inference-server .

# Continuous batching (default). The hf-cache volume keeps model downloads across runs.
docker run --gpus all -p 8000:8000 -v hf-cache:/models llm-inference-server

# Any other server / knob via env vars:
docker run --gpus all -p 8000:8000 -v hf-cache:/models \
  -e SERVER=quantized_server -e MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
  llm-inference-server
```

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

# Quantized continuous (compare to continuous_batch.json)
python scripts/load_test.py \
  --url http://localhost:8000/generate \
  --concurrency 1 2 4 8 16 32 \
  --requests-per-level 20 \
  --out results/quantized_bnb.json

# vLLM baseline (Docker, port 8001) via its OpenAI-compatible API
docker run -d --name vllm --gpus all -p 8001:8000 --ipc=host \
  -v hf-cache:/root/.cache/huggingface vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.75 --max-model-len 2048
python scripts/load_test.py --api openai \
  --url http://localhost:8001/v1/completions \
  --concurrency 1 2 4 8 16 32 \
  --out results/vllm_docker.json
```

## Results

| File | Server / notes |
|------|----------------|
| `results/naive_baseline.json` | Naive, fixed 128 tokens |
| `results/static_batch.json` | Static, fixed 128 |
| `results/continuous_batch.json` | Continuous fp16 1.5B, fixed 128 |
| `results/static_mixed.json` | Static, mixed 32/128/256 |
| `results/continuous_mixed.json` | Continuous fp16 1.5B, mixed 32/128/256 |
| `results/quantized_bnb.json` | Continuous BnB NF4 1.5B, fixed 128 |
| `results/quantized_bnb_7b.json` | Continuous BnB NF4 7B, fixed 128 (c≤16) |
| `results/continuous_docker.json` | Continuous fp16 1.5B in Docker, fixed 128 |
| `results/continuous_docker_mixed.json` | Continuous fp16 1.5B in Docker, mixed 32/128/256 |
| `results/vllm_docker.json` | vLLM (default), fixed 128 |
| `results/vllm_docker_mixed.json` | vLLM (default), mixed 32/128/256 |
| `results/vllm_docker_seqs8.json` | vLLM `--max-num-seqs 8`, fixed 128 |
| `results/vllm_docker_seqs8_mixed.json` | vLLM `--max-num-seqs 8`, mixed 32/128/256 |

Rows above the Docker entries ran natively on Windows; Docker rows ran on the same GPU via WSL2.

**Headline findings (4070):**
- Naive throughput stays ~flat (~0.37 req/s) while p99 climbs with concurrency.
- Static batching raises peak throughput to ~2.3 req/s on uniform 1.5B fp16.
- Continuous matches/beats static around moderate concurrency on uniform load, and **clearly wins on mixed lengths** (~1.8–1.9× static req/s, much lower p50).
- BnB 4-bit on 1.5B cuts weight VRAM (~1.1 vs ~3 GiB) but is ~30% slower than fp16 continuous.
- BnB enables **Qwen2.5-7B-Instruct** on the same 12 GB card (~5.3 GiB load, ~1.1 req/s peak) — the real quantization win.
- In Docker, continuous holds ~2.3 req/s flat past c=8 (Windows sagged to ~1.6).
- vLLM is ~2.3× faster at the same batch cap of 8, and reaches ~15 req/s at c=32 uncapped.
