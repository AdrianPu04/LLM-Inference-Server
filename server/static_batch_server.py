"""
Static batching server.

Collects concurrent /generate requests over a short window (or until
max batch size), pads them, and runs a single model.generate() call.

Run:
    uvicorn server.static_batch_server:app --host 0.0.0.0 --port 8000

Knobs (env):
    MODEL_NAME          default Qwen/Qwen2.5-1.5B-Instruct
    MAX_BATCH_SIZE      default 32
    BATCH_TIMEOUT_MS    default 30
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))
BATCH_TIMEOUT_S = int(os.environ.get("BATCH_TIMEOUT_MS", "30")) / 1000.0

state: dict = {}


@dataclass
class PendingRequest:
    req: "GenerateRequest"
    future: asyncio.Future
    enqueued_at: float


def _run_batch(batch: list[PendingRequest]) -> list[tuple[int, list[int], str]]:
    """Blocking: tokenize, generate, unpack one static batch."""
    tokenizer = state["tokenizer"]
    model = state["model"]

    prompts = [item.req.prompt for item in batch]
    max_new_tokens = max(item.req.max_new_tokens for item in batch)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    padded_len = inputs["input_ids"].shape[1]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i in range(len(batch)):
        prompt_tokens = int(attention_mask[i].sum().item())
        completion_ids = output_ids[i][padded_len:].tolist()
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        results.append((prompt_tokens, completion_ids, text))
    return results


async def _collect_batch(queue: asyncio.Queue) -> list[PendingRequest]:
    """Wait for the first request, then gather until size or timeout."""
    first = await queue.get()
    batch = [first]
    deadline = time.perf_counter() + BATCH_TIMEOUT_S

    while len(batch) < MAX_BATCH_SIZE:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        try:
            item = await asyncio.wait_for(queue.get(), timeout=remaining)
            batch.append(item)
        except asyncio.TimeoutError:
            break
    return batch


async def batch_worker() -> None:
    queue: asyncio.Queue = state["queue"]
    loop = asyncio.get_running_loop()

    while True:
        batch = await _collect_batch(queue)
        try:
            results = await loop.run_in_executor(None, _run_batch, batch)
            for item, (prompt_tokens, completion_ids, text) in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(
                        {
                            "text": text,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": len(completion_ids),
                        }
                    )
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    state["tokenizer"] = tokenizer
    state["model"] = model
    state["queue"] = asyncio.Queue()
    state["worker"] = asyncio.create_task(batch_worker())
    print(
        f"Model loaded. max_batch_size={MAX_BATCH_SIZE} "
        f"batch_timeout_ms={int(BATCH_TIMEOUT_S * 1000)}"
    )
    yield

    state["worker"].cancel()
    try:
        await state["worker"]
    except asyncio.CancelledError:
        pass
    state.clear()


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    await state["queue"].put(PendingRequest(req=req, future=future, enqueued_at=start))
    result = await future
    latency = time.perf_counter() - start

    return GenerateResponse(
        text=result["text"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        latency_s=latency,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "max_batch_size": MAX_BATCH_SIZE,
        "batch_timeout_ms": int(BATCH_TIMEOUT_S * 1000),
    }
