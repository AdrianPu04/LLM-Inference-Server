"""
Phase 1: Naive baseline.

One request at a time, no batching, plain HuggingFace .generate().
This is deliberately bad — it's the control group every later phase
(static batching, continuous batching, quantization) gets measured against.

Run:
    uvicorn naive_server:app --host 0.0.0.0 --port 8000

Model default is small (fits comfortably on a 12-16GB 4070). Override with
the MODEL_NAME env var once you've confirmed the pipeline works end to end.
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    state["tokenizer"] = tokenizer
    state["model"] = model
    state["lock"] = asyncio.Lock()
    print("Model loaded.")
    yield
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
    tokenizer = state["tokenizer"]
    model = state["model"]

    def _blocking_generate():
        inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = output_ids[0][prompt_tokens:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        return prompt_tokens, completion_ids, text

    async with state["lock"]:  
        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        prompt_tokens, completion_ids, text = await loop.run_in_executor(
            None, _blocking_generate
        )
        latency = time.perf_counter() - start

    return GenerateResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=len(completion_ids),
        latency_s=latency,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}