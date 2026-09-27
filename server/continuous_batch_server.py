"""
Continuous batching server (PyTorch-native; no vLLM).

Iteration-level scheduling on a dedicated GPU worker thread:
  - In-place batched KV (one DynamicCache for the running set;
    rebuild only when sequences join/leave)
  - Decode burst before re-checking the admit queue
  - SDPA attention; optional torch.compile (TORCH_COMPILE=1)

Run:
    uvicorn server.continuous_batch_server:app --host 0.0.0.0 --port 8000

Knobs (env):
    MODEL_NAME       default Qwen/Qwen2.5-1.5B-Instruct
    MAX_BATCH_SIZE   default 32
    MAX_SEQ_LEN      default 2048
    DECODE_BURST     default 64
    TORCH_COMPILE    default 0
"""

from __future__ import annotations

import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "2048"))
DECODE_BURST = int(os.environ.get("DECODE_BURST", "64"))
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "0").strip().lower() in {"1", "true", "yes", "on"}

state: dict = {}


@dataclass
class Sequence:
    prompt: str
    max_new_tokens: int
    future: object  # asyncio.Future
    loop: object  # asyncio event loop for thread-safe completion
    enqueued_at: float
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_ids: list[int] = field(default_factory=list)
    past_key_values: object | None = None
    past_len: int = 0
    next_token: int | None = None
    done: bool = False


def _cache_from_kv(kv_layers: list[tuple[torch.Tensor, torch.Tensor]]):
    from transformers import DynamicCache

    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_layers):
        cache.update(k, v, layer_idx)
    return cache


def _pad_cache(cache, target_len: int):
    cur = cache.layers[0].keys.shape[2]
    if cur == target_len:
        return cache
    if cur > target_len:
        raise ValueError(f"cache len {cur} > target {target_len}")
    pad = target_len - cur
    kv = []
    for layer in cache.layers:
        kv.append(
            (
                F.pad(layer.keys, (0, 0, pad, 0)),
                F.pad(layer.values, (0, 0, pad, 0)),
            )
        )
    return _cache_from_kv(kv)


def _cat_caches(caches: list):
    n_layers = len(caches[0].layers)
    kv = []
    for layer_idx in range(n_layers):
        keys = torch.cat([c.layers[layer_idx].keys for c in caches], dim=0)
        values = torch.cat([c.layers[layer_idx].values for c in caches], dim=0)
        kv.append((keys, values))
    return _cache_from_kv(kv)


def _index_cache(cache, indices: list[int]):
    idx = torch.tensor(indices, device=cache.layers[0].keys.device, dtype=torch.long)
    kv = []
    for layer in cache.layers:
        kv.append((layer.keys.index_select(0, idx), layer.values.index_select(0, idx)))
    return _cache_from_kv(kv)


def _complete_ok(seq: Sequence) -> None:
    seq.past_key_values = None
    tokenizer = state["tokenizer"]
    ids = list(seq.generated_ids)
    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]
    payload = {
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "prompt_tokens": len(seq.prompt_token_ids),
        "completion_tokens": len(ids),
    }

    def _set():
        if not seq.future.done():
            seq.future.set_result(payload)

    seq.loop.call_soon_threadsafe(_set)


def _complete_err(seq: Sequence, exc: BaseException) -> None:
    seq.past_key_values = None

    def _set():
        if not seq.future.done():
            seq.future.set_exception(exc)

    seq.loop.call_soon_threadsafe(_set)


def _prefill(seq: Sequence) -> None:
    tokenizer = state["tokenizer"]
    model = state["model"]
    device = model.device

    input_ids = tokenizer(seq.prompt, return_tensors="pt")["input_ids"].to(device)
    seq.prompt_token_ids = input_ids[0].tolist()
    if input_ids.shape[1] + seq.max_new_tokens > MAX_SEQ_LEN:
        raise ValueError(
            f"prompt ({input_ids.shape[1]}) + max_new_tokens ({seq.max_new_tokens}) "
            f"exceeds MAX_SEQ_LEN ({MAX_SEQ_LEN})"
        )

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
        )

    seq.past_key_values = out.past_key_values
    seq.past_len = input_ids.shape[1]
    seq.next_token = int(out.logits[0, -1].argmax(dim=-1).item())
    seq.generated_ids = [seq.next_token]
    if seq.next_token == tokenizer.eos_token_id or len(seq.generated_ids) >= seq.max_new_tokens:
        seq.done = True


class RunningBatch:
    """In-flight sequences sharing one batched KV cache."""

    def __init__(self):
        self.seqs: list[Sequence] = []
        self.past = None
        self.tokens: torch.Tensor | None = None
        self.past_lens: list[int] = []

    def __len__(self) -> int:
        return len(self.seqs)

    @property
    def empty(self) -> bool:
        return not self.seqs

    def start(self, seq: Sequence) -> None:
        device = state["model"].device
        self.seqs = [seq]
        self.past = seq.past_key_values
        seq.past_key_values = None
        self.past_lens = [seq.past_len]
        self.tokens = torch.tensor([[seq.next_token]], dtype=torch.long, device=device)

    def add(self, seq: Sequence) -> None:
        device = state["model"].device
        tensor_len = self.past.layers[0].keys.shape[2]
        target = max(tensor_len, seq.past_len)
        if tensor_len < target:
            self.past = _pad_cache(self.past, target)
        new_cache = _pad_cache(seq.past_key_values, target)
        seq.past_key_values = None
        self.past = _cat_caches([self.past, new_cache])
        self.seqs.append(seq)
        self.past_lens.append(seq.past_len)
        new_tok = torch.tensor([[seq.next_token]], dtype=torch.long, device=device)
        self.tokens = torch.cat([self.tokens, new_tok], dim=0)

    def decode_step(self) -> None:
        model = state["model"]
        eos_id = state["tokenizer"].eos_token_id
        device = model.device
        b = len(self.seqs)
        max_past = self.past.layers[0].keys.shape[2]

        attn = torch.zeros(b, max_past + 1, dtype=torch.long, device=device)
        pos = torch.empty(b, 1, dtype=torch.long, device=device)
        for i, plen in enumerate(self.past_lens):
            attn[i, max_past - plen :] = 1
            attn[i, -1] = 1
            pos[i, 0] = plen

        with torch.no_grad():
            out = model(
                input_ids=self.tokens,
                attention_mask=attn,
                position_ids=pos,
                past_key_values=self.past,
                use_cache=True,
            )

        self.past = out.past_key_values
        next_tokens = out.logits[:, -1, :].argmax(dim=-1)
        self.tokens = next_tokens.unsqueeze(1)
        next_list = next_tokens.tolist()

        for i, seq in enumerate(self.seqs):
            self.past_lens[i] += 1
            seq.past_len = self.past_lens[i]
            tok = int(next_list[i])
            seq.next_token = tok
            seq.generated_ids.append(tok)
            if tok == eos_id or len(seq.generated_ids) >= seq.max_new_tokens:
                seq.done = True

    def pop_finished(self) -> list[Sequence]:
        if not any(s.done for s in self.seqs):
            return []
        finished = []
        keep_idx = []
        keep_seqs = []
        keep_lens = []
        for i, seq in enumerate(self.seqs):
            if seq.done:
                finished.append(seq)
            else:
                keep_idx.append(i)
                keep_seqs.append(seq)
                keep_lens.append(self.past_lens[i])
        if not keep_idx:
            self.seqs = []
            self.past = None
            self.tokens = None
            self.past_lens = []
            return finished

        self.past = _index_cache(self.past, keep_idx)
        self.tokens = self.tokens[keep_idx]
        self.seqs = keep_seqs
        self.past_lens = keep_lens
        return finished


def _gpu_worker(stop_event: threading.Event) -> None:
    incoming: queue.Queue = state["incoming"]
    batch = RunningBatch()

    def admit_one(seq: Sequence) -> Sequence | None:
        if seq.future.done():
            return None
        try:
            _prefill(seq)
        except Exception as exc:
            _complete_err(seq, exc)
            return None
        if seq.done:
            _complete_ok(seq)
            return None
        return seq

    while not stop_event.is_set():
        try:
            if batch.empty:
                try:
                    item = incoming.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    break
                admitted = admit_one(item)
                if admitted is not None:
                    batch.start(admitted)
                continue

            while len(batch) < MAX_BATCH_SIZE:
                try:
                    item = incoming.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    stop_event.set()
                    break
                admitted = admit_one(item)
                if admitted is not None:
                    batch.add(admitted)

            for _ in range(DECODE_BURST):
                if batch.empty:
                    break
                batch.decode_step()
                for seq in batch.pop_finished():
                    _complete_ok(seq)
                if not incoming.empty() and len(batch) < MAX_BATCH_SIZE:
                    break
        except Exception as exc:
            print(f"[gpu_worker] error: {exc!r}")
            for seq in list(batch.seqs):
                _complete_err(seq, exc)
            batch = RunningBatch()


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
        attn_implementation="sdpa",
    )
    model.eval()
    if TORCH_COMPILE:
        print("torch.compile enabled (first requests will be slow)...")
        model = torch.compile(model, mode="reduce-overhead")

    state["tokenizer"] = tokenizer
    state["model"] = model
    state["incoming"] = queue.Queue()
    stop_event = threading.Event()
    state["stop_event"] = stop_event
    worker = threading.Thread(target=_gpu_worker, args=(stop_event,), name="gpu-worker", daemon=True)
    state["worker"] = worker
    worker.start()

    print(
        f"Model loaded. continuous | batch={MAX_BATCH_SIZE} "
        f"decode_burst={DECODE_BURST} compile={TORCH_COMPILE}"
    )
    yield

    stop_event.set()
    state["incoming"].put(None)
    worker.join(timeout=30)
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
    import asyncio

    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    seq = Sequence(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        future=future,
        loop=loop,
        enqueued_at=start,
    )
    state["incoming"].put(seq)
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
        "scheduler": "continuous_gpu_thread",
        "decode_burst": DECODE_BURST,
        "torch_compile": TORCH_COMPILE,
    }
