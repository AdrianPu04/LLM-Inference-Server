"""
Continuous batching with a static, slot-based KV cache and CUDA-graph decode.

Same scheduler and API as continuous_batch_server.py, but the decode step has
fixed shapes so it can be captured once and replayed:
  - KV cache pre-allocated as [MAX_BATCH_SIZE, kv_heads, MAX_SEQ_LEN, head_dim]
    per layer; each sequence owns one slot (row) until it finishes
  - Decode forward written against the model's own modules (no DynamicCache),
    writing each new K/V at its slot's position and masking the rest
  - One CUDA graph per (batch bucket, length bucket); a step replays the
    smallest graph covering the occupied slots and longest active sequence
  - Prefill stays eager (variable prompt length) and is copied into the slot

Supports Llama-style decoders (Qwen2, Llama, Mistral): embed -> N x
[norm, attn with RoPE, norm, MLP] -> norm -> lm_head.

Run:
    uvicorn server.cuda_graph_server:app --host 0.0.0.0 --port 8000

Knobs (env):
    MODEL_NAME       default Qwen/Qwen2.5-1.5B-Instruct
    MAX_BATCH_SIZE   default 32
    MAX_SEQ_LEN      default 2048
    DECODE_BURST     default 64
    CUDA_GRAPHS      default 1 (0 = run the same static decode eagerly)
"""

from __future__ import annotations

import asyncio
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
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "2048"))
DECODE_BURST = int(os.environ.get("DECODE_BURST", "64"))
CUDA_GRAPHS = os.environ.get("CUDA_GRAPHS", "1").strip().lower() in {"1", "true", "yes", "on"}

state: dict = {}


def _buckets(limit: int, start: int) -> list[int]:
    out, b = [], start
    while b < limit:
        out.append(b)
        b *= 2
    out.append(limit)
    return out


def _pick(buckets: list[int], need: int) -> int:
    for b in buckets:
        if b >= need:
            return b
    raise ValueError(f"need {need} exceeds largest bucket {buckets[-1]}")


@dataclass
class Sequence:
    prompt: str
    max_new_tokens: int
    future: object
    loop: object
    enqueued_at: float
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_ids: list[int] = field(default_factory=list)
    pos: int = 0  # position of the token fed at the next decode step
    slot: int = -1
    slot_kv: object | None = None  # prefill DynamicCache, held until copied into the slot
    first_token: int = 0
    done: bool = False


def _complete_ok(seq: Sequence) -> None:
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
    def _set():
        if not seq.future.done():
            seq.future.set_exception(exc)

    seq.loop.call_soon_threadsafe(_set)


class StaticDecoder:
    """Static KV cache + fixed-shape decode step, optionally CUDA-graphed."""

    def __init__(self, model, max_batch: int, max_len: int):
        cfg = model.config
        self.model = model
        self.inner = model.model
        self.device = model.device
        self.dtype = model.dtype
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.group = self.n_heads // self.n_kv
        self.max_batch = max_batch
        self.max_len = max_len

        shape = (max_batch, self.n_kv, max_len, self.head_dim)
        n_layers = len(self.inner.layers)
        self.k_cache = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(n_layers)]
        self.v_cache = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(n_layers)]

        # Graph inputs/state: must keep fixed addresses, so only ever written in place.
        self.tokens = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self.pos = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self.active = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self.arange_b = torch.arange(max_batch, device=self.device)
        self.arange_l = torch.arange(max_len, device=self.device)

        self.batch_buckets = _buckets(max_batch, 1)
        self.len_buckets = _buckets(max_len, min(256, max_len))
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.graph_out: dict[tuple[int, int], torch.Tensor] = {}

    def _step(self, b: int, length: int) -> torch.Tensor:
        """One decode token for slots [0, b), attending over cache positions [0, length)."""
        tokens = self.tokens[:b]
        pos = self.pos[:b]
        rows = self.arange_b[:b]

        h = self.inner.embed_tokens(tokens).unsqueeze(1)  # [b, 1, hidden]
        cos, sin = self.inner.rotary_emb(h, pos.unsqueeze(1))
        # Stale K/V from earlier occupants sits past each slot's position; mask it out.
        mask = (self.arange_l[:length].unsqueeze(0) <= pos.unsqueeze(1))[:, None, None, :]

        for i, layer in enumerate(self.inner.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(b, 1, self.n_kv, self.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(b, 1, self.n_kv, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            self.k_cache[i][rows, :, pos] = k[:, :, 0]
            self.v_cache[i][rows, :, pos] = v[:, :, 0]
            keys = self.k_cache[i][:b, :, :length]
            values = self.v_cache[i][:b, :, :length]

            # GQA without copying K/V: fold each KV head's query group into the query-length dim.
            # (SDPA's enable_gqa=True with a mask falls back to a ~6x slower kernel.)
            q = q.reshape(b, self.n_kv, self.group, self.head_dim)
            o = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
            h = h + attn.o_proj(o.reshape(b, 1, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        logits = self.model.lm_head(self.inner.norm(h))
        next_tokens = logits[:, -1, :].argmax(dim=-1)

        # Feed back for the next step without leaving the GPU; idle slots stay at pos 0.
        self.tokens[:b].copy_(next_tokens)
        self.pos[:b].add_(self.active[:b])
        return next_tokens

    @torch.no_grad()
    def capture(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        # Largest first so smaller graphs can reuse its memory from the shared pool.
        for b in reversed(self.batch_buckets):
            for length in reversed(self.len_buckets):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self._step(b, length)
                torch.cuda.current_stream().wait_stream(stream)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    out = self._step(b, length)
                self.graphs[(b, length)] = graph
                self.graph_out[(b, length)] = out
        self.tokens.zero_()
        self.pos.zero_()
        torch.cuda.synchronize()

    @torch.no_grad()
    def load_slot(self, slot: int, past_key_values, prompt_len: int, next_token: int) -> None:
        for i, layer in enumerate(past_key_values.layers):
            self.k_cache[i][slot, :, :prompt_len].copy_(layer.keys[0])
            self.v_cache[i][slot, :, :prompt_len].copy_(layer.values[0])
        self.tokens[slot] = next_token
        self.pos[slot] = prompt_len
        self.active[slot] = 1

    def free_slot(self, slot: int) -> None:
        self.active[slot] = 0
        self.pos[slot] = 0

    @torch.no_grad()
    def decode(self, b_need: int, len_need: int) -> list[int]:
        b = _pick(self.batch_buckets, b_need)
        length = _pick(self.len_buckets, len_need)
        if CUDA_GRAPHS:
            self.graphs[(b, length)].replay()
            out = self.graph_out[(b, length)]
        else:
            out = self._step(b, length)
        return out.tolist()


def _prefill(seq: Sequence) -> None:
    tokenizer = state["tokenizer"]
    model = state["model"]

    input_ids = tokenizer(seq.prompt, return_tensors="pt")["input_ids"].to(model.device)
    seq.prompt_token_ids = input_ids[0].tolist()
    if input_ids.shape[1] + seq.max_new_tokens > MAX_SEQ_LEN:
        raise ValueError(
            f"prompt ({input_ids.shape[1]}) + max_new_tokens ({seq.max_new_tokens}) "
            f"exceeds MAX_SEQ_LEN ({MAX_SEQ_LEN})"
        )

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)

    next_token = int(out.logits[0, -1].argmax(dim=-1).item())
    seq.generated_ids = [next_token]
    seq.pos = input_ids.shape[1]
    if next_token == tokenizer.eos_token_id or seq.max_new_tokens <= 1:
        seq.done = True
        return
    seq.slot_kv = out.past_key_values
    seq.first_token = next_token


class SlotBatch:
    def __init__(self, decoder: StaticDecoder):
        self.dec = decoder
        self.slots: list[Sequence | None] = [None] * decoder.max_batch

    def __len__(self) -> int:
        return sum(s is not None for s in self.slots)

    @property
    def empty(self) -> bool:
        return all(s is None for s in self.slots)

    def add(self, seq: Sequence) -> None:
        slot = self.slots.index(None)
        seq.slot = slot
        self.slots[slot] = seq
        self.dec.load_slot(slot, seq.slot_kv, seq.pos, seq.first_token)
        seq.slot_kv = None

    def decode_step(self) -> list[Sequence]:
        eos_id = state["tokenizer"].eos_token_id
        live = [s for s in self.slots if s is not None]
        b_need = max(s.slot for s in live) + 1
        len_need = max(s.pos for s in live) + 1
        next_tokens = self.dec.decode(b_need, len_need)

        finished = []
        for seq in live:
            tok = next_tokens[seq.slot]
            seq.pos += 1
            seq.generated_ids.append(tok)
            if tok == eos_id or len(seq.generated_ids) >= seq.max_new_tokens:
                finished.append(seq)
                self.slots[seq.slot] = None
                self.dec.free_slot(seq.slot)
        return finished

    def fail_all(self, exc: BaseException) -> None:
        for i, seq in enumerate(self.slots):
            if seq is not None:
                _complete_err(seq, exc)
                self.dec.free_slot(i)
                self.slots[i] = None


def _gpu_worker(stop_event: threading.Event) -> None:
    incoming: queue.Queue = state["incoming"]
    batch = SlotBatch(state["decoder"])

    def admit(seq: Sequence) -> None:
        if seq.future.done():
            return
        try:
            _prefill(seq)
        except Exception as exc:
            _complete_err(seq, exc)
            return
        if seq.done:
            _complete_ok(seq)
            return
        batch.add(seq)

    while not stop_event.is_set():
        try:
            if batch.empty:
                try:
                    item = incoming.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    break
                admit(item)
                continue

            while len(batch) < MAX_BATCH_SIZE:
                try:
                    item = incoming.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    stop_event.set()
                    break
                admit(item)

            for _ in range(DECODE_BURST):
                if batch.empty:
                    break
                for seq in batch.decode_step():
                    _complete_ok(seq)
                if not incoming.empty() and len(batch) < MAX_BATCH_SIZE:
                    break
        except Exception as exc:
            print(f"[gpu_worker] error: {exc!r}")
            batch.fail_all(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    decoder = StaticDecoder(model, MAX_BATCH_SIZE, MAX_SEQ_LEN)
    if CUDA_GRAPHS:
        t0 = time.perf_counter()
        decoder.capture()
        print(
            f"Captured {len(decoder.graphs)} CUDA graphs in {time.perf_counter() - t0:.1f}s "
            f"(batch {decoder.batch_buckets} x len {decoder.len_buckets})"
        )

    state["tokenizer"] = tokenizer
    state["model"] = model
    state["decoder"] = decoder
    state["incoming"] = queue.Queue()
    stop_event = threading.Event()
    worker = threading.Thread(target=_gpu_worker, args=(stop_event,), name="gpu-worker", daemon=True)
    worker.start()

    print(f"Model loaded. cuda_graph | batch={MAX_BATCH_SIZE} graphs={CUDA_GRAPHS}")
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
    return GenerateResponse(
        text=result["text"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        latency_s=time.perf_counter() - start,
    )


@app.get("/health")
async def health():
    decoder = state.get("decoder")
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "max_batch_size": MAX_BATCH_SIZE,
        "scheduler": "static_kv_cuda_graph" if CUDA_GRAPHS else "static_kv_eager",
        "decode_burst": DECODE_BURST,
        "cuda_graphs": CUDA_GRAPHS,
        "graph_count": len(decoder.graphs) if decoder else 0,
    }
