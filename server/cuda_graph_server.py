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
  - Prefill also writes straight into the slots. Short prompts (<= PREFILL_CHUNK
    tokens) are batched into one right-padded forward, graphed per
    (rows bucket, prompt-length bucket), so admitting new requests costs a few ms
    instead of ~30 ms of Python/kernel-launch overhead
  - Longer prompts are prefilled PREFILL_CHUNK tokens at a time (each chunk
    attends to the slot's cached prefix), with a decode step between chunks, so
    running sequences keep generating while a long prompt is ingested

Supports Llama-style decoders (Qwen2, Llama, Mistral): embed -> N x
[norm, attn with RoPE, norm, MLP] -> norm -> lm_head.

Run:
    uvicorn server.cuda_graph_server:app --host 0.0.0.0 --port 8000

Knobs (env):
    MODEL_NAME       default Qwen/Qwen2.5-1.5B-Instruct
    MAX_BATCH_SIZE   default 32
    MAX_SEQ_LEN      default 2048
    DECODE_BURST     default 64
    CUDA_GRAPHS      default 1 (0 = run the same static decode/prefill eagerly)
    TORCH_COMPILE    default auto (torch.compile the decode layer to fuse its elementwise ops
                     when Triton and a C compiler are available: on in the Docker image, off on Windows)
    PREFILL_CHUNK    default 512 (longer prompts are chunked and interleaved with decode)
    PREFILL_TOKEN_BUDGET  default 4096 (max padded tokens per batched prefill forward)
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import queue
import shutil
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
def _can_compile() -> bool:
    return importlib.util.find_spec("triton") is not None and any(
        shutil.which(cc) for cc in ("cc", "gcc", "clang")
    )


_torch_compile = os.environ.get("TORCH_COMPILE", "auto").strip().lower()
TORCH_COMPILE = _can_compile() if _torch_compile == "auto" else _torch_compile in {"1", "true", "yes", "on"}
PREFILL_CHUNK = int(os.environ.get("PREFILL_CHUNK", "512"))
PREFILL_TOKEN_BUDGET = int(os.environ.get("PREFILL_TOKEN_BUDGET", "4096"))

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
    prefilled: int = 0  # prompt tokens already in the slot's KV cache
    decoding: bool = False
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

    def __init__(
        self,
        model,
        max_batch: int,
        max_len: int,
        prefill_chunk: int = PREFILL_CHUNK,
        prefill_budget: int = PREFILL_TOKEN_BUDGET,
    ):
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
        self.prefill_chunk = min(prefill_chunk, max_len)
        self.prefill_budget = max(prefill_budget, self.prefill_chunk)

        # One extra row past the real slots: padding rows of a graphed prefill write there.
        self.scratch = max_batch
        shape = (max_batch + 1, self.n_kv, max_len, self.head_dim)
        n_layers = len(self.inner.layers)
        self.k_cache = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(n_layers)]
        self.v_cache = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(n_layers)]

        # Graph inputs/state: must keep fixed addresses, so only ever written in place.
        self.tokens = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self.pos = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self.active = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self.arange_b = torch.arange(max_batch, device=self.device)
        self.arange_l = torch.arange(max_len, device=self.device)
        self.pf_ids = torch.zeros(max_batch, self.prefill_chunk, dtype=torch.long, device=self.device)
        self.pf_slots = torch.full((max_batch,), self.scratch, dtype=torch.long, device=self.device)
        self.pf_last = torch.zeros(max_batch, dtype=torch.long, device=self.device)

        self.batch_buckets = _buckets(max_batch, 1)
        self.len_buckets = _buckets(max_len, min(256, max_len))
        self.prefill_len_buckets = _buckets(self.prefill_chunk, min(16, self.prefill_chunk))
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.graph_out: dict[tuple[int, int], torch.Tensor] = {}
        self.prefill_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.prefill_out: dict[tuple[int, int], torch.Tensor] = {}


    def _decode_layer(self, layer, k_cache, v_cache, h, cos, sin, mask, rows, pos, length: int):
        """One decoder layer of the decode step. Kept separate so TORCH_COMPILE can compile it
        once per shape and reuse it for every layer (parameters are inputs, not constants)."""
        b = h.shape[0]
        q, k, v = self._attn_inputs(layer.self_attn, layer.input_layernorm(h), cos, sin)

        k_cache[rows, :, pos] = k[:, :, 0]
        v_cache[rows, :, pos] = v[:, :, 0]
        keys = k_cache[:b, :, :length]
        values = v_cache[:b, :, :length]

        # GQA without copying K/V: fold each KV head's query group into the query-length dim.
        # (SDPA's enable_gqa=True with a mask falls back to a ~6x slower kernel.)
        q = q.reshape(b, self.n_kv, self.group, self.head_dim)
        o = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        h = h + layer.self_attn.o_proj(o.reshape(b, 1, -1))
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def _attn_inputs(self, attn, x: torch.Tensor, cos, sin):
        """q [n, heads, L, d] and k, v [n, kv_heads, L, d] for x [n, L, hidden], RoPE applied."""
        n, length = x.shape[:2]
        q = attn.q_proj(x).view(n, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(n, length, self.n_kv, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(n, length, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def prefill_fits(self, n: int, length: int) -> bool:
        """Whether n short prompts of at most `length` tokens fit one batched prefill."""
        if CUDA_GRAPHS:
            n, length = _pick(self.batch_buckets, n), _pick(self.prefill_len_buckets, length)
        return n * length <= self.prefill_budget

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
            h = self._decode_layer(layer, self.k_cache[i], self.v_cache[i], h, cos, sin, mask, rows, pos, length)

        logits = self.model.lm_head(self.inner.norm(h))
        next_tokens = logits[:, -1, :].argmax(dim=-1)

        # Feed back for the next step without leaving the GPU; idle slots stay at pos 0.
        self.tokens[:b].copy_(next_tokens)
        self.pos[:b].add_(self.active[:b])
        return next_tokens

    def _prefill_fresh(self, n: int, length: int) -> torch.Tensor:
        """Prefill pf_ids[:n, :length] from position 0 into slots pf_slots[:n]; next token per row.

        Rows are right-padded: causal attention keeps real tokens from seeing the pads after them,
        and the pads' K/V (past each prompt's end) is overwritten by decode before it is ever read.
        """
        ids = self.pf_ids[:n, :length]
        slots = self.pf_slots[:n].unsqueeze(1)
        positions = self.arange_l[:length]

        h = self.inner.embed_tokens(ids)  # [n, length, hidden]
        cos, sin = self.inner.rotary_emb(h, positions.unsqueeze(0))
        for i, layer in enumerate(self.inner.layers):
            q, k, v = self._attn_inputs(layer.self_attn, layer.input_layernorm(h), cos, sin)

            self.k_cache[i][slots, :, positions] = k.transpose(1, 2)
            self.v_cache[i][slots, :, positions] = v.transpose(1, 2)

            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            h = h + layer.self_attn.o_proj(o.transpose(1, 2).reshape(n, length, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        last = h[self.arange_b[:n], self.pf_last[:n]]
        return self.model.lm_head(self.inner.norm(last)).argmax(dim=-1)

    def _prefill_at(self, slot: int, ids: torch.Tensor, start: int) -> torch.Tensor:
        """Prefill one chunk of a long prompt at positions [start, start + len(ids)) of `slot`."""
        length = ids.shape[0]
        end = start + length
        positions = self.arange_l[start:end]

        h = self.inner.embed_tokens(ids).unsqueeze(0)  # [1, length, hidden]
        cos, sin = self.inner.rotary_emb(h, positions.unsqueeze(0))
        # Causal over the cached prefix plus this chunk, repeated per query head in a KV group
        # to match the folded GQA layout below.
        mask = (self.arange_l[:end].unsqueeze(0) <= positions.unsqueeze(1)).repeat(self.group, 1)[None, None]
        for i, layer in enumerate(self.inner.layers):
            q, k, v = self._attn_inputs(layer.self_attn, layer.input_layernorm(h), cos, sin)

            self.k_cache[i][slot, :, start:end] = k[0]
            self.v_cache[i][slot, :, start:end] = v[0]
            keys = self.k_cache[i][slot : slot + 1, :, :end]
            values = self.v_cache[i][slot : slot + 1, :, :end]

            q = q.reshape(1, self.n_kv, self.group * length, self.head_dim)
            o = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
            o = o.reshape(1, self.n_heads, length, self.head_dim).transpose(1, 2).reshape(1, length, -1)
            h = h + layer.self_attn.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        return self.model.lm_head(self.inner.norm(h[:, -1])).argmax(dim=-1)

    def compile(self) -> None:
        # ~1-4 s per decode shape (one compile serves all layers); capture() triggers them.
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, 2 * len(self.batch_buckets) * len(self.len_buckets)
        )
        self._decode_layer = torch.compile(self._decode_layer, dynamic=False, fullgraph=True)

    @staticmethod
    def _capture_one(fn, pool):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            out = fn()
        return graph, out

    @torch.no_grad()
    def capture(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        # Largest first so smaller graphs can reuse its memory from the shared pool.
        for n in reversed(self.batch_buckets):
            for length in reversed(self.prefill_len_buckets):
                if n * length > self.prefill_budget:
                    continue
                graph, out = self._capture_one(lambda: self._prefill_fresh(n, length), pool)
                self.prefill_graphs[(n, length)] = graph
                self.prefill_out[(n, length)] = out
        for b in reversed(self.batch_buckets):
            for length in reversed(self.len_buckets):
                graph, out = self._capture_one(lambda: self._step(b, length), pool)
                self.graphs[(b, length)] = graph
                self.graph_out[(b, length)] = out
        self.tokens.zero_()
        self.pos.zero_()
        torch.cuda.synchronize()

    @torch.no_grad()
    def prefill(self, slots: list[int], prompts: list[list[int]]) -> list[int]:
        """Batched prefill of whole short prompts into `slots`; returns each one's first new token."""
        n, longest = len(slots), max(len(p) for p in prompts)
        if CUDA_GRAPHS:
            rows, length = _pick(self.batch_buckets, n), _pick(self.prefill_len_buckets, longest)
        else:
            rows, length = n, longest
        ids = torch.zeros(rows, length, dtype=torch.long)
        for r, p in enumerate(prompts):
            ids[r, : len(p)] = torch.tensor(p)
        pad = rows - n
        self.pf_ids[:rows, :length].copy_(ids)
        self.pf_slots[:rows].copy_(torch.tensor(slots + [self.scratch] * pad))
        self.pf_last[:rows].copy_(torch.tensor([len(p) - 1 for p in prompts] + [0] * pad))
        if CUDA_GRAPHS:
            self.prefill_graphs[(rows, length)].replay()
            out = self.prefill_out[(rows, length)]
        else:
            out = self._prefill_fresh(rows, length)
        return out[:n].tolist()

    @torch.no_grad()
    def prefill_long(self, slot: int, ids: list[int], start: int) -> int:
        """One chunk of a prompt longer than prefill_chunk; the token is only meaningful on the last chunk."""
        return self._prefill_at(slot, torch.tensor(ids, device=self.device), start).item()

    def reserve_slot(self, slot: int) -> None:
        # Decode steps still run over this row while it is prefilling; park its writes at the
        # last position, which no prompt reaches and decode overwrites before reading.
        self.active[slot] = 0
        self.pos[slot] = self.max_len - 1

    def activate(self, slots: list[int], first_tokens: list[int], lens: list[int]) -> None:
        idx = torch.tensor(slots, device=self.device)
        self.tokens[idx] = torch.tensor(first_tokens, device=self.device)
        self.pos[idx] = torch.tensor(lens, device=self.device)
        self.active[idx] = 1

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


class SlotBatch:
    def __init__(self, decoder: StaticDecoder):
        self.dec = decoder
        self.slots: list[Sequence | None] = [None] * decoder.max_batch
        self.prefilling: list[Sequence] = []  # slotted, prompt not fully in the cache yet (FIFO)

    def __len__(self) -> int:
        return sum(s is not None for s in self.slots)

    @property
    def empty(self) -> bool:
        return all(s is None for s in self.slots)

    @property
    def free(self) -> int:
        return self.slots.count(None)

    def admit(self, seqs: list[Sequence]) -> list[tuple[Sequence, Exception]]:
        """Tokenize seqs and reserve a slot for each; prefill_step() then fills them in.

        Returns the rejected ones with their error. Caller must not pass more than self.free.
        """
        tokenizer = state["tokenizer"]
        failed = []
        for seq in seqs:
            ids = tokenizer(seq.prompt)["input_ids"]
            if len(ids) + seq.max_new_tokens > MAX_SEQ_LEN:
                failed.append((seq, ValueError(
                    f"prompt ({len(ids)}) + max_new_tokens ({seq.max_new_tokens}) "
                    f"exceeds MAX_SEQ_LEN ({MAX_SEQ_LEN})"
                )))
                continue
            seq.prompt_token_ids = ids
            seq.slot = self.slots.index(None)
            self.slots[seq.slot] = seq
            self.dec.reserve_slot(seq.slot)
            self.prefilling.append(seq)
        return failed

    def prefill_step(self) -> list[Sequence]:
        """Run one prefill forward: either a batch of whole short prompts or one chunk of a long one.

        Returns sequences that finished on their first token (EOS or max_new_tokens <= 1).
        """
        if not self.prefilling:
            return []
        chunk = self.dec.prefill_chunk
        head = self.prefilling[0]
        if len(head.prompt_token_ids) > chunk:
            start = head.prefilled
            ids = head.prompt_token_ids[start : start + chunk]
            first = self.dec.prefill_long(head.slot, ids, start)
            head.prefilled += len(ids)
            if head.prefilled < len(head.prompt_token_ids):
                return []
            self.prefilling.pop(0)
            return self._start_decoding([head], [first])

        group, longest = [], 0
        for seq in self.prefilling:
            n = len(seq.prompt_token_ids)
            if n > chunk or not self.dec.prefill_fits(len(group) + 1, max(longest, n)):
                break
            group.append(seq)
            longest = max(longest, n)
        del self.prefilling[: len(group)]
        first = self.dec.prefill([s.slot for s in group], [s.prompt_token_ids for s in group])
        return self._start_decoding(group, first)

    def _start_decoding(self, seqs: list[Sequence], first: list[int]) -> list[Sequence]:
        eos_id = state["tokenizer"].eos_token_id
        finished, go = [], []
        for seq, tok in zip(seqs, first):
            seq.prefilled = len(seq.prompt_token_ids)
            seq.generated_ids = [tok]
            seq.pos = seq.prefilled
            if tok == eos_id or seq.max_new_tokens <= 1:
                seq.done = True
                finished.append(seq)
                self.slots[seq.slot] = None
                self.dec.free_slot(seq.slot)
            else:
                seq.decoding = True
                go.append(seq)
        if go:
            self.dec.activate([s.slot for s in go], [s.generated_ids[0] for s in go], [s.pos for s in go])
        return finished

    def decode_step(self) -> list[Sequence]:
        live = [s for s in self.slots if s is not None and s.decoding]
        if not live:
            return []
        eos_id = state["tokenizer"].eos_token_id
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
        self.prefilling.clear()
        for i, seq in enumerate(self.slots):
            if seq is not None:
                _complete_err(seq, exc)
                self.dec.free_slot(i)
                self.slots[i] = None


def _gpu_worker(stop_event: threading.Event) -> None:
    incoming: queue.Queue = state["incoming"]
    batch = SlotBatch(state["decoder"])

    def admit_waiting(first: Sequence | None = None) -> bool:
        """Drain queued requests into free slots (prefill happens in the loop). False means shutdown."""
        items = [first] if first is not None else []
        alive = True
        while len(items) < batch.free:
            try:
                item = incoming.get_nowait()
            except queue.Empty:
                break
            if item is None:
                alive = False
                break
            items.append(item)
        items = [s for s in items if not s.future.done()]
        if not items:
            return alive
        try:
            failed = batch.admit(items)
        except Exception as exc:
            for seq in items:
                if seq.slot < 0:
                    _complete_err(seq, exc)
            raise
        for seq, exc in failed:
            _complete_err(seq, exc)
        return alive

    while not stop_event.is_set():
        try:
            if batch.empty:
                try:
                    item = incoming.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    break
                if not admit_waiting(item):
                    stop_event.set()
                continue

            if not admit_waiting():
                stop_event.set()

            for seq in batch.prefill_step():
                _complete_ok(seq)
            if batch.prefilling:
                # More prefill queued (a long prompt's next chunk, or over budget): give running
                # sequences one token, then come back, instead of stalling them for all of it.
                for seq in batch.decode_step():
                    _complete_ok(seq)
                continue

            for _ in range(DECODE_BURST):
                if batch.empty:
                    break
                for seq in batch.decode_step():
                    _complete_ok(seq)
                if not incoming.empty() and batch.free:
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
    if TORCH_COMPILE:
        decoder.compile()
    if CUDA_GRAPHS:
        t0 = time.perf_counter()
        decoder.capture()
        print(
            f"Captured {len(decoder.graphs)} decode + {len(decoder.prefill_graphs)} prefill CUDA graphs "
            f"in {time.perf_counter() - t0:.1f}s (batch {decoder.batch_buckets} x len {decoder.len_buckets}; "
            f"prefill len {decoder.prefill_len_buckets}, <= {decoder.prefill_budget} tokens)"
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
        "torch_compile": TORCH_COMPILE,
        "graph_count": len(decoder.graphs) if decoder else 0,
        "prefill_graph_count": len(decoder.prefill_graphs) if decoder else 0,
        "prefill_chunk": decoder.prefill_chunk if decoder else PREFILL_CHUNK,
    }
