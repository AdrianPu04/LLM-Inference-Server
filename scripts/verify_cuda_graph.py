"""
Check that server/cuda_graph_server.py's static-KV decode matches HF generate().

Runs greedy decoding for several prompts through HF (one at a time, no padding)
and through StaticDecoder/SlotBatch with staggered admission, mixed lengths and
slot reuse, then compares token IDs. Tests both eager and CUDA-graph modes.

    python scripts/verify_cuda_graph.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import server.cuda_graph_server as cgs  # noqa: E402

PROMPTS = [
    ("Explain the difference between a stack and a queue.", 48),
    ("Write a short story about a robot learning to paint.", 96),
    ("Summarize the plot of Romeo and Juliet in three sentences.", 24),
    ("What are the main causes of the French Revolution?", 64),
    ("Describe how photosynthesis works at a high level.", 80),
    ("List three uses of a hash map.", 32),
]


def reference(model, tokenizer, prompt, n):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        # Override Qwen's generation_config (repetition_penalty etc.) to get plain argmax, like the servers.
        out = model.generate(
            ids,
            max_new_tokens=n,
            do_sample=False,
            repetition_penalty=1.0,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    toks = out[0, ids.shape[1]:].tolist()
    if tokenizer.eos_token_id in toks:
        toks = toks[: toks.index(tokenizer.eos_token_id) + 1]
    return toks


def run_static(decoder):
    batch = cgs.SlotBatch(decoder)
    seqs = [
        cgs.Sequence(prompt=p, max_new_tokens=n, future=None, loop=None, enqueued_at=0.0)
        for p, n in PROMPTS
    ]
    pending = list(seqs)

    def admit(k):
        for _ in range(k):
            if pending and len(batch) < decoder.max_batch:
                s = pending.pop(0)
                cgs._prefill(s)
                if not s.done:
                    batch.add(s)

    admit(2)
    step = 0
    while not batch.empty or pending:
        if step in (5, 20, 40):  # staggered joins; later ones land in freed slots
            admit(2)
        if batch.empty:
            admit(1)
            continue
        batch.decode_step()
        step += 1
    return [s.generated_ids for s in seqs]


def main():
    name = cgs.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    cgs.state["tokenizer"] = tokenizer
    cgs.state["model"] = model

    refs = [reference(model, tokenizer, p, n) for p, n in PROMPTS]

    decoder = cgs.StaticDecoder(model, max_batch=4, max_len=512)
    ok = True
    for graphs in (False, True):
        cgs.CUDA_GRAPHS = graphs
        if graphs:
            decoder.capture()
        outs = run_static(decoder)
        label = "cuda-graph" if graphs else "eager"
        for (prompt, _), ref, got in zip(PROMPTS, refs, outs):
            match = 0
            for a, b in zip(ref, got):
                if a != b:
                    break
                match += 1
            status = "OK " if ref == got else "DIFF"
            ok &= ref == got
            print(f"[{label:10}] {status} {match:3}/{len(ref):3} tokens match  {prompt[:40]!r}")
    print("\nALL MATCH" if ok else "\nMISMATCHES (fp16 kernel differences can flip a near-tie argmax)")


if __name__ == "__main__":
    main()
