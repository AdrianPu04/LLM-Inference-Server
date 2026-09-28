"""
Render the charts used in WRITEUP.md / README.md from the JSON files in results/.

    pip install matplotlib
    python scripts/plot_results.py            # writes docs/img/*.png
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "img"

OURS = "#2563eb"
VLLM = "#f97316"
GREY = "#9ca3af"


def load(name: str) -> dict[int, dict]:
    """Rows keyed by concurrency; repeated levels in one file are averaged."""
    rows = defaultdict(list)
    for r in json.loads((RESULTS / f"{name}.json").read_text()):
        rows[r["concurrency"]].append(r)
    out = {}
    for c, rs in rows.items():
        keys = [k for k, v in rs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        out[c] = {k: sum(r[k] or 0 for r in rs) / len(rs) for k in keys}
    return dict(sorted(out.items()))


def series(name: str, key: str):
    d = load(name)
    return list(d), [row[key] for row in d.values()]


def style(ax, title, xlabel, ylabel, log_x=True):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"wrote docs/img/{name}")


def phases_1_3():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for name, label, color in [
        ("naive_baseline", "Naive", GREY),
        ("static_batch", "Static batching", "#10b981"),
        ("continuous_batch", "Continuous batching", OURS),
    ]:
        a.plot(*series(name, "throughput_req_per_s"), "o-", label=label, color=color)
        b.plot(*series(name, "p99_latency_s"), "o-", label=label, color=color)
    style(a, "Throughput (Windows, 4070, batch 8)", "concurrency", "req/s")
    style(b, "p99 latency", "concurrency", "seconds")
    a.legend()
    save(fig, "phases_1_3.png")


def progression():
    steps = [
        ("sweep_bs8", "Continuous, batch 8 (original)"),
        ("sweep_bs32", "Continuous, batch 32"),
        ("static_eager_bs32", "Static KV cache (eager)"),
        ("cuda_graph_bs32", "+ CUDA-graph decode"),
        ("batched_prefill_bs32", "+ batched prefill"),
        ("chunked_prefill_bs32", "+ graphed / chunked prefill"),
        ("compiled_bs32", "+ torch.compile'd layer"),
    ]
    labels = [label for _, label in steps]
    values = [load(name)[32]["throughput_tokens_per_s"] for name, _ in steps]
    vllm = load("vllm_docker_64")[32]["throughput_tokens_per_s"]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(labels, values, color=[GREY] * 2 + [OURS] * 5)
    ax.axvline(vllm, color=VLLM, lw=2, ls="--", label=f"vLLM ({vllm:.0f} tok/s)")
    for bar, v in zip(bars, values):
        ax.text(v + 30, bar.get_y() + bar.get_height() / 2, f"{v:.0f}", va="center", fontsize=9)
    ax.invert_yaxis()
    style(ax, "Throughput at 32 concurrent requests (4070, Docker, 128 tokens)", "tokens/s", "", log_x=False)
    ax.grid(True, axis="x", alpha=0.3)
    ax.grid(False, axis="y")
    ax.set_xlim(0, max(values) * 1.1)
    ax.legend(loc="upper right")
    save(fig, "progression_c32.png")


def vs_vllm():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    lines = [
        ("continuous_docker", "Original continuous (batch 8)", GREY, "o--"),
        ("compiled_bs32", "Ours, final (batch 32)", OURS, "o-"),
        ("compiled_bs64", "Ours, final (batch 64)", OURS, "s:"),
        ("vllm_docker_64", "vLLM", VLLM, "o-"),
        ("vllm_docker_128", "vLLM (128 req/level)", VLLM, "s:"),
    ]
    for name, label, color, fmt in lines:
        a.plot(*series(name, "throughput_tokens_per_s"), fmt, label=label, color=color)
        b.plot(*series(name, "p50_latency_s"), fmt, label=label, color=color)
    style(a, "Throughput vs concurrency (4070, Docker)", "concurrency", "tokens/s")
    style(b, "Median latency (128-token completions)", "concurrency", "seconds")
    b.set_ylim(bottom=0)
    a.legend(fontsize=8)
    save(fig, "vs_vllm.png")


def long_prompts():
    runs = [
        ("long_prompts_chunk2048", "Ours, whole-prompt\nprefill", GREY),
        ("long_prompts_chunk512", "Ours, chunked\nprefill (512)", OURS),
        ("vllm_docker_long_prompts", "vLLM", VLLM),
    ]
    rows = [load(name)[32] for name, _, _ in runs]
    labels = [label for _, label, _ in runs]
    colors = [color for _, _, color in runs]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title, unit in [
        (a, "throughput_tokens_per_s", "Throughput", "tokens/s"),
        (b, "short_p50_latency_s", "Short-request median latency", "seconds"),
    ]:
        vals = [r[key] for r in rows]
        bars = ax.bar(labels, vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}" if v > 50 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        style(ax, title, "", unit, log_x=False)
        ax.grid(False, axis="x")
    fig.suptitle("Every 8th prompt ~1,500 tokens, 32 concurrent requests", fontsize=11)
    save(fig, "long_prompts.png")


def gke():
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, label, color, fmt in [
        ("gke_continuous_batch", "Original continuous (batch 8)", GREY, "o--"),
        ("gke_vllm_seqs8", "vLLM, max 8 seqs", VLLM, "o:"),
        ("gke_vllm", "vLLM, default", VLLM, "o-"),
    ]:
        ax.plot(*series(name, "throughput_req_per_s"), fmt, label=label, color=color)
    style(ax, "GKE, one NVIDIA L4 (pre-Phase 6 server)", "concurrency", "req/s")
    ax.legend(fontsize=8)
    save(fig, "gke_l4.png")


if __name__ == "__main__":
    phases_1_3()
    progression()
    vs_vllm()
    long_prompts()
    gke()
