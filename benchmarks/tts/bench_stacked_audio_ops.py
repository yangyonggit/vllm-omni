#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark: MOSS-TTS talker stacked audio ops vs per-head loop.

Measures per-step latency of the two hot paths replaced by PR #4230:

  1. audio_head  — n_vq serial nn.Linear calls  vs  stacked_w @ h (batched matmul)
  2. audio_embed — n_vq serial Embedding lookups vs  stacked_emb[arange, codes].sum(0)

No model checkpoint is required; weights are randomly initialised.

Usage::

    python benchmarks/tts/bench_stacked_audio_ops.py \\
        [--n-vq 16 32] \\
        [--hidden 2048] \\
        [--vocab 4096] \\
        [--num-runs 500]

Output is printed as a Markdown table suitable for pasting into a PR description.
"""

from __future__ import annotations

import argparse
import time

import torch


def _sync(device: torch.device) -> None:
    torch.accelerator.synchronize(device)


def _bench(fn, n: int, device: torch.device) -> float:
    """Return mean latency in ms over n runs (excludes first warm-up call)."""
    fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Bench MOSS-TTS stacked audio ops vs loop")
    parser.add_argument("--n-vq", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--num-runs", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda")
    N: int = args.num_runs
    H: int = args.hidden
    V: int = args.vocab + 1  # +1 for pad sentinel

    print(f"Device: {torch.cuda.get_device_name(device)}  hidden={H}  vocab={V - 1}  n_runs={N}\n")

    head_rows: list[tuple] = []
    emb_rows: list[tuple] = []

    for n_vq in args.n_vq:
        h = torch.randn(H, device=device)
        codes = torch.randint(0, V - 1, (n_vq,), device=device)
        idx = torch.arange(n_vq, device=device)

        # --- audio head ---
        heads = [torch.nn.Linear(H, V, bias=False).to(device) for _ in range(n_vq)]
        stacked_head = torch.stack([lin.weight.detach() for lin in heads])

        head_loop_ms = _bench(lambda: [heads[i](h) for i in range(n_vq)], N, device)
        head_stack_ms = _bench(lambda: stacked_head @ h, N, device)
        head_rows.append((n_vq, head_loop_ms, head_stack_ms))

        # --- audio embed ---
        embs = [torch.nn.Embedding(V, H).to(device) for _ in range(n_vq)]
        stacked_emb = torch.stack([e.weight.detach() for e in embs])

        emb_loop_ms = _bench(lambda: sum(embs[i](codes[i : i + 1]) for i in range(n_vq)), N, device)
        emb_stack_ms = _bench(lambda: stacked_emb[idx, codes].sum(0), N, device)
        emb_rows.append((n_vq, emb_loop_ms, emb_stack_ms))

    print("**audio_head** (batched matmul vs per-head nn.Linear loop)")
    print("| n_vq | loop (ms) | stacked (ms) | speedup |")
    print("|------|----------|-------------|---------|")
    for n_vq, loop_ms, stack_ms in head_rows:
        print(f"| {n_vq} | {loop_ms:.3f} | {stack_ms:.3f} | {loop_ms / stack_ms:.2f}x |")

    print()
    print("**audio_embed** (batched gather vs per-head Embedding loop)")
    print("| n_vq | loop (ms) | stacked (ms) | speedup |")
    print("|------|----------|-------------|---------|")
    for n_vq, loop_ms, stack_ms in emb_rows:
        print(f"| {n_vq} | {loop_ms:.3f} | {stack_ms:.3f} | {loop_ms / stack_ms:.2f}x |")


if __name__ == "__main__":
    main()
