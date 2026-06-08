#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark: MOSS-TTS codec decoder eager vs CUDA Graph.

Measures per-call latency of MossAudioTokenizerModel.batch_decode (eager)
against MossTTSCUDAGraphCodecWrapper.decode (CUDA Graph) across a range of
code-frame counts (T) that cover typical streaming chunk sizes.

Default chunk sizes match the capture list in modeling_moss_tts_codec.py.

Usage::

    # both paths (default)
    python benchmarks/tts/bench_codec_cudagraph.py

    # eager only (no CUDA Graph warmup)
    python benchmarks/tts/bench_codec_cudagraph.py --mode eager

    # CUDA Graph only
    python benchmarks/tts/bench_codec_cudagraph.py --mode graph

    # custom sizes
    python benchmarks/tts/bench_codec_cudagraph.py --chunk-sizes 25 50 100

Output is printed as a Markdown table suitable for pasting into a PR description.
"""

from __future__ import annotations

import argparse
import time

import torch

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
)
from vllm_omni.model_executor.models.moss_tts.moss_codec_cudagraph import (
    MossTTSCUDAGraphCodecWrapper,
)


def _time_fn(fn, n: int, device: torch.device) -> float:
    """Return mean latency in ms over n runs (excludes first warm-up call)."""
    fn()  # warm-up
    torch.accelerator.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.accelerator.synchronize(device)
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> None:
    # Default sizes match the capture list in modeling_moss_tts_codec.py
    _DEFAULT_SIZES = [4, 8, 16, 25, 32, 50, 64, 100, 128, 200, 256]

    parser = argparse.ArgumentParser(description="Bench MOSS-TTS codec CUDA Graph vs eager")
    parser.add_argument("--codec-path", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--n-vq", type=int, default=16)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=_DEFAULT_SIZES)
    parser.add_argument("--num-runs", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=["both", "eager", "graph"],
        default="both",
        help="'both' compares eager vs CUDA Graph; 'eager'/'graph' times one path only",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    n_vq: int = args.n_vq
    sizes: list[int] = sorted(args.chunk_sizes)
    n_runs: int = args.num_runs
    mode: str = args.mode

    print(f"Loading codec from {args.codec_path} …")
    cfg = MossAudioTokenizerConfig.from_pretrained(args.codec_path)
    model = MossAudioTokenizerModel(cfg).to(device=device, dtype=torch.float32).eval()

    wrapper = None
    if mode in ("both", "graph"):
        print(f"Warming up CUDA Graph (n_vq={n_vq}, sizes={sizes}) …")
        wrapper = MossTTSCUDAGraphCodecWrapper(model, capture_sizes=sizes, num_quantizers=n_vq)
        wrapper.warmup(device)

    print(f"\nBenchmark: mode={mode} n_vq={n_vq} n_runs={n_runs} device={torch.cuda.get_device_name(device)}\n")

    if mode == "both":
        print(f"| {'T':>6} | {'Eager (ms)':>12} | {'CUDA Graph (ms)':>16} | {'Speedup':>8} |")
        print(f"|{'-' * 8}|{'-' * 14}|{'-' * 18}|{'-' * 10}|")
    elif mode == "eager":
        print(f"| {'T':>6} | {'Eager (ms)':>12} |")
        print(f"|{'-' * 8}|{'-' * 14}|")
    else:
        print(f"| {'T':>6} | {'CUDA Graph (ms)':>16} |")
        print(f"|{'-' * 8}|{'-' * 18}|")

    for T in sizes:
        codes = torch.zeros(n_vq, T, dtype=torch.long, device=device)

        if mode in ("both", "eager"):
            eager_ms = _time_fn(
                lambda: model.batch_decode(codes_list=[codes], num_quantizers=n_vq),
                n_runs,
                device,
            )
        if mode in ("both", "graph") and wrapper is not None:
            graph_ms = _time_fn(lambda: wrapper.decode(codes), n_runs, device)

        if mode == "both":
            print(f"| {T:>6} | {eager_ms:>12.2f} | {graph_ms:>16.2f} | {eager_ms / graph_ms:>7.2f}x |")
        elif mode == "eager":
            print(f"| {T:>6} | {eager_ms:>12.2f} |")
        else:
            print(f"| {T:>6} | {graph_ms:>16.2f} |")


if __name__ == "__main__":
    main()
