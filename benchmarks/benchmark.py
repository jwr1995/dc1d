"""
Benchmarks for dc1d.

Uses ``torch.utils.benchmark.Timer``, which handles warmup, adaptive repeat
counts and -- critically on CUDA -- ``torch.cuda.synchronize()`` around the timed
region. The benchmarks that used to live at the bottom of ``dc1d/nn.py`` and in
``playground/param_example.py`` wrapped ``time.time()`` around an asynchronous
CUDA launch, so on GPU they measured Python dispatch overhead rather than
execution. They also gave the deformable path three warmup iterations while the
``nn.Conv1d`` comparison got a single cold call including algorithm selection,
which biased the comparison in dc1d's favour.

Run with:
    uv run python benchmarks/benchmark.py
    uv run python benchmarks/benchmark.py --device cuda
"""

from __future__ import annotations

import argparse

import torch
import torch.utils.benchmark as benchmark
from torch import nn

from dc1d.nn import DeformConv1d
from dc1d.ops import efficient_linterpolate, output_length


def fmt(measurement: benchmark.Measurement) -> str:
    return f"{measurement.median * 1e3:9.3f} ms"


def bench(stmt: str, globals_: dict, label: str, sub_label: str, description: str):
    timer = benchmark.Timer(
        stmt=stmt,
        globals=globals_,
        label=label,
        sub_label=sub_label,
        description=description,
    )
    return timer.blocked_autorange(min_run_time=1.0)


def run(device: str, batch: int, channels: int, length: int, kernel_size: int, dilation: int):
    results = []

    x = torch.randn(batch, channels, length, device=device, requires_grad=True)
    n_offsets = output_length(length, kernel_size, dilation, 1)
    offsets = torch.zeros(batch, 1, n_offsets, kernel_size, device=device, requires_grad=True)

    deform = DeformConv1d(
        channels, channels, kernel_size, dilation=dilation, groups=channels, padding="valid"
    ).to(device)
    vanilla = nn.Conv1d(
        channels, channels, kernel_size, dilation=dilation, groups=channels, padding="valid"
    ).to(device)

    sub = f"B={batch} C={channels} L={length} K={kernel_size} d={dilation}"
    g = {
        "x": x,
        "offsets": offsets,
        "deform": deform,
        "vanilla": vanilla,
        "efficient_linterpolate": efficient_linterpolate,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "torch": torch,
    }

    results.append(
        bench(
            "efficient_linterpolate(x, offsets, kernel_size, dilation, 1)",
            g,
            "forward",
            sub,
            "interpolation only",
        )
    )
    results.append(bench("deform(x, offsets)", g, "forward", sub, "DeformConv1d"))
    results.append(bench("vanilla(x)", g, "forward", sub, "nn.Conv1d"))

    # Backward, timed separately. The graph is rebuilt inside the timed region,
    # so these numbers are forward+backward; the forward table above is what to
    # subtract if you want backward alone.
    results.append(
        bench(
            "y = deform(x, offsets); y.sum().backward()",
            g,
            "forward+backward",
            sub,
            "DeformConv1d",
        )
    )
    results.append(
        bench("y = vanilla(x); y.sum().backward()", g, "forward+backward", sub, "nn.Conv1d")
    )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--kernel-size", type=int, default=3)
    # NOTE: the original benchmark wrote `2^7` for the "large dilation" case,
    # which is XOR in Python and evaluates to 5.
    parser.add_argument("--dilation", type=int, default=2**3)
    args = parser.parse_args()

    print(f"device: {args.device}  torch: {torch.__version__}")
    if args.device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    results = run(
        args.device, args.batch, args.channels, args.length, args.kernel_size, args.dilation
    )
    benchmark.Compare(results).print()

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(args.batch, args.channels, args.length, device="cuda")
        n = output_length(args.length, args.kernel_size, args.dilation, 1)
        offsets = torch.zeros(args.batch, 1, n, args.kernel_size, device="cuda")
        efficient_linterpolate(x, offsets, args.kernel_size, args.dilation, 1)
        torch.cuda.synchronize()
        print(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
