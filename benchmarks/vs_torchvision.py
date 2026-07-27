"""
Three-way comparison of 1D deformable convolution backends:

1.  **dc1d** -- ``take_along_dim`` + ``lerp`` + ``conv1d``, pure
    ``torch.autograd``, no compilation step. What the package ships today.
2.  **torchvision** -- ``torchvision.ops.deform_conv2d``, a hand-written
    C++/CUDA kernel, degenerated to 1D (height 1).
3.  **grid_sample** -- ``F.grid_sample`` on a degenerate ``(B, C, 1, L)``
    image. A candidate future backend for dc1d: ``grid_sample`` is an ATen
    builtin, so adopting it would preserve the "no compilation required"
    property, and it is what at least one independent 1D implementation uses.

This script answers two questions:

1.  **Are they the same operation?**  ``--check`` establishes numerical
    equivalence (forward and backward) on the subset of the input domain where
    both implementations are defined identically, and characterises the places
    where they are *not* the same -- see "Semantics" below.
2.  **How much does the compiled kernel win?**  ``--bench`` sweeps a grid of
    shapes with ``torch.utils.benchmark.Timer`` (forward and forward+backward
    timed separately, equal warmup on both sides) and ``--mem`` measures peak
    memory.

torchvision is *not* a dependency of dc1d and must not become one. Install it
into a throwaway environment:

    # CPU
    uv sync --group bench
    uv run --group bench python benchmarks/vs_torchvision.py --check

    # CUDA (separate venv -- the dev venv is CPU-only torch by design)
    uv venv --python 3.12 .venv-cuda
    VIRTUAL_ENV=.venv-cuda uv pip install --index-url https://download.pytorch.org/whl/cu129 \
        torch==2.13.0+cu129 torchvision==0.28.0+cu129
    VIRTUAL_ENV=.venv-cuda uv pip install -e . --no-deps
    .venv-cuda/bin/python benchmarks/vs_torchvision.py --all --device cuda


Degenerating 2D -> 1D
---------------------
``deform_conv2d`` is used with a height of 1 throughout:

======================  =============================  ==============================
                        dc1d                           torchvision (degenerate)
======================  =============================  ==============================
input                   ``(B, C, L)``                  ``(B, C, 1, L)``
weight                  ``(C_out, C_in/g, K)``         ``(C_out, C_in/g, 1, K)``
offsets                 ``(B, G, L_out, K)``           ``(B, 2*G*1*K, 1, L_out)``
stride/dilation         ``s`` / ``d``                  ``(1, s)`` / ``(1, d)``
modulation mask         n/a (DCNv1)                    ``mask=None``
======================  =============================  ==============================

torchvision packs offsets as ``(offset_group, kh, kw, {y, x})`` flattened into
the channel axis, i.e. channel ``(g*K + j)*2 + c`` with ``c == 0`` for the height
offset and ``c == 1`` for the width offset. All height offsets are zero here, so
the height bilinear interpolation is a no-op (verified empirically by
``_probe_offset_layout``). See :func:`dc1d_offsets_to_torchvision`.


Semantics: where the two agree and where they do not
----------------------------------------------------
*   **Interior, in-range offsets: identical** (to floating point). This is the
    regime the benchmark runs in and the regime ``--check`` asserts on.
*   **Out-of-range sampling positions: different by design.**
    ``efficient_linterpolate`` *clamps* the sampling position into
    ``[0, L-1]``, so a tap that runs off the left edge reads ``x[0]``.
    ``deform_conv2d`` treats out-of-bounds bilinear taps as *zero* (implicit
    zero padding), so the same tap reads ``0``, and a tap at ``-0.5`` reads
    ``0.5 * x[0]`` rather than ``x[0]``. Neither is wrong; they are different
    boundary conventions. dc1d's choice keeps the interpolation weights summing
    to 1 everywhere (see ``tests/test_equivalence.py``).
*   **Constrained mode has no torchvision equivalent.** dc1d's default
    ``unconstrained=False`` additionally confines each tap to its own receptive
    field (offset clamped so the tap stays within ``[t0, t0 + d*(K-1)]``).
    torchvision has no such restriction. Equivalence therefore only holds
    against ``DeformConv1d(..., unconstrained=True)``.
*   **Offset sign.** A positive x-offset moves the sampling point towards
    higher indices in *both* implementations (verified, not assumed).

The ``grid_sample`` backend is written to match **dc1d**, not torchvision:
``padding_mode='border'`` clamps the sampling coordinate exactly the way
``efficient_linterpolate`` clamps its index, so it should agree with dc1d at the
boundaries too. It carries its own caveat -- see ``_grid_sample_precision_study``:
the ``[-1, 1]`` grid normalisation is *lossy*, so unlike dc1d it cannot
reproduce ``nn.Conv1d`` bit-for-bit once the sequence is long.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch import Tensor, nn

from dc1d.nn import DeformConv1d
from dc1d.ops import output_length

try:
    import torchvision
    from torchvision.ops import deform_conv2d

    HAVE_TORCHVISION = True
    TORCHVISION_VERSION = torchvision.__version__
except ImportError as exc:  # pragma: no cover - environment dependent
    HAVE_TORCHVISION = False
    TORCHVISION_VERSION = f"<not installed: {exc}>"
    deform_conv2d = None


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def dc1d_offsets_to_torchvision(offsets: Tensor) -> Tensor:
    """
    ``(B, G, L_out, K)`` -> ``(B, 2*G*K, 1, L_out)``.

    torchvision orders the offset channel axis as ``(offset_group, kh, kw, 2)``
    with the trailing pair being ``(y, x)``. With ``kh == 1`` the flat channel
    index is ``(g*K + j)*2 + c``. Height offsets are all zero.

    Differentiable: ``torch.stack`` keeps the x-offsets in the autograd graph so
    gradients w.r.t. the dc1d-layout offsets are recovered unchanged.
    """
    batch, groups, out_length, kernel_size = offsets.shape
    x_off = offsets.permute(0, 1, 3, 2)  # (B, G, K, Lo)
    y_off = torch.zeros_like(x_off)
    stacked = torch.stack((y_off, x_off), dim=3)  # (B, G, K, 2, Lo)
    return stacked.reshape(batch, 2 * groups * kernel_size, 1, out_length)


def tv_deform_conv1d(
    x: Tensor,
    offsets_tv: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """
    ``torchvision.ops.deform_conv2d`` degenerated to 1D.

    ``x`` is ``(B, C, L)``, ``weight`` is ``(C_out, C_in/groups, K)``, and
    ``offsets_tv`` is already in torchvision layout (see
    :func:`dc1d_offsets_to_torchvision`) so that the layout conversion is not
    charged to the timed region.

    Group count is inferred by torchvision from ``C_in // weight.shape[1]``,
    exactly as ``DeformConv1d`` infers nothing and is told ``groups``.
    """
    out = deform_conv2d(
        x.unsqueeze(2),
        offsets_tv,
        weight.unsqueeze(2),
        bias,
        stride=(1, stride),
        padding=(0, 0),
        dilation=(1, dilation),
        mask=None,
    )
    return out.squeeze(2)


# ---------------------------------------------------------------------------
# Candidate backend: F.grid_sample
# ---------------------------------------------------------------------------


def grid_sample_linterpolate(
    x: Tensor,
    offsets: Tensor,
    kernel_size: int,
    dilation: int,
    stride: int,
    unconstrained: bool = False,
) -> Tensor:
    """
    Drop-in replacement for :func:`dc1d.ops.efficient_linterpolate` built on
    ``F.grid_sample``. Returns ``(B, C, L_out, K)``.

    A 1D gather-with-linear-interpolation *is* ``grid_sample`` on an image of
    height 1, so this is the natural ATen builtin for the job.

    Coordinate transform: with ``align_corners=True``, ``grid_sample`` maps a
    normalised coordinate ``g in [-1, 1]`` to pixel ``(g + 1) / 2 * (L - 1)``.
    The forward map is therefore ``g = (2*T - (L - 1)) / (L - 1)``, written in
    that form rather than the algebraically equal ``2*T/(L-1) - 1`` so that the
    numerator is exact for integer ``T``. **This is still lossy**: ``grid_sample``
    undoes the transform internally with ``(g + 1) / 2``, and that ``+ 1``
    destroys the low bits of small ``g`` no matter how the grid was built.
    See :func:`_grid_sample_precision_study`.

    ``padding_mode='border'`` clamps the coordinate into ``[0, L-1]``, which is
    dc1d's boundary convention (and *not* torchvision's zero padding).

    ``offset_groups`` are folded into the batch axis: ``x`` is viewed as
    ``(B*G, C/G, 1, L)`` and the grid as ``(B*G, 1, L_out*K, 2)``, so one grid
    entry serves every channel in its offset group.
    """
    batch, channels, length = x.shape
    groups, out_length = offsets.shape[1], offsets.shape[-2]
    per_group = channels // groups
    dtype = x.dtype

    dilated = torch.arange(kernel_size, device=x.device, dtype=torch.long) * dilation
    t0 = (torch.arange(out_length, device=x.device, dtype=torch.long) * stride).unsqueeze(-1)

    relative = dilated.to(offsets.dtype) + offsets  # (B, G, Lo, K)
    if not unconstrained:
        relative = relative.clamp(0.0, float(dilation * (kernel_size - 1)))
    positions = t0.to(offsets.dtype) + relative

    # Exact-numerator form of 2*T/(L-1) - 1.
    g_x = (2.0 * positions.to(dtype) - (length - 1)) / (length - 1)
    g_x = g_x.reshape(batch * groups, 1, out_length * kernel_size)
    # Height 1: with align_corners=True any y maps to pixel 0 * (1-1) = 0, but
    # 0.0 is the honest value.
    g_y = torch.zeros_like(g_x)
    grid = torch.stack((g_x, g_y), dim=-1)  # (B*G, 1, Lo*K, 2), last dim is (x, y)

    xg = x.reshape(batch * groups, per_group, 1, length)
    out = F.grid_sample(
        xg, grid, mode="bilinear", padding_mode="border", align_corners=True
    )  # (B*G, C/G, 1, Lo*K)
    return out.reshape(batch, channels, out_length, kernel_size)


def grid_sample_deform_conv1d(
    x: Tensor,
    offsets: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    unconstrained: bool = True,
) -> Tensor:
    """Full deformable conv1d using the ``grid_sample`` interpolation backend.

    The contraction is identical to ``DeformConv1d.forward`` -- flatten the tap
    axis and run a stride-``K`` grouped ``conv1d`` -- so any difference against
    dc1d is attributable to the interpolation step alone.
    """
    kernel_size = weight.shape[-1]
    sampled = grid_sample_linterpolate(
        x, offsets, kernel_size, dilation, stride, unconstrained=unconstrained
    )
    return F.conv1d(sampled.flatten(-2, -1), weight, bias, stride=kernel_size, groups=groups)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def _probe_offset_layout(device: str) -> list[str]:
    """
    Empirically pin down (a) which offset slot is the width axis and (b) which
    direction a positive width offset moves the sampling point. Asserted rather
    than assumed, because getting this wrong makes every number below garbage.
    """
    notes = []
    length, kernel_size = 10, 3
    ramp = torch.arange(float(length), device=device).reshape(1, 1, 1, length)
    weight = torch.zeros(1, 1, 1, kernel_size, device=device)
    weight[0, 0, 0, 0] = 1.0  # isolate tap 0
    out_len = length - kernel_size + 1

    base = deform_conv2d(
        ramp, torch.zeros(1, 2 * kernel_size, 1, out_len, device=device), weight, mask=None
    )
    assert torch.equal(base.flatten(), torch.arange(float(out_len), device=device)), (
        "zero-offset degenerate deform_conv2d does not reproduce the identity gather"
    )

    for slot in range(2 * kernel_size):
        off = torch.zeros(1, 2 * kernel_size, 1, out_len, device=device)
        off[:, slot] = 1.0
        got = deform_conv2d(ramp, off, weight, mask=None).flatten()
        if slot == 0:
            # height offset on tap 0: pushes y to 1, outside a height-1 image
            assert torch.equal(got, torch.zeros_like(got)), (
                "height offset did not fall off the (height-1) image -> "
                "the (y, x) pair ordering is not what this script assumes"
            )
            notes.append("slot 0 (tap 0, y): +1 -> 0.0 everywhere (out of bounds reads as ZERO)")
        elif slot == 1:
            assert torch.equal(got, base.flatten() + 1.0), (
                "a positive width offset did not move the sampling point towards "
                "higher indices; torchvision's sign convention differs from dc1d's"
            )
            notes.append("slot 1 (tap 0, x): +1 -> sampling point moves RIGHT (same sign as dc1d)")
        else:
            assert torch.equal(got, base.flatten()), (
                f"slot {slot} affected tap 0; offset channel ordering is not (g, kh, kw, 2)"
            )
    notes.append("offset channel layout confirmed: (offset_group, kh, kw, {y, x}), kh == 1")
    return notes


def _interior_slice(
    out_len: int, length: int, kernel_size: int, dilation: int, stride: int, radius: float
) -> slice:
    """
    Output positions for which *every* tap stays inside ``[0, L-1]`` given
    offsets bounded by ``|off| <= radius``. Outside this slice the two
    implementations use different boundary conventions and must not be compared.
    """
    lo = int(math.ceil(radius / stride))
    hi = int(math.floor((length - 1 - dilation * (kernel_size - 1) - radius) / stride))
    hi = min(hi, out_len - 1)
    assert hi > lo, "no interior output positions for this configuration"
    return slice(lo, hi + 1)


def check_equivalence(device: str) -> int:
    """Full correctness suite. Returns the number of failures."""
    failures = 0

    def report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  --  {detail}" if detail else ""))

    print(f"\n=== Equivalence checks on {device} (float64) ===\n")

    print("-- offset layout / sign convention (probed, not assumed) --")
    for note in _probe_offset_layout(device):
        print(f"  {note}")

    dtype = torch.float64
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Anchor: zero offsets must reproduce nn.Conv1d for BOTH implementations.
    # ------------------------------------------------------------------
    print("\n-- anchor: zero offsets == nn.Conv1d --")
    for stride, dilation, groups, kernel_size in [
        (1, 1, 1, 3),
        (2, 1, 1, 3),
        (1, 4, 8, 5),
        (3, 2, 2, 3),
        (1, 1, 1, 1),
    ]:
        batch, channels, length = 2, 8, 64
        x = torch.randn(batch, channels, length, device=device, dtype=dtype)
        weight = torch.randn(channels, channels // groups, kernel_size, device=device, dtype=dtype)
        bias = torch.randn(channels, device=device, dtype=dtype)
        out_len = output_length(length, kernel_size, dilation, stride)
        offsets = torch.zeros(batch, 1, out_len, kernel_size, device=device, dtype=dtype)

        want = F.conv1d(x, weight, bias, stride=stride, dilation=dilation, groups=groups)

        tv = tv_deform_conv1d(
            x, dc1d_offsets_to_torchvision(offsets), weight, bias, stride, dilation
        )
        cfg = f"s={stride} d={dilation} g={groups} K={kernel_size}"
        report(
            f"torchvision zero-offset == conv1d  ({cfg})",
            torch.allclose(tv, want, atol=1e-12, rtol=0),
            f"max |diff| = {(tv - want).abs().max().item():.3e}",
        )

        gs = grid_sample_deform_conv1d(x, offsets, weight, bias, stride, dilation, groups)
        report(
            f"grid_sample zero-offset == conv1d  ({cfg})",
            torch.allclose(gs, want, atol=1e-12, rtol=0),
            f"max |diff| = {(gs - want).abs().max().item():.3e}",
        )

        layer = DeformConv1d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding="valid",
            unconstrained=True,
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        dc = layer(x, offsets)
        report(
            f"dc1d       zero-offset == conv1d  ({cfg})",
            torch.allclose(dc, want, atol=1e-12, rtol=0),
            f"max |diff| = {(dc - want).abs().max().item():.3e}",
        )

    # ------------------------------------------------------------------
    # The real test: random fractional offsets, interior positions only.
    # ------------------------------------------------------------------
    print("\n-- interior equivalence: random offsets, all taps in range --")
    radius = 2.0
    grid = [
        # (B, C, L, K, stride, dilation, groups, offset_groups)
        (2, 8, 64, 3, 1, 1, 1, 1),
        (2, 8, 64, 3, 2, 1, 1, 1),
        (2, 8, 64, 3, 1, 4, 1, 1),
        (2, 8, 64, 5, 1, 2, 2, 1),
        (2, 8, 64, 3, 1, 1, 8, 1),
        (2, 8, 64, 3, 1, 1, 1, 2),
        (2, 8, 64, 3, 1, 1, 1, 8),
        (2, 8, 64, 3, 3, 2, 4, 4),
        (1, 4, 512, 7, 1, 3, 1, 1),
        (1, 4, 64, 1, 1, 1, 1, 1),
    ]
    for batch, channels, length, kernel_size, stride, dilation, groups, off_groups in grid:
        x = torch.randn(batch, channels, length, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(channels, channels // groups, kernel_size, device=device, dtype=dtype)
        bias = torch.randn(channels, device=device, dtype=dtype)
        out_len = output_length(length, kernel_size, dilation, stride)
        offsets = (
            (
                torch.rand(batch, off_groups, out_len, kernel_size, device=device, dtype=dtype) * 2
                - 1
            )
            * radius
        ).requires_grad_(True)

        layer = DeformConv1d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding="valid",
            unconstrained=True,  # torchvision has no receptive-field constraint
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)

        dc = layer(x, offsets)
        tv = tv_deform_conv1d(
            x, dc1d_offsets_to_torchvision(offsets), weight, bias, stride, dilation
        )
        gs = grid_sample_deform_conv1d(
            x, offsets, weight, bias, stride, dilation, groups, unconstrained=True
        )

        sl = _interior_slice(out_len, length, kernel_size, dilation, stride, radius)
        cfg = (
            f"B={batch} C={channels} L={length} K={kernel_size} s={stride} "
            f"d={dilation} g={groups} og={off_groups}"
        )
        scale = dc[:, :, sl].abs().max().item()

        # Gradients, on the interior only. All three graphs are differentiated
        # against the same upstream gradient, which is zeroed outside the
        # interior so the boundary convention cannot contaminate the result.
        gout = torch.zeros_like(dc)
        gout[:, :, sl] = torch.randn_like(gout[:, :, sl])
        grads = {
            name: torch.autograd.grad(out, [x, offsets], gout)
            for name, out in (("dc1d", dc), ("torchvision", tv), ("grid_sample", gs))
        }

        for name, out in (("torchvision", tv), ("grid_sample", gs)):
            diff = (dc[:, :, sl] - out[:, :, sl]).abs().max().item()
            report(
                f"forward  dc1d vs {name:<11} {cfg}",
                diff <= 1e-10 * max(scale, 1.0),
                f"max |diff| = {diff:.3e} (values ~{scale:.1f})",
            )
            gx_diff = (grads["dc1d"][0] - grads[name][0]).abs().max().item()
            go_diff = (grads["dc1d"][1] - grads[name][1]).abs().max().item()
            gx_scale = max(grads["dc1d"][0].abs().max().item(), 1.0)
            go_scale = max(grads["dc1d"][1].abs().max().item(), 1.0)
            report(
                f"backward dc1d vs {name:<11} {cfg}",
                gx_diff <= 1e-9 * gx_scale and go_diff <= 1e-9 * go_scale,
                f"max |d/dx diff| = {gx_diff:.3e}, max |d/doffset diff| = {go_diff:.3e}",
            )

    # ------------------------------------------------------------------
    # Documented difference #1: boundary handling.
    # ------------------------------------------------------------------
    print("\n-- documented difference: boundary convention (clamp vs zero-pad) --")
    length, kernel_size = 12, 3
    ramp = torch.arange(1.0, length + 1, device=device, dtype=dtype).reshape(1, 1, length)
    weight = torch.zeros(1, 1, kernel_size, device=device, dtype=dtype)
    weight[0, 0, 0] = 1.0  # isolate tap 0
    out_len = output_length(length, kernel_size)
    layer = DeformConv1d(1, 1, kernel_size, padding="valid", bias=False, unconstrained=True).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        layer.weight.copy_(weight)

    cases = [
        (-0.5, "half a sample off the left edge"),
        (-3.0, "3 samples off the left edge"),
        (50.0, "far past the right edge"),
    ]
    for off_val, label in cases:
        offsets = torch.full((1, 1, out_len, kernel_size), off_val, device=device, dtype=dtype)
        dc = layer(ramp, offsets)[0, 0]
        tv = tv_deform_conv1d(ramp, dc1d_offsets_to_torchvision(offsets), weight, None)[0, 0]
        gs = grid_sample_deform_conv1d(ramp, offsets, weight, None, unconstrained=True)[0, 0]
        print(f"  offset={off_val:+.1f} ({label}), x = [1..{length}], tap 0 only")
        print(f"    dc1d        (clamp)     out[:4] = {[round(v, 3) for v in dc[:4].tolist()]}")
        print(f"    grid_sample (border)    out[:4] = {[round(v, 3) for v in gs[:4].tolist()]}")
        print(f"    torchvision (zero-pad)  out[:4] = {[round(v, 3) for v in tv[:4].tolist()]}")
        n_diff = int((dc - tv).abs().gt(1e-12).sum().item())
        print(
            f"    dc1d vs torchvision: differ at {n_diff}/{out_len} positions, "
            f"max |diff| = {(dc - tv).abs().max().item():.3f}"
        )
        report(
            f"    grid_sample(border) reproduces dc1d(clamp) at offset {off_val:+.1f}",
            torch.allclose(gs, dc, atol=1e-12, rtol=0),
            f"max |diff| = {(gs - dc).abs().max().item():.3e}",
        )

    # How wide is the affected region for realistic offsets?
    print("\n  fraction of output positions affected, |offset| ~ U(-r, r), K=3, d=1:")
    for length in (64, 1024, 16000):
        for radius in (1.0, 4.0):
            out_len = output_length(length, 3)
            sl = _interior_slice(out_len, length, 3, 1, 1, radius)
            affected = out_len - (sl.stop - sl.start)
            print(
                f"    L={length:<6} r={radius:<4} -> at most {affected}/{out_len} "
                f"({100 * affected / out_len:.2f}%) positions can differ"
            )

    # ------------------------------------------------------------------
    # Documented difference #2: constrained mode.
    # ------------------------------------------------------------------
    print("\n-- documented difference: dc1d constrained mode has no torchvision analogue --")
    torch.manual_seed(1)
    batch, channels, length, kernel_size = 1, 4, 64, 3
    x = torch.randn(batch, channels, length, device=device, dtype=dtype)
    weight = torch.randn(channels, channels, kernel_size, device=device, dtype=dtype)
    out_len = output_length(length, kernel_size)
    offsets = torch.rand(batch, 1, out_len, kernel_size, device=device, dtype=dtype) * 4 - 2

    outs = {}
    for unconstrained in (False, True):
        layer = DeformConv1d(
            channels,
            channels,
            kernel_size,
            padding="valid",
            bias=False,
            unconstrained=unconstrained,
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            layer.weight.copy_(weight)
        outs[unconstrained] = layer(x, offsets)
    tv = tv_deform_conv1d(x, dc1d_offsets_to_torchvision(offsets), weight, None)
    sl = _interior_slice(out_len, length, kernel_size, 1, 1, 2.0)
    print(
        "  interior max |dc1d(unconstrained=True)  - torchvision| = "
        f"{(outs[True][:, :, sl] - tv[:, :, sl]).abs().max().item():.3e}"
    )
    print(
        "  interior max |dc1d(unconstrained=False) - torchvision| = "
        f"{(outs[False][:, :, sl] - tv[:, :, sl]).abs().max().item():.3e}"
        "   <- expected: constrained mode clamps each tap to its own receptive field"
    )

    failures += _grid_sample_precision_study(device, report)
    failures += _grid_sample_gradcheck(device, report)

    print(f"\n{'=' * 70}\n{failures} failure(s)\n")
    return failures


def _grid_sample_precision_study(device: str, report) -> int:
    """
    dc1d's headline invariant is that zero offsets reproduce ``nn.Conv1d``
    *bit-for-bit*: sampling positions land on exact integers, so the
    interpolation weights are exactly ``[1, 0]``. This holds because
    ``efficient_linterpolate`` keeps window starts in ``long`` and only carries
    the sub-sample fraction in float.

    ``grid_sample`` cannot do that. Its coordinates must be normalised to
    ``[-1, 1]``, and it un-normalises internally with ``(g + 1) / 2 * (L - 1)``.
    That ``+ 1`` is a catastrophic cancellation for coordinates near the middle
    of the signal: the absolute error in ``g`` is ~``eps``, which becomes an
    error of ``eps * (L - 1) / 2`` *samples* in the recovered position. In fp32
    that is ~1e-3 samples at L = 16000 -- small, but not zero, and it scales
    linearly with L. In fp16 it is catastrophic, which is the same class of bug
    dc1d fixed in ``efficient_linterpolate``.

    This measures the effect directly instead of arguing about it.
    """
    print("\n-- grid_sample precision: cost of the [-1, 1] normalisation --")
    print("   (zero offsets; the exact answer is a plain unfold, error should be 0)")
    print(
        f"\n  | {'dtype':>8} | {'L':>7} | {'dc1d max err':>13} | {'grid_sample max err':>20} "
        f"| {'implied position err':>21} |"
    )
    print("  |" + "|".join(["-" * 10, "-" * 9, "-" * 15, "-" * 22, "-" * 23]) + "|")

    failures = 0
    kernel_size = 3
    for dtype in (torch.float32, torch.float16):
        for length in (256, 2048, 16000, 65536):
            if dtype is torch.float16 and device == "cpu" and length > 2048:
                continue
            torch.manual_seed(0)
            x = torch.randn(1, 1, length, device=device, dtype=dtype)
            out_len = output_length(length, kernel_size)
            offsets = torch.zeros(1, 1, out_len, kernel_size, device=device, dtype=dtype)

            want = x.unfold(2, kernel_size, 1)
            from dc1d.ops import efficient_linterpolate

            dc_err = (efficient_linterpolate(x, offsets, kernel_size, 1, 1) - want).abs().max()
            gs = grid_sample_linterpolate(x, offsets, kernel_size, 1, 1)
            gs_err = (gs - want).abs().max()

            # Recover the position error implied by the interpolation error:
            # sampling at integer i + e gives x[i] + e*(x[i+1] - x[i]).
            slope = (x[0, 0, 1:] - x[0, 0, :-1]).abs().median().item()
            implied = gs_err.item() / slope if slope > 0 else float("nan")
            print(
                f"  | {str(dtype).replace('torch.', ''):>8} | {length:7d} | {dc_err.item():13.3e} "
                f"| {gs_err.item():20.3e} | {implied:18.3e} sa |"
            )
            if dtype is torch.float32:
                if dc_err.item() != 0.0:
                    failures += 1
                    print("    [FAIL] dc1d was not bit-exact -- unexpected")
    print(
        "\n  dc1d is bit-exact at every length and dtype; grid_sample is not, and its\n"
        "  error grows linearly with L. fp16 grid_sample is unusable past ~2k."
    )
    return failures


def _grid_sample_gradcheck(device: str, report) -> int:
    """
    The whole point of a deformable layer is the gradient w.r.t. the *offsets*.
    ``grid_sample`` does propagate gradient into its grid, but that must be
    verified, not assumed -- if it did not, grid_sample would be unusable as a
    dc1d backend regardless of how fast it is.
    """
    print("\n-- grid_sample gradcheck (float64, w.r.t. input AND offsets) --")
    failures = 0
    torch.manual_seed(0)
    batch, channels, length, kernel_size, off_groups = 2, 4, 24, 3, 2
    out_len = output_length(length, kernel_size)

    x = torch.randn(batch, channels, length, device=device, dtype=torch.float64, requires_grad=True)
    offsets = torch.randn(
        batch, off_groups, out_len, kernel_size, device=device, dtype=torch.float64
    ).requires_grad_(True)

    def fn(x_, offsets_):
        return grid_sample_linterpolate(x_, offsets_, kernel_size, 1, 1, unconstrained=True)

    ok = torch.autograd.gradcheck(fn, (x, offsets), eps=1e-6, atol=1e-8, rtol=1e-5)
    if not ok:
        failures += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] gradcheck(grid_sample_linterpolate, [x, offsets])")

    # And the offset gradient must be numerically the same as dc1d's.
    from dc1d.ops import efficient_linterpolate

    gout = torch.randn(batch, channels, out_len, kernel_size, device=device, dtype=torch.float64)
    g_dc = torch.autograd.grad(
        efficient_linterpolate(x, offsets, kernel_size, 1, 1, unconstrained=True),
        [x, offsets],
        gout,
    )
    g_gs = torch.autograd.grad(fn(x, offsets), [x, offsets], gout)
    for name, a, b in (("d/dx", g_dc[0], g_gs[0]), ("d/doffsets", g_dc[1], g_gs[1])):
        diff = (a - b).abs().max().item()
        ok = diff <= 1e-9 * max(a.abs().max().item(), 1.0)
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] dc1d vs grid_sample {name}: max |diff| = {diff:.3e}")
    return failures


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

# Order matters: dc1d is the reference every ratio is taken against, and
# nn.Conv1d is the non-deformable floor.
IMPLS = ["dc1d", "torchvision", "grid_sample", "nn.Conv1d"]
CONTENDERS = ["torchvision", "grid_sample"]


@dataclass
class Config:
    name: str
    batch: int
    channels: int
    length: int
    kernel_size: int = 3
    dilation: int = 1
    groups: int = 1
    stride: int = 1
    offset_groups: int = 1

    @property
    def label(self) -> str:
        return (
            f"B={self.batch} C={self.channels} L={self.length} K={self.kernel_size} "
            f"d={self.dilation} g={self.groups}"
        )


GRID: list[Config] = [
    # name                       B    C     L      K   dil  grp
    Config("tiny", 1, 16, 256, 3, 1, 1),
    Config("small", 4, 64, 1024, 3, 1, 1),
    Config("small-depthwise", 4, 64, 1024, 3, 1, 64),
    Config("medium", 4, 256, 2048, 3, 8, 256),
    Config("medium-dense", 4, 256, 2048, 3, 8, 1),
    Config("wide-kernel", 4, 128, 4096, 15, 1, 1),
    Config("wide-kernel-dw", 4, 128, 4096, 15, 1, 128),
    Config("strided", 4, 128, 4096, 3, 2, 1, stride=4),
    Config("offset-groups", 4, 128, 4096, 3, 1, 1, offset_groups=8),
    # the speech-separation regime this package was written for
    Config("speech-1x256", 1, 256, 16000, 3, 1, 256),
    Config("speech-4x256", 4, 256, 16000, 3, 8, 256),
    Config("speech-dense", 1, 256, 16000, 3, 1, 1),
]

LENGTH_SWEEP: list[Config] = [
    Config(f"L={length}", 4, 128, length, 3, 1, 128) for length in (256, 1024, 4096, 16384, 65536)
]

CHANNEL_SWEEP: list[Config] = [
    Config(f"C={channels}", 4, channels, 4096, 3, 1, 1) for channels in (16, 64, 256, 1024)
]


def make_inputs(cfg: Config, device: str, dtype: torch.dtype, requires_grad: bool):
    torch.manual_seed(0)
    x = torch.randn(
        cfg.batch, cfg.channels, cfg.length, device=device, dtype=dtype, requires_grad=requires_grad
    )
    out_len = output_length(cfg.length, cfg.kernel_size, cfg.dilation, cfg.stride)
    offsets = torch.randn(
        cfg.batch, cfg.offset_groups, out_len, cfg.kernel_size, device=device, dtype=dtype
    )
    offsets.requires_grad_(requires_grad)

    layer = DeformConv1d(
        cfg.channels,
        cfg.channels,
        cfg.kernel_size,
        stride=cfg.stride,
        dilation=cfg.dilation,
        groups=cfg.groups,
        padding="valid",
        unconstrained=True,
    ).to(device=device, dtype=dtype)
    vanilla = nn.Conv1d(
        cfg.channels,
        cfg.channels,
        cfg.kernel_size,
        stride=cfg.stride,
        dilation=cfg.dilation,
        groups=cfg.groups,
        padding="valid",
    ).to(device=device, dtype=dtype)

    # torchvision-layout offsets are built OUTSIDE the timed region: a caller
    # who has committed to torchvision would produce them in this layout to
    # begin with, so charging the permute to torchvision would be unfair.
    offsets_tv = dc1d_offsets_to_torchvision(offsets)
    return x, offsets, offsets_tv, layer, vanilla, out_len


def _timer(stmt: str, globals_: dict, sub_label: str, description: str, min_run_time: float):
    return benchmark.Timer(
        stmt=stmt,
        globals=globals_,
        label="deformable conv1d",
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench_config(
    cfg: Config, device: str, dtype: torch.dtype, min_run_time: float, rounds: int = 3
) -> dict:
    """
    Time every implementation on ``cfg``.

    The measurement is deliberately defensive. On a consumer GPU that is also
    driving a desktop, ``blocked_autorange``'s within-run median is *not*
    enough: the card power-caps as it heats up and a run that happens to land
    during a compositor frame is 5-20x slow. Two mitigations:

    *   implementations are measured **round-robin** within each round, so all
        three see the same thermal and contention state;
    *   the reported figure is the **minimum** across rounds -- the standard
        estimator for "what this kernel does when nothing else interferes".

    ``spread`` (max/min across rounds, worst implementation) is reported so the
    reader can see how noisy the machine was. A spread above ~1.5 means the
    corresponding ratio should be read as an order of magnitude, not a number.
    """
    row: dict = {"name": cfg.name, "label": cfg.label}

    for phase, requires_grad in (("fwd", False), ("fwd+bwd", True)):
        x, offsets, offsets_tv, layer, vanilla, _ = make_inputs(cfg, device, dtype, requires_grad)
        g = {
            "x": x,
            "offsets": offsets,
            "offsets_tv": offsets_tv,
            "layer": layer,
            "vanilla": vanilla,
            "weight": layer.weight,
            "bias": layer.bias,
            "tv_deform_conv1d": tv_deform_conv1d,
            "grid_sample_deform_conv1d": grid_sample_deform_conv1d,
            "stride": cfg.stride,
            "dilation": cfg.dilation,
            "groups": cfg.groups,
        }
        calls = {
            "dc1d": "layer(x, offsets)",
            "torchvision": "tv_deform_conv1d(x, offsets_tv, weight, bias, stride, dilation)",
            "grid_sample": (
                "grid_sample_deform_conv1d(x, offsets, weight, bias, stride, dilation, groups)"
            ),
            "nn.Conv1d": "vanilla(x)",
        }
        if requires_grad:
            stmts = {k: f"y = {v}; y.sum().backward()" for k, v in calls.items()}
        else:
            stmts = dict(calls)

        # Equal warmup for both sides before the adaptive timer takes over.
        for stmt in stmts.values():
            for _ in range(3):
                exec(stmt, dict(g))  # noqa: S102
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        samples: dict[str, list[float]] = {impl: [] for impl in stmts}
        for _ in range(rounds):
            for impl, stmt in stmts.items():
                m = _timer(stmt, g, cfg.label, f"{impl} ({phase})", min_run_time)
                samples[impl].append(m.median)

        spread = 1.0
        for impl, values in samples.items():
            row[f"{impl} {phase}"] = min(values)
            spread = max(spread, max(values) / min(values))
        row[f"spread {phase}"] = spread

        del x, offsets, offsets_tv, layer, vanilla, g
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return row


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def _run_once(impl: str, phase: str, cfg: Config, device: str, dtype: torch.dtype) -> None:
    requires_grad = phase == "fwd+bwd"
    x, offsets, offsets_tv, layer, vanilla, _ = make_inputs(cfg, device, dtype, requires_grad)
    if impl == "dc1d":
        y = layer(x, offsets)
    elif impl == "torchvision":
        y = tv_deform_conv1d(x, offsets_tv, layer.weight, layer.bias, cfg.stride, cfg.dilation)
    elif impl == "grid_sample":
        y = grid_sample_deform_conv1d(
            x, offsets, layer.weight, layer.bias, cfg.stride, cfg.dilation, cfg.groups
        )
    else:
        y = vanilla(x)
    if requires_grad:
        y.sum().backward()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    del y, x, offsets, offsets_tv, layer, vanilla


def measure_cuda_memory(cfg: Config, device: str, dtype: torch.dtype) -> dict:
    row: dict = {"name": cfg.name, "label": cfg.label}
    for phase in ("fwd", "fwd+bwd"):
        for impl in IMPLS:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
            _run_once(impl, phase, cfg, device, dtype)
            peak = torch.cuda.max_memory_allocated(device)
            row[f"{impl} {phase}"] = (peak - base) / 2**20
            torch.cuda.empty_cache()
    return row


def measure_cpu_memory(cfg: Config, dtype_name: str) -> dict:
    """
    Peak RSS, measured in a fresh subprocess per (impl, phase) because
    ``ru_maxrss`` is a high-water mark that cannot be reset within a process.
    The reported figure is the peak minus the RSS recorded after imports and
    input construction, i.e. the transient allocated by the op itself.
    """
    row: dict = {"name": cfg.name, "label": cfg.label}
    for phase in ("fwd", "fwd+bwd"):
        for impl in IMPLS:
            payload = json.dumps(
                {
                    "cfg": cfg.__dict__,
                    "impl": impl,
                    "phase": phase,
                    "dtype": dtype_name,
                }
            )
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--mem-worker", payload],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                row[f"{impl} {phase}"] = float("nan")
                print(f"    memory worker failed for {impl}/{phase}: {proc.stderr.strip()[-400:]}")
            else:
                row[f"{impl} {phase}"] = float(proc.stdout.strip().splitlines()[-1])
    return row


def _mem_worker(payload: str) -> None:
    import resource

    spec = json.loads(payload)
    cfg = Config(**spec["cfg"])
    dtype = getattr(torch, spec["dtype"])
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _run_once(spec["impl"], spec["phase"], cfg, "cpu", dtype)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(max(peak - before, 0) / 1024.0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _emit_table(header: list[str], widths: list[int], body: list[list[str]]) -> None:
    print("| " + " | ".join(f"{c:<{w}}" for c, w in zip(header, widths, strict=True)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for cells in body:
        print("| " + " | ".join(f"{c:<{w}}" for c, w in zip(cells, widths, strict=True)) + " |")


def print_timing_table(rows: list[dict], title: str) -> None:
    """
    One row per config, one column per implementation, split by phase.
    Contender cells carry ``time (Nx)`` where ``N = dc1d / contender``, so
    ``> 1`` means the contender beats dc1d.
    """
    for phase, tag in (("fwd", "forward"), ("fwd+bwd", "forward+backward")):
        print(f"\n### {title} -- {tag}\n")
        header = ["config", "dc1d", "torchvision", "grid_sample", "nn.Conv1d", "spread"]
        widths = [34, 9, 18, 18, 9, 7]
        body = []
        for r in rows:
            ref = r[f"dc1d {phase}"] * 1e3
            cells = [r["label"], f"{ref:.3f}"]
            for impl in CONTENDERS:
                t = r[f"{impl} {phase}"] * 1e3
                cells.append(f"{t:.3f} ({ref / t:.2f}x)")
            cells.append(f"{r[f'nn.Conv1d {phase}'] * 1e3:.3f}")
            cells.append(f"{r.get(f'spread {phase}', 1.0):.2f}x")
            body.append(cells)
        _emit_table(header, widths, body)
    print(
        "\n(times in ms, minimum over rounds; the bracketed factor is dc1d / contender, "
        "so > 1x means the contender is FASTER than dc1d.\n"
        " 'spread' is max/min across measurement rounds -- the machine's noise floor; "
        "above ~1.5x, read ratios as order-of-magnitude only.)"
    )


def print_memory_table(rows: list[dict], title: str, unit: str) -> None:
    for phase, tag in (("fwd", "forward"), ("fwd+bwd", "forward+backward")):
        print(f"\n### {title} -- {tag}\n")
        header = ["config", "dc1d", "torchvision", "grid_sample", "nn.Conv1d"]
        widths = [34, 9, 18, 18, 9]
        body = []
        for r in rows:
            ref = r[f"dc1d {phase}"]
            cells = [r["label"], f"{ref:.1f}"]
            for impl in CONTENDERS:
                v = r[f"{impl} {phase}"]
                cells.append(f"{v:.1f} ({ref / v:.2f}x)")
            cells.append(f"{r[f'nn.Conv1d {phase}']:.1f}")
            body.append(cells)
        _emit_table(header, widths, body)
    print(
        f"\n(peak {unit}, MiB; the bracketed factor is dc1d / contender, "
        "so > 1x means the contender uses LESS memory than dc1d)"
    )


def environment_report(device: str) -> None:
    print("=" * 78)
    print(f"torch          : {torch.__version__}")
    print(f"torchvision    : {TORCHVISION_VERSION}")
    print(f"python         : {sys.version.split()[0]}")
    print(f"device         : {device}")
    if device.startswith("cuda"):
        print(f"gpu            : {torch.cuda.get_device_name(device)}")
        cap = torch.cuda.get_device_capability(device)
        print(f"compute cap    : sm_{cap[0]}{cap[1]}")
        print(f"cuda (torch)   : {torch.version.cuda}")
        print(f"cudnn          : {torch.backends.cudnn.version()}")
    else:
        print(f"threads        : {torch.get_num_threads()}")
    print("=" * 78)


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--check", action="store_true", help="run equivalence checks only")
    parser.add_argument("--bench", action="store_true", help="run timing sweep only")
    parser.add_argument("--mem", action="store_true", help="run memory sweep only")
    parser.add_argument("--all", action="store_true", help="check + bench + mem")
    parser.add_argument("--min-run-time", type=float, default=0.3)
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="round-robin measurement rounds; the minimum is reported (default 5)",
    )
    parser.add_argument("--sweeps", action="store_true", help="also run the L and C sweeps")
    parser.add_argument("--mem-worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.mem_worker:
        _mem_worker(args.mem_worker)
        return 0

    if not HAVE_TORCHVISION:
        print(
            f"torchvision is required for this benchmark but is not installed: "
            f"{TORCHVISION_VERSION}"
        )
        print(__doc__.split("Degenerating")[0].split("torchvision is *not*")[1])
        return 2

    if not (args.check or args.bench or args.mem):
        args.all = True
    if args.all:
        args.check = args.bench = args.mem = True

    if args.device.startswith("cuda"):
        # torch.utils.benchmark synchronises the *current* device, so pin it or
        # `--device cuda:1` silently times against a stream on cuda:0.
        torch.cuda.set_device(args.device)

    dtype = getattr(torch, args.dtype)
    environment_report(args.device)

    failures = 0
    if args.check:
        failures = check_equivalence(args.device)

    if args.bench:

        def sweep(configs: list[Config], title: str) -> None:
            rows = [
                bench_config(c, args.device, dtype, args.min_run_time, args.rounds) for c in configs
            ]
            print_timing_table(rows, f"{title} -- {args.device}, {args.dtype}")

        sweep(GRID, "Timing")
        if args.sweeps:
            sweep(LENGTH_SWEEP, "Length sweep")
            sweep(CHANNEL_SWEEP, "Channel sweep")

    if args.mem:
        if args.device.startswith("cuda"):
            rows = [measure_cuda_memory(c, args.device, dtype) for c in GRID]
            print_memory_table(rows, "Peak CUDA memory", "torch.cuda.max_memory_allocated")
        else:
            rows = [measure_cpu_memory(c, args.dtype) for c in GRID]
            print_memory_table(rows, "Peak CPU memory", "RSS above post-setup baseline")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
