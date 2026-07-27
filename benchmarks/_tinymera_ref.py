"""
Vendored copy of tinymera's independent 1D deformable convolution kernels.

Provenance
----------
Repository : https://github.com/jwr1995/tinymera  (private)
Branch     : ``fix/causality``  (open as tinymera PR #1 against
             ``feature/exciting-plc``)
Commit     : 04593f38ce162c7e72dac061b1fc6eef3a2fd0b0  (2026-07-27)
Source     : ``tinymera/ops/deform_conv1d.py``
             blob eb1dd701a606dbf22db26118a53a223a6c8fc278

The two functions below are **verbatim copies** of ``deform_conv1d_grid_sample``
and ``deform_conv1d_gather`` from that file, with only the module docstring and
imports adapted. They are vendored rather than imported because tinymera is a
private application repo, is not pip-installable, and must not become a
dependency of dc1d -- this file exists purely so that
``benchmarks/backends.py`` can benchmark against a second, independently
written implementation of the same operator.

Do not "fix" anything in here. The point of the comparison is to measure what
tinymera actually does. Defects found in this code are recorded in
``benchmarks/BACKENDS.md`` and belong upstream in tinymera, not here.

Interface differences from dc1d worth knowing before reading the numbers
------------------------------------------------------------------------
*   **Offsets are always per-channel**: ``offsets`` is ``(B, C_in, T_out, K)``.
    There is no ``offset_groups`` knob -- tinymera is hardwired to the
    ``offset_groups == in_channels`` case, which is dc1d's most expensive
    setting. Comparisons at ``offset_groups < C`` therefore require broadcasting
    the offsets up to ``C``, which is done outside the timed region.
*   **No constrained mode.** dc1d's default ``unconstrained=False`` (each tap
    confined to its own receptive field) has no tinymera analogue. Compare
    against ``DeformConv1d(..., unconstrained=True)``.
*   **Boundary convention is a clamp** (``pos.clamp(0, T_in - 1)`` in the gather
    kernel, ``padding_mode='border'`` in the grid_sample kernel), which is the
    *same* convention as dc1d and *not* torchvision's zero padding.
*   ``causal=True`` clamps offsets to ``<= 0``. dc1d has no causal mode; the
    benchmark always passes ``causal=False``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["deform_conv1d_gather", "deform_conv1d_grid_sample"]


# ── grid_sample kernel ────────────────────────────────────────────────────────


def deform_conv1d_grid_sample(
    x: Tensor,
    offsets: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    causal: bool = False,
) -> Tensor:
    """
    1D deformable convolution via ``F.grid_sample`` (bilinear, align_corners=True).

    Args:
        x:       (B, C_in, T_in)
        offsets: (B, C_in, T_out, K)
        weight:  (C_out, C_in//groups, K)
        bias:    (C_out,) or None
        stride:  output step size in input samples
        dilation: spacing between kernel taps
        groups:  number of convolution groups (C_in must be divisible by groups)
        causal:  when True, offsets are clamped to ``<= 0`` so no tap can sample
                 forward in time

    Returns:
        (B, C_out, T_out)
    """
    B, C_in, T_in = x.shape
    _, _, T_out, K = offsets.shape
    C_out = weight.shape[0]

    # ── nominal tap positions for each (t_out, k) ─────────────────────────────
    # Always fp32: index arithmetic in bf16/fp16 quantises or overflows.
    t_out = torch.arange(T_out, device=x.device, dtype=torch.float32)  # (T_out,)
    k_idx = torch.arange(K, device=x.device, dtype=torch.float32)  # (K,)
    # nominal[t, k] = t*stride + k*dilation
    nominal = t_out[:, None] * stride + k_idx[None, :] * dilation  # (T_out, K)

    # ── absolute fractional positions ─────────────────────────────────────────
    off = offsets.float()
    if causal:
        off = off.clamp(max=0.0)
    # pos: (B, C_in, T_out, K) — clamped to [0, T_in-1]
    pos = nominal.unsqueeze(0).unsqueeze(0) + off
    pos = pos.clamp(0.0, float(T_in - 1))

    # ── fold C into B so each channel gets its own grid ───────────────────────
    x_4d = x.float().reshape(B * C_in, 1, 1, T_in)

    grid_x = pos.reshape(B * C_in, T_out, K) * (2.0 / (T_in - 1)) - 1.0
    grid_y = torch.zeros_like(grid_x)

    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B*C_in, T_out, K, 2)
    grid = grid.permute(0, 2, 1, 3)  # (B*C_in, K, T_out, 2)

    sampled = F.grid_sample(
        x_4d, grid, mode="bilinear", padding_mode="border", align_corners=True
    )  # (B*C_in, 1, K, T_out)

    sampled = sampled.squeeze(1).reshape(B, C_in, K, T_out).to(x.dtype)

    # ── weighted sum across taps ──────────────────────────────────────────────
    C_per_group = C_in // groups
    sampled_g = sampled.reshape(B, groups, C_per_group, K, T_out)
    w_g = weight.reshape(groups, C_out // groups, C_per_group, K)
    out = torch.einsum("bgckt,gock->bgot", sampled_g, w_g)
    out = out.reshape(B, C_out, T_out)

    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out


# ── gather kernel ─────────────────────────────────────────────────────────────


def deform_conv1d_gather(
    x: Tensor,
    offsets: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    causal: bool = False,
) -> Tensor:
    """
    1D deformable convolution via floor/ceil gather + linear interpolation.

    Args:
        x:       (B, C_in, T_in)
        offsets: (B, C_in, T_out, K)
        weight:  (C_out, C_in//groups, K)
        bias:    (C_out,) or None
        stride:  output step size in input samples
        dilation: spacing between kernel taps
        groups:  number of convolution groups
        causal:  when True, offsets are clamped to ``<= 0``

    Returns:
        (B, C_out, T_out)
    """
    B, C_in, T_in = x.shape
    _, _, T_out, K = offsets.shape
    C_out = weight.shape[0]

    # ── fractional sample positions ───────────────────────────────────────────
    t_out = torch.arange(T_out, device=x.device, dtype=torch.float32)
    k_idx = torch.arange(K, device=x.device, dtype=torch.float32)
    nominal = t_out[:, None] * stride + k_idx[None, :] * dilation  # (T_out, K)

    off = offsets.float()
    if causal:
        off = off.clamp(max=0.0)

    pos = nominal.unsqueeze(0).unsqueeze(0) + off  # (B, C_in, T_out, K)
    pos = pos.clamp(0.0, float(T_in - 1))

    # ── integer floor/ceil indices ────────────────────────────────────────────
    pos_floor = pos.floor().long()  # (B, C_in, T_out, K)
    pos_ceil = (pos_floor + 1).clamp(max=T_in - 1)
    frac = (pos - pos_floor.to(pos.dtype)).to(x.dtype)  # (B, C_in, T_out, K)

    # ── gather floor and ceil samples ─────────────────────────────────────────
    b_idx = torch.arange(B, device=x.device).view(B, 1, 1, 1)
    c_idx = torch.arange(C_in, device=x.device).view(1, C_in, 1, 1)

    x_floor = x[b_idx, c_idx, pos_floor]  # (B, C_in, T_out, K)
    x_ceil = x[b_idx, c_idx, pos_ceil]  # (B, C_in, T_out, K)

    # ── linear interpolation ──────────────────────────────────────────────────
    sampled = x_floor + frac * (x_ceil - x_floor)  # (B, C_in, T_out, K)

    # ── weighted sum across taps ──────────────────────────────────────────────
    C_per_group = C_in // groups
    sampled_g = sampled.reshape(B, groups, C_per_group, T_out, K)
    w_g = weight.reshape(groups, C_out // groups, C_per_group, K)
    out = torch.einsum("bgctk,gock->bgot", sampled_g, w_g)
    out = out.reshape(B, C_out, T_out)

    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out
