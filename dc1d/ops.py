"""
ops.py provides linear interpolation operation functions for deformable convolution

Author: William Ravenscroft, August 2022
Copyright William Ravenscroft 2022
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "efficient_linterpolate",
    "full_seq_linterpolate",
    "kernel_width_linterpolate",
    "output_length",
]

# Which gather+lerp implementation `efficient_linterpolate` uses by default.
# See `_GATHER_LERP_IMPLS` and `benchmarks/BACKENDS.md` section 5.9 for the
# measurements behind this choice.
DEFAULT_GATHER_LERP = "autograd"


def output_length(
    length: int,
    kernel_size: int,
    dilation: int = 1,
    stride: int = 1,
) -> int:
    """
    Number of output positions (equivalently, the number of offset positions the
    interpolation kernels expect) for a valid convolution over ``length`` samples.

    This is the same closed form used by :class:`torch.nn.Conv1d`::

        floor((length - dilation * (kernel_size - 1) - 1) / stride) + 1

    Args:
        length (int): Input sequence length (after any padding has been applied).
        kernel_size (int): Convolution kernel size.
        dilation (int): Convolution kernel dilation factor. Default: 1.
        stride (int): Convolution kernel stride. Default: 1.
    """
    receptive_field = dilation * (kernel_size - 1) + 1
    if length < receptive_field:
        return 0
    return (length - receptive_field) // stride + 1


def _dilated_positions_long(
    kernel_size: int,
    dilation: int,
    device: torch.device,
    dilated_positions: Tensor | None = None,
) -> Tensor:
    """
    Integer-valued kernel tap positions ``[0, d, 2d, ..., d*(k-1)]``.

    ``dilated_positions`` may be supplied (e.g. from a module buffer) to keep the
    historical API; it is only ever an integer ramp, so it is rounded to ``long``
    rather than being trusted as a float.
    """
    if dilated_positions is None:
        return torch.arange(kernel_size, device=device, dtype=torch.long) * dilation
    return dilated_positions.to(device=device).round().long()


# ---------------------------------------------------------------------------
# The gather + lerp core, and its backward.
#
# Every variant computes exactly the same forward:
#
#     x0 = x[i],  x1 = x[i + 1],  out = lerp(x0, x1, w)
#
# with ``i`` an integer index carrying no gradient and ``w`` the sub-sample
# fraction. They differ only in how the backward is obtained, and therefore in
# what has to be kept alive between the forward and the backward:
#
#   ``autograd``   let autograd differentiate through ``take_along_dim`` and
#                  ``lerp``. Correct, and the only variant that needs no
#                  hand-written maths -- but ``lerp``'s backward needs both
#                  ``x0`` and ``x1``, so two (B, C, L_out, K) tensors are held.
#   ``save-diff``  a custom Function that stores the single tensor the offset
#                  gradient actually needs, ``x1 - x0``. Half the saved
#                  footprint of ``autograd``.
#   ``recompute``  a custom Function that stores nothing but the index and the
#                  fraction and re-gathers ``x0``/``x1`` in the backward. Lowest
#                  saved footprint; pays two extra gathers.
#
# The backward maths, with ``g`` the upstream gradient:
#
#     dL/dx    scatter-add ``g * (1 - w)`` at ``i`` and ``g * w`` at ``i + 1``
#     dL/dw    ``sum_c g * (x1 - x0)``, summed over the channels sharing an
#              offset group; ``i`` is detached and ``dw/d(offset) == 1``, so
#              this is also dL/d(offset) before the group reduction.
#
# **Determinism.** ``dL/dx`` is a scatter-add, which on CUDA accumulates with
# atomics and is therefore *not* bitwise reproducible run to run. Measured, for
# all three variants alike -- this is not something the custom Function
# introduces, because the ``autograd`` variant's ``take_along_dim`` backward is
# the same scatter-add. ``dL/d(offsets)`` is a plain reduction and is
# reproducible. Under ``torch.use_deterministic_algorithms(True)`` PyTorch
# substitutes a deterministic ``scatter_add_`` rather than raising, and all
# three variants become bitwise reproducible; nothing here needs a
# ``deterministic`` opt-out. See BACKENDS.md section 5.9 and
# ``tests/test_gradients.py::test_input_gradient_scatter_is_deterministic_on_cpu``.
# ---------------------------------------------------------------------------


def _gather_pair(xg: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
    """``(x[i], x[i + 1])`` along the length axis of ``(B, G, C/G, L)``."""
    return torch.take_along_dim(xg, idx, dim=3), torch.take_along_dim(xg, idx + 1, dim=3)


def _scatter_input_grad(grad_out: Tensor, idx: Tensor, w: Tensor, length: int) -> Tensor:
    """
    ``dL/dx``: accumulate ``g * (1 - w)`` at ``i`` and ``g * w`` at ``i + 1``.

    The two halves are concatenated into a single ``scatter_add_`` rather than
    issued as two, which halves the number of atomic kernels. The index is
    ``expand``ed to the channel axis rather than materialised: ``scatter_add_``
    reads it through a strided iterator, so a stride-0 view costs nothing and
    avoids an int64 tensor the size of the output -- the same reason the forward
    uses ``take_along_dim`` rather than ``gather``.
    """
    batch, groups, per_group, _ = grad_out.shape
    hi = grad_out * w
    lo = grad_out - hi  # == grad_out * (1 - w) up to one rounding
    src = torch.cat((lo, hi), dim=3)
    index = torch.cat((idx, idx + 1), dim=3).expand(batch, groups, per_group, src.shape[3])
    grad_x = torch.zeros(
        (batch, groups, per_group, length), device=grad_out.device, dtype=grad_out.dtype
    )
    return grad_x.scatter_add_(3, index, src)


def _gather_lerp_autograd(xg: Tensor, idx: Tensor, w: Tensor) -> Tensor:
    """Plain autograd: the historical implementation."""
    x0, x1 = _gather_pair(xg, idx)
    return torch.lerp(x0, x1, w)


class _GatherLerpSaveDiff(torch.autograd.Function):
    """Custom Function saving ``x1 - x0`` instead of both ``x0`` and ``x1``."""

    @staticmethod
    def forward(ctx, xg: Tensor, idx: Tensor, w: Tensor) -> Tensor:  # type: ignore[override]
        x0, x1 = _gather_pair(xg, idx)
        # `torch.lerp`, not `x0 + w * (x1 - x0)`: at w == 1 (a tap clamped
        # against the right edge) lerp returns x1 exactly, while the algebraic
        # form returns x0 + (x1 - x0), which is not x1 in floating point. The
        # nn.Conv1d bit-exactness invariant depends on this.
        out = torch.lerp(x0, x1, w)
        # `x1 - x0` is read by the offset gradient and by nothing else, so an
        # inference pass must not pay for it: without this guard the forward is
        # 5-17% slower than the plain-autograd kernel (BACKENDS.md 5.9).
        # `ctx.needs_input_grad` is populated before `forward` runs; it is not
        # affected by `torch.no_grad()`, which is the one case this still
        # over-computes -- inference on a leaf that happens to require grad.
        diff = x1 - x0 if ctx.needs_input_grad[2] else None
        ctx.save_for_backward(idx, w, diff)
        ctx.length = xg.shape[3]
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        idx, w, diff = ctx.saved_tensors
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = _scatter_input_grad(grad_out, idx, w, ctx.length)
        if ctx.needs_input_grad[2]:
            grad_w = (grad_out * diff).sum(dim=2, keepdim=True)
        return grad_x, None, grad_w


class _GatherLerpRecompute(torch.autograd.Function):
    """Custom Function saving nothing but the index and the fraction."""

    @staticmethod
    def forward(ctx, xg: Tensor, idx: Tensor, w: Tensor) -> Tensor:  # type: ignore[override]
        x0, x1 = _gather_pair(xg, idx)
        out = torch.lerp(x0, x1, w)
        ctx.save_for_backward(xg, idx, w)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        xg, idx, w = ctx.saved_tensors
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = _scatter_input_grad(grad_out, idx, w, xg.shape[3])
        if ctx.needs_input_grad[2]:
            x0, x1 = _gather_pair(xg, idx)
            grad_w = (grad_out * (x1 - x0)).sum(dim=2, keepdim=True)
        return grad_x, None, grad_w


_GATHER_LERP_IMPLS = {
    "autograd": _gather_lerp_autograd,
    "save-diff": _GatherLerpSaveDiff.apply,
    "recompute": _GatherLerpRecompute.apply,
}


def _resolve_gather_lerp(name: str | None):
    if name is None or name == "auto":
        name = DEFAULT_GATHER_LERP
    try:
        return _GATHER_LERP_IMPLS[name]
    except KeyError:
        raise ValueError(
            f"unknown gather/lerp implementation {name!r}; "
            f"expected one of {sorted(_GATHER_LERP_IMPLS)}"
        ) from None


def efficient_linterpolate(
    x: Tensor,
    offsets: Tensor,
    kernel_size: int,
    dilation: int,
    stride: int,
    dilated_positions: Tensor | None = None,
    device: torch.device | str | None = None,  # deprecated, ignored; kept for back-compat
    _test: bool = False,
    unconstrained: bool = False,
    _gather_lerp: str | None = None,
) -> Tensor:
    """
    Memory-efficient linear interpolation of the deformed sampling positions.

    Args:
        x (Tensor): Input tensor of shape ``(batch, channels, length)``.
        offsets (Tensor): Offsets of shape
            ``(batch, offset_groups, out_length, kernel_size)``. ``offset_groups``
            must divide ``channels``.
        kernel_size (int): Convolution kernel size.
        dilation (int): Convolution kernel dilation factor.
        stride (int): Convolution kernel stride.
        dilated_positions (Tensor, optional): Precomputed integer kernel tap
            positions. Recomputed from ``kernel_size``/``dilation`` when omitted.
        device: Deprecated and ignored; the device is taken from ``x``.
        unconstrained (bool): When ``False`` (default) each kernel tap is confined
            to its own receptive field. When ``True`` taps may sample anywhere in
            the sequence.
        _gather_lerp: Private. Which gather+lerp implementation to use --
            ``'autograd'``, ``'save-diff'`` or ``'recompute'``. All three produce
            the same forward bit-for-bit and differ only in how the backward is
            obtained; the choice exists so ``benchmarks/backends.py`` can compare
            them. ``None`` means :data:`DEFAULT_GATHER_LERP`.

    Returns:
        Tensor of shape ``(batch, channels, out_length, kernel_size)``.
    """
    if x.device != offsets.device:
        raise ValueError(
            f"x and offsets must be on the same device, got {x.device} and {offsets.device}"
        )
    if x.dim() != 3:
        raise ValueError(f"x must be 3D (batch, channels, length), got shape {tuple(x.shape)}")
    if offsets.dim() != 4:
        raise ValueError(
            "offsets must be 4D (batch, offset_groups, out_length, kernel_size), "
            f"got shape {tuple(offsets.shape)}"
        )

    batch, channels, length = x.shape
    if length < 2:
        raise ValueError(f"input length must be at least 2 for interpolation, got {length}")

    groups, out_length = offsets.shape[1], offsets.shape[-2]
    if offsets.shape[0] != batch:
        raise ValueError(
            f"offsets batch size {offsets.shape[0]} does not match input batch size {batch}"
        )
    if offsets.shape[-1] != kernel_size:
        raise ValueError(
            f"offsets last dimension {offsets.shape[-1]} does not match kernel_size {kernel_size}"
        )
    if channels % groups != 0:
        raise ValueError(
            f"number of offset groups ({groups}) must divide the number of channels ({channels})"
        )

    dilated = _dilated_positions_long(kernel_size, dilation, x.device, dilated_positions)
    max_tap = dilation * (kernel_size - 1)  # compile-time constant, no device reduction

    # ------------------------------------------------------------------
    # Precision decomposition.
    #
    # The sampling position is T = t0 + dilated_position + offset. The first two
    # terms are exact integers and are kept in `long`; only the sub-sample
    # fraction is ever carried in the (possibly fp16, under autocast) offset
    # dtype. Materialising T as a single low-precision float -- as this function
    # used to -- destroys the window positions for sequences longer than ~2k,
    # silently and without producing NaNs.
    # ------------------------------------------------------------------
    t0 = (torch.arange(out_length, device=x.device, dtype=torch.long) * stride).unsqueeze(-1)

    # Position of each tap *relative to its window start*; small magnitude.
    relative = dilated.to(offsets.dtype) + offsets  # (B, G, Lo, K)
    if not unconstrained:
        relative = relative.clamp(0.0, float(max_tap))

    with torch.no_grad():
        relative_floor = torch.floor(relative)
    frac = relative - relative_floor  # in [0, 1); d(frac)/d(offsets) == 1

    with torch.no_grad():
        index_raw = t0 + relative_floor.long()  # exact integer window position
        index = index_raw.clamp(0, length - 2)
        below = index_raw < 0
        above = index_raw > length - 2

    # Clamping T into [0, length - 1] expressed on the split representation:
    # below the range the pair weight is (1, 0), above it is (0, 1). Matches the
    # zero subgradient a clamp on T would have produced.
    zero = torch.zeros((), device=x.device, dtype=frac.dtype)
    one = torch.ones((), device=x.device, dtype=frac.dtype)
    frac = torch.where(below, zero, torch.where(above, one, frac))

    if _test:
        print("x:", tuple(x.shape))
        print("offsets:", tuple(offsets.shape))
        print("t0s:", tuple(t0.shape))
        print("dilated positions:", tuple(dilated.shape))
        print("index:", tuple(index.shape))
        print("frac:", tuple(frac.shape))

    # ------------------------------------------------------------------
    # Gather + lerp.
    #
    # `take_along_dim` broadcasts its non-indexed dimensions, so the index tensor
    # never has to be tiled up to the channel count. Viewing the channel axis as
    # (groups, channels_per_group) lets a single index serve every channel in its
    # offset group, including the 1 < groups < channels case.
    # ------------------------------------------------------------------
    per_group = channels // groups
    xg = x.reshape(batch, groups, per_group, length)
    idx = index.reshape(batch, groups, 1, out_length * kernel_size)
    weight = frac.to(x.dtype).reshape(batch, groups, 1, out_length * kernel_size)

    # x1 is by construction x0's right neighbour, so the two bilinear weights are
    # exactly (1 - frac) and frac -- i.e. a lerp. The tap axis stays flattened
    # through the gather so that the (1, out_length * kernel_size) index and
    # fraction broadcast across the channels of their offset group.
    out = _resolve_gather_lerp(_gather_lerp)(xg, idx, weight)
    return out.reshape(batch, channels, out_length, kernel_size)


def full_seq_linterpolate(
    x: Tensor,
    offsets: Tensor,
    kernel_size: int,
    dilation: int,
    stride: int,
    dilated_positions: Tensor | None = None,
    device: torch.device | str | None = None,
    _test: bool = False,
) -> Tensor:
    """
    Full sequence linear interpolation function for 1D deformable convolution.

    This materialises an ``out_length x kernel_size x length`` weight tensor and
    should only be used for short sequences; prefer
    :func:`efficient_linterpolate`.

    Args:
        x (Tensor): Input tensor of shape ``(batch, channels, length)``.
        offsets (Tensor): Offsets of shape
            ``(batch, offset_groups, out_length, kernel_size)``.
        kernel_size (int): Convolution kernel size.
        dilation (int): Convolution kernel dilation factor.
        stride (int): Convolution kernel stride.
        dilated_positions (Tensor, optional): Precomputed kernel tap positions.
        device: Deprecated and ignored; the device is taken from ``x``.
    """
    device = x.device
    if dilated_positions is None:
        dilated_positions = torch.linspace(
            0, dilation * kernel_size - dilation, kernel_size, device=device
        )  # kernel_size
    else:
        dilated_positions = dilated_positions.to(device)

    max_t0 = (offsets.shape[-2] - 1) * stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2], device=device).unsqueeze(
        -1
    )  # out_length x 1
    dilated_offsets_repeated = dilated_positions + offsets
    T = t0s + dilated_offsets_repeated  # batch x groups x out_length x kernel_size

    if _test:
        print("x:", x.shape)  # batch x in_channels x input_length
        print("offsets:", offsets.shape)  # batch x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:", t0s.shape)  # out_length x 1
        print("dilated positions:", dilated_positions.shape)  # kernel_size
        print("dilated_offsets_repeated:", dilated_offsets_repeated.shape)

    max_U = x.shape[-1] - 1
    U = torch.linspace(0, max_U, max_U + 1, device=device).repeat(1, 1, 1, 1, 1)
    abs_sub = 1 - torch.abs(U - T.unsqueeze(-1))
    _zeros = torch.zeros(abs_sub.shape, device=device)
    G = torch.max(_zeros, abs_sub)  # batch x groups x out_length x kernel_size x length

    if _test:
        print("T:", T.shape)
        print("U:", U.shape)
        print("abs_sub:", abs_sub.shape)
        print("G:", G.shape)

    mx = torch.multiply(G.moveaxis((0, 1), (2, 3)), x)
    x_offset = torch.sum(mx, dim=-1).moveaxis((0, 1), (-2, -1))

    if _test:
        print("mx:", mx.shape)
        print("x_offset:", x_offset.shape)
        print(
            "Desired shape:",
            (x.shape[0], x.shape[1], offsets.shape[-2], kernel_size),
            "(batch_size, in_channels, output_length, kernel_size)",
        )
    return x_offset


def kernel_width_linterpolate(
    x: Tensor,
    offsets: Tensor,
    kernel_size: int,
    dilation: int,
    stride: int,
    dilated_positions: Tensor | None = None,
    device: torch.device | str | None = None,
    _test: bool = False,
    _max_memory: bool = True,
) -> Tensor:
    """
    Receptive-field-width linear interpolation for 1D deformable convolution.

    Args:
        x (Tensor): Input tensor of shape ``(batch, channels, length)``.
        offsets (Tensor): Offsets of shape
            ``(batch, offset_groups, out_length, kernel_size)``.
        kernel_size (int): Convolution kernel size.
        dilation (int): Convolution kernel dilation factor.
        stride (int): Convolution kernel stride.
        dilated_positions (Tensor, optional): Precomputed kernel tap positions.
        device: Deprecated and ignored; the device is taken from ``x``.
        _max_memory (bool): Vectorise over the output length (fast, more memory)
            rather than looping over output positions.
    """
    if x.device != offsets.device:
        raise ValueError(
            f"x and offsets must be on the same device, got {x.device} and {offsets.device}"
        )
    device = x.device
    kernel_rfield = dilation * (kernel_size - 1) + 1
    if dilated_positions is None:
        dilated_positions = torch.linspace(0, kernel_rfield - 1, kernel_size, device=device)
    else:
        dilated_positions = dilated_positions.to(device)

    max_t0 = (offsets.shape[-2] - 1) * stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2], device=device).unsqueeze(-1)
    dilated_offsets_repeated = dilated_positions + offsets

    T = t0s + dilated_offsets_repeated  # batch x groups x out_length x kernel_size
    T = torch.max(T, t0s)
    T = torch.min(T, t0s + float(dilation * (kernel_size - 1)))

    if _test:
        print("x:", x.shape)
        print("offsets:", offsets.shape)
        print("max_t0:", max_t0)
        print("t0s:", t0s.shape)
        print("dilated positions:", dilated_positions.shape)
        print("dilated_offsets_repeated:", dilated_offsets_repeated.shape)
        print("T:", T.shape)

    if _max_memory:
        U = t0s + torch.linspace(0, kernel_rfield - 1, kernel_rfield, device=device).repeat(
            1, 1, 1, 1
        )
        if _test:
            print("U:", U.shape)

        abs_sub = 1 - torch.abs(U.unsqueeze(-1) - T.unsqueeze(-2))
        if _test:
            print("abs_sub:", abs_sub.shape)

        _zeros = torch.zeros(abs_sub.shape, device=device)
        x = x.unfold(dimension=2, size=kernel_rfield, step=stride).unsqueeze(-1)
        if _test:
            print("x unfolded:", x.shape)

        G = torch.max(_zeros, abs_sub)
        if _test:
            print("G:", G.shape)

        mx = torch.multiply(G, x)
        return torch.sum(mx, dim=-2)  # batch x channels x out_length x kernel_size

    outputs = []
    for i in range(t0s.shape[0]):
        t0 = int(t0s[i, 0].item())
        max_U = int(t0 + kernel_rfield - 1)
        U = torch.linspace(t0, max_U, kernel_rfield, device=device)
        abs_sub = 1 - torch.abs(U.repeat(1, 1, T.shape[-1], 1) - T[:, :, i, :].unsqueeze(-1))
        _zeros = torch.zeros(abs_sub.shape, device=device)
        G = torch.max(_zeros, abs_sub)
        mx = torch.multiply(G, x[:, :, t0 : max_U + 1].unsqueeze(-2))
        outputs.append(torch.sum(mx, dim=-1))
    return torch.stack(outputs, dim=2)


if __name__ == "__main__":
    import time

    # Use small values to easily observe effects
    batch_size = 1
    length = 150
    channels = 12
    kernel_size = 3
    dilation = 3
    groups = 12
    stride = 2
    _test = True  # Leave as False unless you want to see all intermediate shapes
    torch.random.manual_seed(1234)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.rand(batch_size, channels, length, requires_grad=True, device=dev)

    num_samples = output_length(length, kernel_size, dilation, stride)
    offsets = -0.5 * torch.ones(batch_size, groups, num_samples, kernel_size, device=dev)

    if dev == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    x_offset = efficient_linterpolate(
        x, offsets, kernel_size, dilation, stride, unconstrained=False, _test=_test
    )
    if dev == "cuda":
        torch.cuda.synchronize()
    stop = time.perf_counter()
    print("Elapsed:", stop - start, "s")

    if _test:
        print(f"Input {x.shape}:", x)
        print(f"Output {x_offset.shape}:", x_offset)
