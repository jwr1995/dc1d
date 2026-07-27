"""
Equivalence tests: with zero offsets a deformable convolution *is* an ordinary
convolution, so `DeformConv1d` must reproduce `nn.Conv1d` bit-for-bit.

This is the test whose absence let the stride/dilation and offset-group bugs
survive for three years -- it pins down indexing, dilation, stride, groups and
the output contraction all at once.
"""

import pytest
import torch
import torch.nn.functional as F

from dc1d.nn import DeformConv1d
from dc1d.ops import (
    efficient_linterpolate,
    full_seq_linterpolate,
    kernel_width_linterpolate,
    output_length,
)


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 4])
@pytest.mark.parametrize("groups", [1, 2, 8])
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_zero_offset_matches_conv1d(stride, dilation, groups, kernel_size):
    torch.manual_seed(0)
    batch, channels, length = 2, 8, 40

    model = DeformConv1d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        padding="valid",
        bias=True,
    )

    x = torch.randn(batch, channels, length)
    n_offsets = output_length(length, kernel_size, dilation, stride)
    offsets = torch.zeros(batch, 1, n_offsets, kernel_size)

    got = model(x, offsets)
    want = F.conv1d(x, model.weight, model.bias, stride=stride, dilation=dilation, groups=groups)

    assert got.shape == want.shape
    if kernel_size == 1 and stride > 1:
        # A strided 1x1 grouped conv and a dense stride-1 grouped conv are
        # dispatched to different oneDNN/cuDNN kernels, so the accumulation order
        # differs by ~1e-7. The interpolation itself is still bit-exact -- see
        # test_zero_offset_gather_is_exact -- so only the final contraction is
        # compared with a tolerance here.
        torch.testing.assert_close(got, want, atol=1e-6, rtol=0)
    else:
        # Bit-for-bit: with zero offsets in constrained mode every sampling
        # position lands on an exact integer, so the interpolation weights are
        # exactly [1, 0].
        assert torch.equal(got, want), f"max abs diff {(got - want).abs().max().item()}"


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 4])
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_zero_offset_gather_is_exact(stride, dilation, kernel_size):
    """The sampled windows themselves must be bit-identical to a plain unfold."""
    torch.manual_seed(0)
    batch, channels, length = 2, 8, 40

    x = torch.randn(batch, channels, length)
    n_offsets = output_length(length, kernel_size, dilation, stride)
    offsets = torch.zeros(batch, 1, n_offsets, kernel_size)

    got = efficient_linterpolate(x, offsets, kernel_size, dilation, stride)
    want = x.unfold(2, dilation * (kernel_size - 1) + 1, stride)[..., ::dilation]

    assert got.shape == want.shape
    assert torch.equal(got, want)


@pytest.mark.parametrize("offset_groups", [1, 2, 4, 8])
def test_zero_offset_matches_conv1d_over_offset_groups(offset_groups):
    """offset_groups strictly between 1 and channels used to crash (U.repeat tiled)."""
    torch.manual_seed(0)
    batch, channels, length, kernel_size = 2, 8, 32, 3

    x = torch.randn(batch, channels, length)
    n_offsets = output_length(length, kernel_size)
    offsets = torch.zeros(batch, offset_groups, n_offsets, kernel_size)

    got = efficient_linterpolate(x, offsets, kernel_size, dilation=1, stride=1)
    want = x.unfold(2, kernel_size, 1)

    assert got.shape == (batch, channels, n_offsets, kernel_size)
    assert torch.equal(got, want)


@pytest.mark.parametrize("offset_groups", [1, 2, 4, 8])
def test_offset_groups_are_applied_per_group(offset_groups):
    """Each offset group must drive exactly channels // offset_groups channels."""
    torch.manual_seed(0)
    batch, channels, length, kernel_size = 1, 8, 32, 3
    per_group = channels // offset_groups

    x = torch.randn(batch, channels, length)
    n_offsets = output_length(length, kernel_size)

    # Group g gets a constant integer shift of g.
    offsets = torch.zeros(batch, offset_groups, n_offsets, kernel_size)
    for g in range(offset_groups):
        offsets[:, g] = float(g)

    got = efficient_linterpolate(x, offsets, kernel_size, dilation=1, stride=1, unconstrained=True)

    for g in range(offset_groups):
        lo, hi = g * per_group, (g + 1) * per_group
        # rows that are fully in-bounds after shifting by g
        n_valid = n_offsets - g
        want = x[:, lo:hi].unfold(2, kernel_size, 1)[:, :, g : g + n_valid]
        assert torch.equal(got[:, lo:hi, :n_valid], want), f"group {g} mismatch"


@pytest.mark.parametrize("shift", [-3, -1, 1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
def test_integer_shift_equivalence(shift, dilation):
    """Integer offsets must reduce to an exact shift of the zero-offset windows."""
    torch.manual_seed(0)
    batch, channels, length, kernel_size = 2, 4, 40, 3

    x = torch.randn(batch, channels, length)
    n_offsets = output_length(length, kernel_size, dilation)

    zero = torch.zeros(batch, 1, n_offsets, kernel_size)
    shifted = torch.full((batch, 1, n_offsets, kernel_size), float(shift))

    base = efficient_linterpolate(x, zero, kernel_size, dilation, 1, unconstrained=True)
    got = efficient_linterpolate(x, shifted, kernel_size, dilation, 1, unconstrained=True)

    lo = max(0, -shift)
    hi = min(n_offsets, n_offsets - shift)
    assert hi > lo
    assert torch.equal(got[:, :, lo:hi], base[:, :, lo + shift : hi + shift])


def test_interpolation_weights_sum_to_one_at_right_edge():
    """
    Unconstrained mode used to clamp T to `length` rather than `length - 1`, so
    both bilinear weights fell to zero at the right edge and the output was
    silently attenuated (to exactly zero at T == length).
    """
    length = 10
    x = torch.ones(1, 1, length)
    # push every tap well past the right edge
    offsets = torch.full((1, 1, length - 2, 3), 50.0)

    got = efficient_linterpolate(x, offsets, 3, 1, 1, unconstrained=True)
    assert torch.allclose(got, torch.ones_like(got)), (
        f"weights do not sum to 1 at the right edge; got {got.flatten()[:5].tolist()}"
    )

    # And it should sample the true last element, not zero.
    ramp = torch.arange(float(length)).reshape(1, 1, length)
    got = efficient_linterpolate(ramp, offsets, 3, 1, 1, unconstrained=True)
    assert torch.equal(got, torch.full_like(got, float(length - 1)))


def test_left_edge_clamp():
    length = 10
    ramp = torch.arange(float(length)).reshape(1, 1, length)
    offsets = torch.full((1, 1, length - 2, 3), -50.0)
    got = efficient_linterpolate(ramp, offsets, 3, 1, 1, unconstrained=True)
    assert torch.equal(got, torch.zeros_like(got))


def test_half_offsets_on_long_sequence_are_exact():
    """
    Regression for the fp16 position bug: sampling positions used to be built by
    `torch.linspace(..., dtype=offsets.dtype)`, which in fp16 has a spacing of 8
    at magnitude 16000. Positions beyond ~2048 were garbage, silently.
    """
    torch.manual_seed(0)
    length, kernel_size = 16000, 3
    x = torch.randn(1, 1, length)
    n_offsets = output_length(length, kernel_size)

    got_half = efficient_linterpolate(
        x, torch.zeros(1, 1, n_offsets, kernel_size, dtype=torch.float16), kernel_size, 1, 1
    )
    want = x.unfold(2, kernel_size, 1)
    assert torch.equal(got_half, want)


def test_fractional_offset_is_a_lerp():
    """A half-sample offset must give the exact midpoint of two samples."""
    ramp = torch.arange(20.0).reshape(1, 1, 20)
    n_offsets = output_length(20, 3)
    offsets = torch.full((1, 1, n_offsets, 3), 0.5)
    got = efficient_linterpolate(ramp, offsets, 3, 1, 1, unconstrained=True)
    # position i + k + 0.5 -> value i + k + 0.5
    idx = torch.arange(n_offsets).unsqueeze(-1) + torch.arange(3).unsqueeze(0)
    want = idx.float().unsqueeze(0).unsqueeze(0) + 0.5
    want = want.clamp(max=19.0)
    assert torch.allclose(got, want)


@pytest.mark.parametrize(
    "kernel,kwargs",
    [
        (kernel_width_linterpolate, {}),
        (kernel_width_linterpolate, {"_max_memory": False}),
        (full_seq_linterpolate, {}),
    ],
    ids=["kernel_width", "kernel_width_loop", "full_seq"],
)
def test_legacy_kernels_agree_and_are_differentiable(kernel, kwargs):
    """
    The two legacy interpolation kernels are still exported, so they must at
    least agree with a plain unfold and participate in autograd. The looping
    branch used to write in place into a preallocated tensor.
    """
    torch.manual_seed(0)
    dilation, kernel_size = 2, 3
    x = torch.randn(2, 4, 20, requires_grad=True)
    n_offsets = output_length(20, kernel_size, dilation)
    offsets = torch.zeros(2, 1, n_offsets, kernel_size, requires_grad=True)

    got = kernel(x, offsets, kernel_size, dilation, 1, **kwargs)
    want = x.detach().unfold(2, dilation * (kernel_size - 1) + 1, 1)[..., ::dilation]
    assert torch.allclose(got, want)

    grad_x, grad_offsets = torch.autograd.grad(got.sum(), [x, offsets])
    assert grad_x is not None and grad_offsets is not None
