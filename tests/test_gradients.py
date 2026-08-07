"""
Gradient tests. The gradient with respect to the *offsets* is the entire point of
a deformable convolution and had never been verified.

`gradcheck` needs the function to be differentiable at the evaluation point. The
interpolation is piecewise linear with kinks at integer sampling positions, so
offsets are chosen to keep every position strictly between two samples.
"""

import pytest
import torch

from dc1d.nn import DeformConv1d, PackedDeformConv1d
from dc1d.ops import efficient_linterpolate, output_length

BATCH, CHANNELS, LENGTH, KERNEL = 2, 4, 16, 3


def _offsets(shape, generator):
    """Offsets in [0.25, 0.75] -- never integral, so never on a kink."""
    return 0.25 + 0.5 * torch.rand(shape, generator=generator, dtype=torch.float64)


@pytest.mark.parametrize("unconstrained", [False, True])
@pytest.mark.parametrize("offset_groups", [1, 2, 4])
def test_gradcheck_interpolation(unconstrained, offset_groups):
    g = torch.Generator().manual_seed(1234)
    n_offsets = output_length(LENGTH, KERNEL)

    x = torch.randn(BATCH, CHANNELS, LENGTH, dtype=torch.float64, generator=g)
    offsets = _offsets((BATCH, offset_groups, n_offsets, KERNEL), g)
    x.requires_grad_(True)
    offsets.requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda a, b: efficient_linterpolate(
            a, b, KERNEL, dilation=1, stride=1, unconstrained=unconstrained
        ),
        (x, offsets),
        eps=1e-6,
        atol=1e-8,
        rtol=1e-5,
    )


@pytest.mark.parametrize("unconstrained", [False, True])
@pytest.mark.parametrize("stride,dilation", [(1, 1), (2, 1), (1, 2)])
def test_gradcheck_module(unconstrained, stride, dilation):
    g = torch.Generator().manual_seed(4321)

    model = DeformConv1d(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        kernel_size=KERNEL,
        stride=stride,
        dilation=dilation,
        padding="valid",
        bias=True,
        unconstrained=unconstrained,
    ).double()

    n_offsets = output_length(LENGTH, KERNEL, dilation, stride)
    x = torch.randn(BATCH, CHANNELS, LENGTH, dtype=torch.float64, generator=g)
    offsets = _offsets((BATCH, 1, n_offsets, KERNEL), g)
    x.requires_grad_(True)
    offsets.requires_grad_(True)

    assert torch.autograd.gradcheck(model, (x, offsets), eps=1e-6, atol=1e-8, rtol=1e-5)


def test_offset_gradient_is_nonzero():
    """A zero offset gradient would make the layer non-deformable in practice."""
    torch.manual_seed(0)
    n_offsets = output_length(LENGTH, KERNEL)
    model = DeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="valid")
    x = torch.randn(BATCH, CHANNELS, LENGTH)
    offsets = torch.full((BATCH, 1, n_offsets, KERNEL), 0.5, requires_grad=True)

    model(x, offsets).square().mean().backward()
    assert offsets.grad is not None
    assert offsets.grad.abs().sum() > 0


def test_packed_offsets_receive_gradient():
    torch.manual_seed(0)
    model = PackedDeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="valid", offset_groups=2)
    x = torch.randn(BATCH, CHANNELS, LENGTH)
    y, offsets = model(x, with_offsets=True)
    y.square().mean().backward()

    assert offsets.requires_grad
    assert model.offset_pconv.weight.grad is not None
    assert model.offset_pconv.weight.grad.abs().sum() > 0
    assert model.offset_dconv.weight.grad is not None
    assert model.offset_dconv.weight.grad.abs().sum() > 0


def test_index_tensor_carries_no_gradient():
    """The floor()/clamp() index path must stay detached (`idx` is an integer
    index and carries no gradient; only the fraction `w` does)."""
    torch.manual_seed(0)
    n_offsets = output_length(LENGTH, KERNEL)
    x = torch.randn(BATCH, CHANNELS, LENGTH, requires_grad=True)
    offsets = torch.full((BATCH, 1, n_offsets, KERNEL), 0.5, requires_grad=True)
    out = efficient_linterpolate(x, offsets, KERNEL, 1, 1, unconstrained=True)

    # d(out)/d(offsets) is exactly x1 - x0 for every element -- the integer index
    # is detached, so nothing else contributes.
    grad_out = torch.ones_like(out)
    (grad_offsets,) = torch.autograd.grad(out, offsets, grad_out)

    diff = x.detach()[:, :, 1:] - x.detach()[:, :, :-1]  # x[j+1] - x[j]
    want = diff.unfold(2, KERNEL, 1).sum(dim=1, keepdim=True)
    # Restrict to rows whose whole window is strictly inside the sequence; the
    # final row saturates against the right edge and correctly has zero gradient.
    rows = want.shape[2]
    assert torch.allclose(grad_offsets[:, :, :rows], want, atol=1e-5)
    assert torch.equal(grad_offsets[:, :, -1, -1], torch.zeros(BATCH, 1))


@pytest.mark.parametrize("impl", ["autograd", "save-diff", "recompute"])
@pytest.mark.parametrize("unconstrained", [False, True])
@pytest.mark.parametrize("offset_groups", [1, 2, 4])
def test_gradcheck_gather_lerp_variants(impl, unconstrained, offset_groups):
    """
    `gradcheck` against a hand-written backward is the only thing standing
    between a custom `autograd.Function` and a silently wrong offset gradient.
    """
    g = torch.Generator().manual_seed(1234)
    n_offsets = output_length(LENGTH, KERNEL)

    x = torch.randn(BATCH, CHANNELS, LENGTH, dtype=torch.float64, generator=g)
    offsets = _offsets((BATCH, offset_groups, n_offsets, KERNEL), g)
    x.requires_grad_(True)
    offsets.requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda a, b: efficient_linterpolate(
            a, b, KERNEL, dilation=1, stride=1, unconstrained=unconstrained, gather_lerp=impl
        ),
        (x, offsets),
        eps=1e-6,
        atol=1e-8,
        rtol=1e-5,
    )


@pytest.mark.parametrize("impl", ["save-diff", "recompute"])
def test_custom_backward_matches_autograd(impl):
    """
    Both gradients, at a configuration with clamped taps at both edges and an
    offset group shared by several channels -- i.e. everywhere the hand-written
    backward could disagree with the one autograd derives.
    """
    torch.manual_seed(0)
    batch, channels, length, kernel_size, offset_groups = 2, 8, 24, 3, 2
    n_offsets = output_length(length, kernel_size)
    x = torch.randn(batch, channels, length, dtype=torch.float64, requires_grad=True)
    offsets = (
        torch.randn(batch, offset_groups, n_offsets, kernel_size, dtype=torch.float64) * 4
    ).requires_grad_(True)
    grad_out = torch.randn(batch, channels, n_offsets, kernel_size, dtype=torch.float64)

    want = torch.autograd.grad(
        efficient_linterpolate(x, offsets, kernel_size, 1, 1, unconstrained=True),
        [x, offsets],
        grad_out,
    )
    got = torch.autograd.grad(
        efficient_linterpolate(x, offsets, kernel_size, 1, 1, unconstrained=True, gather_lerp=impl),
        [x, offsets],
        grad_out,
    )
    for a, b in zip(want, got, strict=True):
        assert torch.allclose(a, b, atol=1e-12, rtol=0)


@pytest.mark.parametrize("impl", ["autograd", "save-diff", "recompute"])
def test_input_gradient_scatter_is_deterministic_on_cpu(impl):
    """
    The input gradient is a scatter-add. On CPU it is deterministic for every
    variant; on CUDA it uses atomics and is not, which is a documented property
    shared with the default kernel -- see `dc1d/ops.py` and BACKENDS.md 5.9.
    """
    torch.manual_seed(0)
    n_offsets = output_length(LENGTH, KERNEL)
    x = torch.randn(BATCH, CHANNELS, LENGTH, dtype=torch.float64, requires_grad=True)
    offsets = torch.randn(BATCH, 1, n_offsets, KERNEL, dtype=torch.float64).requires_grad_(True)
    grad_out = torch.randn(BATCH, CHANNELS, n_offsets, KERNEL, dtype=torch.float64)

    def once():
        return torch.autograd.grad(
            efficient_linterpolate(x, offsets, KERNEL, 1, 1, unconstrained=True, gather_lerp=impl),
            [x, offsets],
            grad_out,
        )

    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        a, b = once(), once()
    finally:
        torch.use_deterministic_algorithms(was)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def _interp(impl):
    n_offsets = output_length(LENGTH, KERNEL)

    def fn(a, b):
        return efficient_linterpolate(a, b, KERNEL, 1, 1, unconstrained=True, gather_lerp=impl)

    return fn, n_offsets


@pytest.mark.parametrize("impl", ["autograd", "recompute"])
def test_gather_lerp_variant_supports_vmap(impl):
    """
    A custom `autograd.Function` is opaque to functorch unless it declares
    `setup_context` and `generate_vmap_rule`. The pure-ATen kernel needs
    neither, so losing `vmap` would be a real capability regression.
    """
    torch.manual_seed(0)
    fn, n_offsets = _interp(impl)
    xs = torch.randn(3, BATCH, CHANNELS, LENGTH, dtype=torch.float64)
    offs = torch.randn(3, BATCH, 1, n_offsets, KERNEL, dtype=torch.float64)

    got = torch.vmap(fn)(xs, offs)
    want = torch.stack([fn(xs[i], offs[i]) for i in range(3)])
    assert torch.equal(got, want)


@pytest.mark.parametrize("impl", ["autograd", "recompute"])
def test_gather_lerp_variant_supports_double_backward(impl):
    """
    `create_graph=True` (gradient penalties, Hessian-vector products) needs the
    backward itself to be differentiable. `recompute` re-derives `x1 - x0` from
    the saved *input*, so the second-order term through `x` survives.
    """
    torch.manual_seed(0)
    fn, n_offsets = _interp(impl)
    x = torch.randn(BATCH, CHANNELS, LENGTH, dtype=torch.float64, requires_grad=True)
    offsets = _offsets(
        (BATCH, 1, n_offsets, KERNEL), torch.Generator().manual_seed(7)
    ).requires_grad_(True)

    grad_x, grad_off = torch.autograd.grad(fn(x, offsets).sum(), [x, offsets], create_graph=True)
    (second,) = torch.autograd.grad(grad_off.sum() + grad_x.sum(), [x])
    # d(sum_i sum_c (x[i+1] - x[i]))/dx is +-1 per element, never identically 0.
    assert second.abs().sum() > 0


def test_save_diff_refuses_double_backward():
    """
    `save-diff` stores `x1 - x0` as a constant, so the second-order term through
    the input would silently be zero. It must raise rather than answer wrongly.
    """
    torch.manual_seed(0)
    fn, n_offsets = _interp("save-diff")
    x = torch.randn(BATCH, CHANNELS, LENGTH, dtype=torch.float64, requires_grad=True)
    offsets = torch.randn(BATCH, 1, n_offsets, KERNEL, dtype=torch.float64).requires_grad_(True)
    with pytest.raises(RuntimeError, match="does not support double backward"):
        torch.autograd.grad(fn(x, offsets).sum(), [x, offsets], create_graph=True)
