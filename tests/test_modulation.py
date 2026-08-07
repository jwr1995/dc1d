"""
DCNv2 modulation: `DeformConv1d(..., mask=...)` and `PackedDeformConv1d(modulated=True)`.

The mask weights each sampled position by a scalar, so the properties worth
pinning are the ones that fix *which* scalar multiplies *which* sample: an
all-ones mask must change nothing at all, a constant mask must scale the
pre-bias output by exactly that constant, and a per-group mask must gate exactly
the channels of its own offset group. The last one is the one that catches a
transposed or mis-strided channel mapping, which is the easy bug here.
"""

import pytest
import torch
from torch import nn

from dc1d.nn import DeformConv1d, PackedDeformConv1d

BATCH, CHANNELS, LENGTH, KERNEL = 2, 8, 64, 3


def _model(**kwargs):
    torch.manual_seed(0)
    return DeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="valid", **kwargs)


def _offsets(model, offset_groups=1, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = model.expected_offset_positions(LENGTH)
    return torch.randn(BATCH, offset_groups, n, KERNEL, generator=g) * scale


def _x(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, CHANNELS, LENGTH, generator=g)


def test_mask_of_ones_is_a_no_op():
    """Bit-exact, not merely close: multiplying by 1.0 is exact in floating point."""
    model = _model()
    x, offsets = _x(), _offsets(model)
    assert torch.equal(model(x, offsets), model(x, offsets, torch.ones_like(offsets)))


def test_mask_of_zeros_leaves_only_the_bias():
    model = _model()
    x, offsets = _x(), _offsets(model)
    out = model(x, offsets, torch.zeros_like(offsets))
    expected = model.bias.reshape(1, -1, 1).expand_as(out)
    assert torch.allclose(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("constant", [0.5, 2.0, -1.5])
def test_constant_mask_scales_the_pre_bias_output(constant):
    """A uniform mask must factor straight out of the contraction."""
    model = _model()
    x, offsets = _x(), _offsets(model)
    bias = model.bias.reshape(1, -1, 1)
    plain = model(x, offsets) - bias
    masked = model(x, offsets, torch.full_like(offsets, constant)) - bias
    assert torch.allclose(masked, constant * plain, atol=1e-6)


def test_mask_gates_exactly_the_channels_of_its_own_offset_group():
    """
    Zeroing offset group 1's mask must be identical to zeroing the input
    channels that group 1 owns. This is an independent derivation of the channel
    mapping: it never multiplies a mask by anything, it deletes input instead.
    """
    offset_groups = 2
    model = _model(groups=1)
    x = _x()
    offsets = _offsets(model, offset_groups=offset_groups)

    mask = torch.ones_like(offsets)
    mask[:, 1] = 0.0
    gated = model(x, offsets, mask)

    per_group = CHANNELS // offset_groups
    x_zeroed = x.clone()
    x_zeroed[:, per_group:] = 0.0
    # Interpolating zeros gives zeros, so deleting the channels is equivalent.
    expected = model(x_zeroed, offsets)
    assert torch.allclose(gated, expected, atol=1e-6)


def test_mask_shape_is_validated():
    model = _model()
    x, offsets = _x(), _offsets(model)
    with pytest.raises(ValueError, match="mask shape"):
        model(x, offsets, torch.ones(BATCH, 1, offsets.shape[-2]))


def test_mask_gradient_is_correct_in_float64():
    """gradcheck against the mask, alongside the existing input/offset checks."""
    torch.manual_seed(0)
    model = DeformConv1d(4, 4, KERNEL, padding="valid").double()
    x = torch.randn(1, 4, 16, dtype=torch.float64)
    n = model.expected_offset_positions(16)
    offsets = torch.randn(1, 1, n, KERNEL, dtype=torch.float64) * 0.5
    mask = torch.rand(1, 1, n, KERNEL, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda m: model(x, offsets, m), (mask,), atol=1e-6)


def test_mask_and_offsets_receive_gradients_together():
    model = _model()
    x = _x()
    offsets = _offsets(model).requires_grad_(True)
    mask = torch.rand_like(offsets).requires_grad_(True)
    model(x, offsets, mask).square().sum().backward()
    assert offsets.grad is not None and offsets.grad.abs().sum() > 0
    assert mask.grad is not None and mask.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# PackedDeformConv1d(modulated=True)
# ---------------------------------------------------------------------------


def test_packed_modulated_is_off_by_default():
    model = PackedDeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="same")
    assert model.modulated is False
    assert not hasattr(model, "mask_pconv")


def test_packed_modulated_mask_starts_at_one_half():
    """Zero-init on the mask projection, so every tap starts uninformative."""
    model = PackedDeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="same", modulated=True).eval()
    _, (_, mask) = model(_x(), with_offsets=True)
    assert mask is not None
    assert torch.allclose(mask, torch.full_like(mask, 0.5), atol=0, rtol=0)


def test_packed_modulated_output_is_half_the_unmodulated_one_at_init():
    """Follows from the 0.5 mask, and checks the mask reaches the contraction."""
    torch.manual_seed(0)
    modulated = PackedDeformConv1d(
        CHANNELS, CHANNELS, KERNEL, padding="same", modulated=True
    ).eval()
    plain = PackedDeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="same").eval()
    plain.load_state_dict(
        {k: v for k, v in modulated.state_dict().items() if not k.startswith(("mask_", "mdp_"))}
    )
    x = _x()
    bias = modulated.bias.reshape(1, -1, 1)
    assert torch.allclose(modulated(x) - bias, 0.5 * (plain(x) - bias), atol=1e-6)


def test_packed_modulated_shapes_and_gradients():
    model = PackedDeformConv1d(
        CHANNELS, CHANNELS, KERNEL, padding="same", offset_groups=2, modulated=True
    )
    x = _x().requires_grad_(True)
    out, (offsets, mask) = model(x, with_offsets=True)
    assert out.shape == (BATCH, CHANNELS, LENGTH)
    assert mask.shape == offsets.shape
    assert ((mask > 0) & (mask < 1)).all(), "sigmoid should keep the mask in (0, 1)"
    out.square().sum().backward()
    assert model.mask_pconv.weight.grad is not None
    assert model.mask_pconv.weight.grad.abs().sum() > 0


def test_packed_unmodulated_return_shape_is_unchanged():
    """Back-compat: `with_offsets` must still return a bare tensor when off."""
    model = PackedDeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="same")
    _, offsets = model(_x(), with_offsets=True)
    assert isinstance(offsets, torch.Tensor)


def test_modulated_layer_survives_a_state_dict_round_trip():
    a = PackedDeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="same", modulated=True).eval()
    b = PackedDeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="same", modulated=True).eval()
    b.load_state_dict(a.state_dict())
    x = _x()
    assert torch.equal(a(x), b(x))


def test_deform_conv1d_mask_matches_an_explicit_reference():
    """
    Independent reference: interpolate, weight, contract, written out by hand
    rather than by calling the layer.
    """
    from dc1d.ops import efficient_linterpolate

    model = _model(groups=1)
    x = _x()
    offsets = _offsets(model, offset_groups=2)
    mask = torch.rand_like(offsets)

    sampled = efficient_linterpolate(
        x, offsets, KERNEL, model.dilation, model.stride, unconstrained=False
    )  # (B, C, Lo, K)
    per_group = CHANNELS // offsets.shape[1]
    weighted = torch.stack(
        [sampled[:, i * per_group : (i + 1) * per_group] * mask[:, i : i + 1] for i in range(2)],
        dim=1,
    ).flatten(1, 2)
    expected = nn.functional.conv1d(
        weighted.flatten(-2, -1), model.weight, model.bias, stride=KERNEL
    )
    assert torch.allclose(model(x, offsets, mask), expected, atol=1e-6)
