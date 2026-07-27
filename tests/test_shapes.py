"""
Shape-contract tests.

`PackedDeformConv1d` used to build its offset convolution with a hardcoded
stride=1 and no dilation, so with stride>1 or dilation>1 it emitted the wrong
number of offset positions and the index clamp inside the interpolation kernel
silently absorbed it into a wrong-length output.
"""

import pytest
import torch

from dc1d.nn import DeformConv1d, PackedDeformConv1d
from dc1d.ops import output_length

BATCH, CHANNELS, LENGTH = 2, 8, 40


def expected_length(length, kernel_size, dilation, stride, padding):
    if padding == "same":
        return length
    return output_length(length, kernel_size, dilation, stride)


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("groups", [1, 2, 8])
@pytest.mark.parametrize("offset_groups", [1, 4, 8])
def test_deform_conv1d_output_shape(stride, dilation, padding, groups, offset_groups):
    if padding == "same" and stride != 1:
        pytest.skip("padding='same' is not defined for strided convolutions")

    kernel_size = 3
    model = DeformConv1d(
        CHANNELS,
        CHANNELS,
        kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
    )
    x = torch.randn(BATCH, CHANNELS, LENGTH)
    n_offsets = model.expected_offset_positions(LENGTH)
    offsets = torch.zeros(BATCH, offset_groups, n_offsets, kernel_size)

    y = model(x, offsets)
    want = expected_length(LENGTH, kernel_size, dilation, stride, padding)
    assert y.shape == (BATCH, CHANNELS, want)


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("offset_groups", [1, 4, 8])
def test_packed_deform_conv1d_output_shape(stride, dilation, padding, offset_groups):
    if padding == "same" and stride != 1:
        pytest.skip("padding='same' is not defined for strided convolutions")

    kernel_size = 3
    model = PackedDeformConv1d(
        CHANNELS,
        CHANNELS,
        kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=CHANNELS,
        offset_groups=offset_groups,
    )
    x = torch.randn(BATCH, CHANNELS, LENGTH)
    y, offsets = model(x, with_offsets=True)

    want = expected_length(LENGTH, kernel_size, dilation, stride, padding)
    assert y.shape == (BATCH, CHANNELS, want)
    assert offsets.shape == (
        BATCH,
        offset_groups,
        model.expected_offset_positions(LENGTH),
        kernel_size,
    )


def test_wrong_offset_count_raises():
    """Silent corruption must become a loud failure."""
    model = DeformConv1d(CHANNELS, CHANNELS, 3, stride=2, padding="valid")
    x = torch.randn(BATCH, CHANNELS, LENGTH)
    bad = torch.zeros(BATCH, 1, LENGTH, 3)  # what a stride-1 offset conv would emit
    with pytest.raises(ValueError, match="offsets has"):
        model(x, bad)


def test_bad_offset_group_count_raises():
    model = DeformConv1d(CHANNELS, CHANNELS, 3, padding="valid")
    x = torch.randn(BATCH, CHANNELS, LENGTH)
    n = model.expected_offset_positions(LENGTH)
    with pytest.raises(ValueError, match="must divide"):
        model(x, torch.zeros(BATCH, 3, n, 3))


def test_packed_rejects_non_divisor_offset_groups():
    with pytest.raises(ValueError, match="divisor"):
        PackedDeformConv1d(8, 8, 3, offset_groups=3)


def test_short_input_raises():
    model = DeformConv1d(CHANNELS, CHANNELS, 3, padding="valid")
    with pytest.raises(ValueError):
        model(torch.randn(BATCH, CHANNELS, 1), torch.zeros(BATCH, 1, 1, 3))


def test_mask_is_rejected():
    model = DeformConv1d(CHANNELS, CHANNELS, 3, padding="valid")
    x = torch.randn(BATCH, CHANNELS, LENGTH)
    n = model.expected_offset_positions(LENGTH)
    with pytest.raises(NotImplementedError):
        model(x, torch.zeros(BATCH, 1, n, 3), mask=torch.zeros(1))
