"""
Regression guards for the two bugs that made the package unusable out of the box:

* `dc1d/nn.py` imported `turtle`, an IDE autocomplete artifact, which pulls in
  tkinter -- `import dc1d.nn` raised ImportError on any interpreter built
  without python3-tk (including the system python on the author's machine).
* `extra_repr` was copy-pasted from `_ConvNd`, which stores padding/dilation as
  tuples. This class stores raw ints, so `print(model)` raised TypeError.
"""

import importlib
import sys

import torch

import dc1d
from dc1d.nn import DeformConv1d, PackedDeformConv1d


def test_package_imports():
    for name in ("dc1d", "dc1d.nn", "dc1d.ops"):
        importlib.import_module(name)
    assert "turtle" not in sys.modules
    assert "tkinter" not in sys.modules


def test_no_torchvision_dependency():
    assert "torchvision" not in sys.modules


def test_version_is_exposed():
    assert isinstance(dc1d.__version__, str)
    assert dc1d.__version__.count(".") >= 1


def test_top_level_reexports():
    assert dc1d.DeformConv1d is DeformConv1d
    assert dc1d.PackedDeformConv1d is PackedDeformConv1d
    assert callable(dc1d.efficient_linterpolate)
    assert callable(dc1d.full_seq_linterpolate)
    assert callable(dc1d.kernel_width_linterpolate)


def test_repr_does_not_raise():
    for model in (
        DeformConv1d(8, 8, 3),
        DeformConv1d(8, 16, 5, stride=2, dilation=3, groups=4, bias=False, padding=2),
        PackedDeformConv1d(8, 8, 3, padding="same", unconstrained=True),
    ):
        text = repr(model)
        assert type(model).__name__ in text
        assert "kernel_size" in text


def test_unconstrained_is_always_a_bool_attribute():
    assert DeformConv1d(8, 8, 3).unconstrained is False
    assert DeformConv1d(8, 8, 3, unconstrained=True).unconstrained is True
    assert DeformConv1d(8, 8, 3, unconstrained=False).unconstrained is False


def test_dilated_positions_is_a_buffer():
    model = DeformConv1d(8, 8, 3, dilation=4)
    names = dict(model.named_buffers())
    assert "dilated_positions" in names
    assert torch.equal(names["dilated_positions"], torch.tensor([0.0, 4.0, 8.0]))
    # non-persistent: it is derived from the config, not learned
    assert "dilated_positions" not in model.state_dict()


def test_forward_does_not_mutate_module_state():
    """forward() used to reassign self.device, which is a torch.compile graph break."""
    model = DeformConv1d(8, 8, 3, padding="valid")
    before = dict(model.__dict__)
    x = torch.randn(2, 8, 20)
    n = model.expected_offset_positions(20)
    model(x, torch.zeros(2, 1, n, 3))
    after = dict(model.__dict__)
    assert before.keys() == after.keys()
    assert not hasattr(model, "device")


def test_state_dict_roundtrip():
    a = DeformConv1d(8, 8, 3, dilation=2, padding="valid")
    b = DeformConv1d(8, 8, 3, dilation=2, padding="valid")
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 8, 20)
    n = a.expected_offset_positions(20)
    offsets = torch.full((2, 1, n, 3), 0.3)
    assert torch.equal(a(x, offsets), b(x, offsets))


def test_readme_example_runs():
    from torch import nn

    batch_size, in_channels, out_channels = 4, 32, 32
    kernel_size, stride, dilation, length = 16, 1, 3, 128

    model = DeformConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding="valid",
        dilation=dilation,
        groups=1,
        bias=True,
    )
    x = torch.rand(batch_size, in_channels, length, requires_grad=True)
    n_offsets = model.expected_offset_positions(length)
    assert n_offsets == length - dilation * (kernel_size - 1)
    offsets = nn.Parameter(torch.ones(batch_size, 1, n_offsets, kernel_size))
    y = model(x, offsets)
    assert y.shape == (batch_size, out_channels, n_offsets)
