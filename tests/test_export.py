"""
Export tests: `torch.export` and ONNX.

The load-bearing one is `test_onnx_is_correct_at_lengths_other_than_the_export_length`.
`DeformConv1d` used to gather with `torch.take_along_dim`, which decomposes to a
negative-index wrap `index % self.size(dim)`; from torch 2.10 the ONNX exporter
constant-folds that modulus against the *export-time* length. A model exported
at L = 200 then emitted a literal `Mod(index, 200)`, so every input longer than
200 wrapped around and read the wrong samples. It produced the right output
shape, raised nothing, and was wrong by the magnitude of the signal.

Bisected: clean on torch 2.7-2.9, broken on 2.10-2.13. A regression, which means
an export validated before 2.10 would have started returning wrong numbers on
upgrade with nothing to indicate it.

A shape assertion cannot catch that, and neither can a numerical test at a
single length: the length it is easiest to test at is the length it was exported
at, which is exactly where the bug is invisible. Hence the sweep below, which
deliberately runs *shorter* and *longer* than the export length. `gather` takes
no negative indices, so it emits no `Mod`.
"""

import pytest
import torch
from torch import nn

from dc1d.nn import DeformConv1d, PackedDeformConv1d

BATCH, CHANNELS, KERNEL = 2, 4, 3
EXPORT_LENGTH = 200
# Deliberately spans both sides of EXPORT_LENGTH.
EVAL_LENGTHS = (120, 200, 201, 300, 777, 1600)


class _Wrapped(nn.Module):
    """DeformConv1d with an explicit offsets input, so both axes are dynamic."""

    def __init__(self, **kwargs):
        super().__init__()
        self.layer = DeformConv1d(CHANNELS, CHANNELS, KERNEL, **kwargs)

    def forward(self, x, offsets):
        return self.layer(x, offsets)


def _inputs(model, length, offset_groups=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, CHANNELS, length, generator=g)
    n = model.layer.expected_offset_positions(length)
    offsets = torch.randn(BATCH, offset_groups, n, KERNEL, generator=g)
    return x, offsets


@pytest.mark.parametrize("stride,dilation", [(1, 1), (2, 1), (1, 4), (3, 2)])
def test_torch_export_generalises_across_lengths(stride, dilation):
    """The exported program must not specialise on the traced length."""
    model = _Wrapped(stride=stride, dilation=dilation, padding="valid").eval()
    example = _inputs(model, EXPORT_LENGTH)
    dynamic = {"x": {2: torch.export.Dim.AUTO}, "offsets": {2: torch.export.Dim.AUTO}}
    exported = torch.export.export(model, example, dynamic_shapes=dynamic)

    for length in EVAL_LENGTHS:
        x, offsets = _inputs(model, length, seed=length)
        with torch.no_grad():
            expected = model(x, offsets)
        got = exported.module()(x, offsets)
        assert got.shape == expected.shape, length
        # Same graph, same kernels: this should be exact, not merely close.
        assert torch.equal(got, expected), f"length {length}"


def test_torch_export_accepts_a_named_dim():
    """A named Dim must survive the shape solver, not just Dim.AUTO."""
    model = _Wrapped(padding="valid").eval()
    example = _inputs(model, EXPORT_LENGTH)
    length = torch.export.Dim("length", min=16, max=100_000)
    exported = torch.export.export(
        model, example, dynamic_shapes={"x": {2: length}, "offsets": {2: length - (KERNEL - 1)}}
    )
    x, offsets = _inputs(model, 777, seed=777)
    assert torch.equal(exported.module()(x, offsets), model(x, offsets))


def test_packed_torch_export_generalises():
    model = PackedDeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="same").eval()
    x = torch.randn(BATCH, CHANNELS, EXPORT_LENGTH)
    exported = torch.export.export(
        model, (x,), dynamic_shapes={"input": {2: torch.export.Dim.AUTO}}
    )
    for length in EVAL_LENGTHS:
        xx = torch.randn(BATCH, CHANNELS, length)
        with torch.no_grad():
            expected = model(xx)
        assert torch.equal(exported.module()(xx), expected), f"length {length}"


# ---------------------------------------------------------------------------
# ONNX. Needs the `export` dependency group: `uv sync --group export`.
# ---------------------------------------------------------------------------


def _onnx():
    pytest.importorskip("onnxscript", reason="needs the `export` dependency group")
    onnx = pytest.importorskip("onnx", reason="needs the `export` dependency group")
    ort = pytest.importorskip("onnxruntime", reason="needs the `export` dependency group")
    return onnx, ort


def _export_onnx(model, example, path, dynamic):
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=list(dynamic),
        output_names=["y"],
        dynamic_shapes=dynamic,
        dynamo=True,
        opset_version=18,
        verbose=False,
    )
    return str(path)


@pytest.mark.parametrize("stride,dilation", [(1, 1), (2, 1), (1, 4)])
def test_onnx_is_correct_at_lengths_other_than_the_export_length(tmp_path, stride, dilation):
    """The regression test for the baked export-time length. See module docstring."""
    _, ort = _onnx()
    model = _Wrapped(stride=stride, dilation=dilation, padding="valid").eval()
    example = _inputs(model, EXPORT_LENGTH)
    path = _export_onnx(
        model,
        example,
        tmp_path / "dc1d.onnx",
        {"x": {2: torch.export.Dim.AUTO}, "offsets": {2: torch.export.Dim.AUTO}},
    )
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    for length in EVAL_LENGTHS:
        x, offsets = _inputs(model, length, seed=length)
        with torch.no_grad():
            expected = model(x, offsets)
        (got,) = session.run(None, {"x": x.numpy(), "offsets": offsets.numpy()})
        assert got.shape == tuple(expected.shape), f"length {length}"
        # The failure this guards against is wrong by the magnitude of the
        # signal (~4e+00 against a ~4e-06 baseline), so the tolerance is not
        # doing subtle work.
        err = torch.from_numpy(got).sub(expected).abs().max().item()
        assert err < 1e-4, f"length {length}: max abs error {err:.3e}"


def test_onnx_graph_does_not_bake_the_export_length_into_a_modulus(tmp_path):
    """
    Pin the root cause, not only the symptom.

    A `Mod` by a constant in the gather path is the signature of the
    `take_along_dim` negative-index wrap being folded against the export-time
    length. There is no legitimate reason for this graph to contain one.
    """
    onnx, _ = _onnx()
    model = _Wrapped(padding="valid").eval()
    example = _inputs(model, EXPORT_LENGTH)
    path = _export_onnx(
        model,
        example,
        tmp_path / "dc1d.onnx",
        {"x": {2: torch.export.Dim.AUTO}, "offsets": {2: torch.export.Dim.AUTO}},
    )
    graph = onnx.load(path).graph
    constants = {i.name for i in graph.initializer}
    baked = [n.name for n in graph.node if n.op_type == "Mod" and n.input[1] in constants]
    assert not baked, f"gather path contains a constant-divisor Mod: {baked}"


def test_onnx_graph_uses_no_custom_operators(tmp_path):
    """The whole point of dc1d is that it is plain ATen; the graph should show it."""
    onnx, _ = _onnx()
    model = _Wrapped(padding="valid").eval()
    example = _inputs(model, EXPORT_LENGTH)
    path = _export_onnx(
        model,
        example,
        tmp_path / "dc1d.onnx",
        {"x": {2: torch.export.Dim.AUTO}, "offsets": {2: torch.export.Dim.AUTO}},
    )
    graph = onnx.load(path).graph
    domains = {node.domain for node in graph.node}
    assert domains <= {"", "ai.onnx"}, f"non-standard operator domains: {domains}"


def test_onnx_modulated_packed_layer_is_correct_across_lengths(tmp_path):
    _, ort = _onnx()
    model = PackedDeformConv1d(
        CHANNELS, CHANNELS, KERNEL, padding="same", offset_groups=2, modulated=True
    ).eval()
    x = torch.randn(BATCH, CHANNELS, EXPORT_LENGTH)
    path = _export_onnx(
        model, (x,), tmp_path / "packed.onnx", {"input": {2: torch.export.Dim.AUTO}}
    )
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for length in EVAL_LENGTHS:
        xx = torch.randn(BATCH, CHANNELS, length)
        with torch.no_grad():
            expected = model(xx)
        (got,) = session.run(None, {"input": xx.numpy()})
        assert got.shape == tuple(expected.shape), f"length {length}"
        err = torch.from_numpy(got).sub(expected).abs().max().item()
        assert err < 1e-4, f"length {length}: max abs error {err:.3e}"


@pytest.mark.parametrize("norm", ["gLN", "cLN"])
def test_normalisation_epsilon_survives_export(tmp_path, norm):
    """
    The exporter folds away a small additive constant, so `var + 1e-9` vanished
    from the graph and a zero-variance input divided 0 by 0: NaN from
    onnxruntime where eager PyTorch returns zeros. `PackedDeformConv1d` hits
    this for real, because `modulated=True` zero-initialises the mask
    projection, which makes the tensor reaching gLN exactly constant at
    initialisation. See `dc1d.nn._rms`.
    """
    onnx, ort = _onnx()
    from dc1d.nn import cLN, gLN

    layer = {"gLN": gLN, "cLN": cLN}[norm](CHANNELS)

    class Constant(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = layer

        def forward(self, x):
            return self.norm(x * 0.0)  # variance is exactly zero

    model = Constant().eval()
    x = torch.randn(BATCH, EXPORT_LENGTH, CHANNELS)
    path = _export_onnx(model, (x,), tmp_path / f"{norm}.onnx", {"x": {1: torch.export.Dim.AUTO}})
    (got,) = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
        None, {"x": x.numpy()}
    )
    assert not torch.from_numpy(got).isnan().any(), f"{norm} produced NaN under onnxruntime"
    with torch.no_grad():
        assert torch.allclose(torch.from_numpy(got), model(x), atol=1e-6)


def test_offset_gradients_still_flow_after_export_friendly_gather():
    """The gather rewrite must not cost the offset gradient, which is the point."""
    model = DeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="valid")
    x = torch.randn(BATCH, CHANNELS, 64, requires_grad=True)
    n = model.expected_offset_positions(64)
    offsets = torch.randn(BATCH, 1, n, KERNEL, requires_grad=True)
    model(x, offsets).sum().backward()
    assert offsets.grad is not None and offsets.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_zero_offsets_still_match_nn_conv1d_exactly():
    """
    The repo's central invariant, restated here because the gather rewrite
    touched the indexing path. `tests/test_equivalence.py` is the thorough
    version; this is a tripwire local to the change.
    """
    model = DeformConv1d(CHANNELS, CHANNELS, KERNEL, padding="valid", bias=True)
    reference = nn.Conv1d(CHANNELS, CHANNELS, KERNEL, padding="valid", bias=True)
    reference.load_state_dict({"weight": model.weight, "bias": model.bias})
    x = torch.randn(BATCH, CHANNELS, 128)
    n = model.expected_offset_positions(128)
    offsets = torch.zeros(BATCH, 1, n, KERNEL)
    assert torch.equal(model(x, offsets), reference(x))
