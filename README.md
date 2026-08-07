# dc1d (DeformConv1d)

A 1D implementation of a deformable convolutional layer implemented in pure Python in PyTorch. The code style is designed to imitate similar classes in PyTorch such as `torch.nn.Conv1d` and `torchvision.ops.DeformConv2d`.

**See it working:** [`docs/demo.ipynb`](docs/demo.ipynb) is a short, plotted walkthrough of what the layer does and why it is correct. It runs in under ten seconds on a laptop CPU and is committed with its outputs, so it renders without executing anything.

The motivation for creating this toolkit is that (as of 19/10/2022) there is no native 1D implementation of deformable convolution in the PyTorch library, and no alternative library which is simple to install (requiring only a basic PyTorch installation, with no additional compilation of C++ or CUDA libraries). The implementation here is written entirely in Python and makes use of `torch.autograd` for backpropagation.

## Requirements

* Python >= 3.10
* PyTorch >= 2.7 (installed automatically as a dependency)

`torchvision` is **not** required. Earlier releases imported private torchvision
symbols at module scope for a dead code path; that path has been removed.

Install PyTorch for your platform/CUDA version first if you want a specific
build: https://pytorch.org/get-started/locally/.

## Installation

*N.B. for the version we released for our paper, stick to `dc1d==0.0.6` or lower*

[`CHANGELOG.md`](CHANGELOG.md) records what changed since, including why a
0.0.x `PackedDeformConv1d` checkpoint will load without complaint and then
behave differently.

```
pip install dc1d
```

Or from source:

```
git clone https://github.com/jwr1995/dc1d.git
cd dc1d
pip install .
```

### Development

The project is managed with [uv](https://docs.astral.sh/uv/):

```
uv sync            # creates .venv with CPU-only torch and the dev tools
uv run pytest      # run the test suite
uv run ruff check .
uv run ruff format --check .
```

## Usage

### `DeformConv1d`

`DeformConv1d` is the deformable convolution layer designed to imitate `torch.nn.Conv1d`.

Note: `DeformConv1d` does **not** compute the offset values used in its `forward(...)` call. These must be computed outside the layer. Use `PackedDeformConv1d` if you want the offsets computed for you.

```python
import torch
from torch import nn

from dc1d.nn import DeformConv1d

# Hyperparameters
batch_size = 16
in_channels = 512
out_channels = 512
kernel_size = 16
stride = 1
padding = "valid"
dilation = 3
groups = 1
length = 128

# Construct layer
model = DeformConv1d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    groups=groups,
    bias=True,
)

x = torch.rand(batch_size, in_channels, length, requires_grad=True)

# Number of offset positions required. This is the same closed form nn.Conv1d
# uses; forward() raises if the offset tensor disagrees with it.
output_length = model.expected_offset_positions(length)
offsets = nn.Parameter(torch.ones(batch_size, 1, output_length, kernel_size))

y = model(x, offsets)  # [batch_size, out_channels, output_length]
```

The offset tensor has shape `[batch_size, offset_groups, output_length, kernel_size]`,
where `offset_groups` may be `1` or any divisor of `in_channels`.

By default each kernel tap is constrained to its own receptive field. Pass
`unconstrained=True` to let taps sample anywhere in the sequence.

### `PackedDeformConv1d`

`PackedDeformConv1d` computes the offsets internally using a depthwise-separable
convolutional block, as detailed in the paper below.

```python
import torch
from dc1d.nn import PackedDeformConv1d

model = PackedDeformConv1d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    dilation=4,
    groups=64,
    offset_groups=64,
    padding="same",
)
y = model(torch.rand(4, 64, 256))
# y, offsets = model(x, with_offsets=True)  # if you want the offsets too
```

### Modulation (DCNv2)

Both layers accept the modulation of [Zhu et al. 2019](https://arxiv.org/abs/1811.11168),
which weights each sampled position by a learned scalar as well as moving it.
Pass `mask` alongside `offsets`, with the same shape:

```python
from dc1d.nn import DeformConv1d

model = DeformConv1d(in_channels=64, out_channels=64, kernel_size=3, padding="same")
y = model(x, offsets, mask=mask.sigmoid())
```

The mask is applied **as given**, matching `torchvision.ops.deform_conv2d`: if
you want the `[0, 1]` modulation of the paper, apply the sigmoid yourself. An
all-ones mask is bit-for-bit identical to passing no mask at all.

`PackedDeformConv1d(..., modulated=True)` predicts the mask for you, from a
second pointwise branch off the same depthwise trunk that produces the offsets,
ending in a sigmoid. The final projection is zero-initialised, so every tap
starts at exactly 0.5 and the layer has to learn to gate; note this halves the
output scale at initialisation relative to `modulated=False`. With
`with_offsets=True` the layer then returns `(y, (offsets, mask))` rather than
`(y, offsets)`. The default is `modulated=False`, which is unchanged DCNv1 and
adds no parameters.

### ONNX export

Both layers export to ONNX through the dynamo exporter with no custom operators,
and the exported graph is length-agnostic:

```python
torch.onnx.export(
    model,
    (x,),
    "model.onnx",
    dynamic_shapes={"input": {2: torch.export.Dim.AUTO}},
    dynamo=True,
)
```

Export at one length and run at any other. `tests/test_export.py` checks this
against `onnxruntime` from 120 to 1600 samples for a model exported at 200, and
requires the `export` dependency group (`uv sync --group export`).

### Going faster, and using less memory

Nothing below changes what the layer returns — the forward stays bit-for-bit
identical, including the invariant that zero offsets reproduce `nn.Conv1d`
exactly.

**`torch.compile` first.** The layer traces to a single graph with no breaks, so
`torch.compile(model)` works with `fullgraph=True` and is worth **2.3–9.1×** on
the forward and **1.2–2.1×** on forward+backward on an RTX 3090. Note that every
distinct sequence length is a separate compilation (~0.3 s), which matters for
variable-length audio.

**Then, if you use depthwise offsets, trade a little backward latency for a lot
of memory.** The default backward keeps ~3× the sampled tensor alive between the
forward and the backward, or ~8.5× when `offset_groups == in_channels`. An
alternative backward keeps ~1×:

```python
import functools
from dc1d.nn import DeformConv1d
from dc1d.ops import efficient_linterpolate

model = DeformConv1d(
    in_channels=512,
    out_channels=512,
    kernel_size=3,
    padding="same",
    interpolation_function=functools.partial(efficient_linterpolate, gather_lerp="recompute"),
)
```

How much this is worth depends on `offset_groups`, and the answer changed in
2026-08 when the default gather was rewritten for ONNX correctness and took most
of the saving with it:

| | `offset_groups < in_channels` | `offset_groups == in_channels` |
|---|---|---|
| what the default already holds | ~3× the sampled tensor | ~8.5× |
| `recompute` peak memory, eager | ~1.3× lower | **1.6–2.1× lower** |
| cost, eager forward+backward | 10–23% slower | 10–23% slower |

So `recompute` is clearly worth it for depthwise offsets, which is the
configuration the DTCN paper uses, and marginal otherwise. Under
`torch.compile` it is **both** 0–12% faster *and* smaller, so if you compile
there is no reason not to use it. Full numbers, including why the hand-written
backward does *not* make eager faster, are in
[`benchmarks/BACKENDS.md`](benchmarks/BACKENDS.md) §5.9 and §5.9.4a.

### Examples and benchmarks

```
python playground/readme_example.py     # the snippet above
python playground/param_example.py      # large-dilation depthwise example
python benchmarks/benchmark.py          # timings (add --device cuda for GPU)
```

The benchmarks use `torch.utils.benchmark.Timer`, which warms up both paths
equally and synchronises CUDA around the timed region. Timings printed by
earlier versions of this repo (a bare `time.time()` around an async CUDA launch,
with warmup for the deformable path only) were not meaningful.

### Demo notebook

`docs/demo.ipynb` demonstrates, on tensors small enough to plot, that zero offsets
reproduce `nn.Conv1d` exactly, that an integer offset is an exact shift, that a
fractional offset interpolates linearly, that the offset gradient matches a finite
difference, that sampling positions survive float16, and that the offsets train. Every
exact property is asserted, so executing the notebook is itself a test. Figures are also
written to `docs/demo/`.

```
uv sync --group demo
uv run --group demo jupyter lab docs/demo.ipynb
uv run --group demo jupyter nbconvert --to notebook --execute --inplace docs/demo.ipynb
```

## Papers

Please cite the following if you use this package:

```
@INPROCEEDINGS{dtcn23,
  author={Ravenscroft, William and Goetze, Stefan and Hain, Thomas},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Deformable Temporal Convolutional Networks for Monaural Noisy Reverberant Speech Separation},
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095230}}
```
