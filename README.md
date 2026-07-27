# dc1d (DeformConv1d)

A 1D implementation of a deformable convolutional layer implemented in pure Python in PyTorch. The code style is designed to imitate similar classes in PyTorch such as `torch.nn.Conv1d` and `torchvision.ops.DeformConv2d`.

The motivation for creating this toolkit is that (as of 19/10/2022) there is no native 1D implementation of deformable convolution in the PyTorch library, and no alternative library which is simple to install (requiring only a basic PyTorch installation, with no additional compilation of C++ or CUDA libraries). The implementation here is written entirely in Python and makes use of `torch.autograd` for backpropagation.

## Requirements

* Python >= 3.10
* PyTorch >= 2.4 (installed automatically as a dependency)

`torchvision` is **not** required. Earlier releases imported private torchvision
symbols at module scope for a dead code path; that path has been removed.

Install PyTorch for your platform/CUDA version first if you want a specific
build: https://pytorch.org/get-started/locally/.

## Installation

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
