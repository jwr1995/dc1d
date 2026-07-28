"""The example from README.md, runnable as a script."""

import torch
from torch import nn

# Import layer
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
bias = True
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
    bias=bias,
)
print(model)

# Generate input sequence
x = torch.rand(batch_size, in_channels, length, requires_grad=True)
print("Input shape:", x.shape)

# Generate offsets. `expected_offset_positions` is the closed form used by the
# layer itself, so it always agrees with the shape check inside forward().
output_length = model.expected_offset_positions(length)
offsets = nn.Parameter(torch.ones(batch_size, 1, output_length, kernel_size))

# Process the input sequence
y = model(x, offsets)
print("Output shape:", y.shape)

# For timing numbers use `python benchmarks/benchmark.py` -- a bare time.time()
# around a CUDA call measures dispatch, not execution.
