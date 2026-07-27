"""
Parameter-sweep example: a large-dilation depthwise deformable convolution with
'same' padding, compared against nn.Conv1d.

For timing numbers use `python benchmarks/benchmark.py`, which synchronises CUDA
properly and warms both paths up equally.
"""

import torch
from torch import nn

from dc1d.nn import DeformConv1d, PackedDeformConv1d

batch_size = 4
in_channels = 64
out_channels = 64
kernel_size = 3
stride = 1
padding = "same"
# NOTE: this used to read `2^7`, which is XOR in Python and evaluates to 5. The
# "large dilation" example had been exercising dilation=5 since 2022.
dilation = 2**5
groups = 64
bias = True
length = 133
packed = False

device = "cuda" if torch.cuda.is_available() else "cpu"

if packed:
    model = PackedDeformConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        offset_groups=in_channels,
        unconstrained=True,
        device=device,
    )
else:
    model = DeformConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        unconstrained=True,
        device=device,
    )

print(model)
print("Parameters:", sum(p.numel() for p in model.parameters()))

x = torch.rand(batch_size, in_channels, length, requires_grad=True, device=device)
print("Input shape", x.shape)

output_length = model.expected_offset_positions(length)
offsets = nn.Parameter(torch.ones(batch_size, 1, output_length, kernel_size, device=device))

y = model(x) if packed else model(x, offsets)
print("Output shape", y.shape)

torch.mean(y).backward()

if not packed:
    assert offsets.grad is not None, "Offsets have no gradient... something has gone wrong"
    print("Offset gradient norm:", offsets.grad.norm().item())

vanilla_model = nn.Conv1d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    groups=groups,
    bias=bias,
    padding_mode="reflect",
    device=device,
)
print("Vanilla shape", vanilla_model(x).shape)
print("Vanilla parameters:", sum(p.numel() for p in vanilla_model.parameters()))
