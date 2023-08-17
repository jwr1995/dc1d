import time
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
    in_channels = in_channels,
    out_channels = out_channels,
    kernel_size = kernel_size,
    stride = stride,
    padding = padding,
    dilation = dilation,
    groups = groups,
    bias = True,
)

# Generate input sequence
x = torch.rand(batch_size, in_channels, length,requires_grad=True)
print(x.shape)

# Generate offsets by first computing the desired output length
output_length = x.shape[-1]-dilation*(kernel_size-1)
offsets = nn.Parameter(torch.ones(batch_size, 1, output_length, kernel_size, requires_grad=True))

# Process the input sequence and time it
start = time.time()
y = model(x, offsets)
end = time.time()

# Print output shape and time taken
print(y.shape)
print("Deformable runtime =",end-start)