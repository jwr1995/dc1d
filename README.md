# DeformConv1D
An 1D implementation of a deformable convolutional layer in PyTorch. The code style is designed to imitate similar classes in PyTorch such as ```torch.nn.Conv1D``` and ```torchvision.ops.DeformConv2D```.

# Requirements
You must install PyTorch. Follow the details on the website to install properly: https://pytorch.org/get-started/locally/

This package was most thoroughly tested on PyTorch 1.12.1 running CUDA 11.6 on Windows with Python version 3.8.

# Installation
```
pip install dc1d
```

# Usage
## Example of how to use the deformable convolutional layer ```DeformConv1d()``` with timing information.
```DeformConv1d``` is the deformable convolution layer designed to imitate ```torch.nn.Conv1d```.
Note: ```DeformConv1d``` does not compute the offset values used in its ```forward(...)``` call. These most be computed outside the layer.

```
from dc1d.nn import DeformConv1d

import time
batch_size = 16
in_channels = 512
out_channels = 512
kernel_size = 16
stride = 1
padding = "same"
dilation = 3
groups = 1
bias = True
length = 128

model = DeformConv1d(
    in_channels = in_channels,
    out_channels = out_channels,
    kernel_size = kernel_size,
    stride = stride,
    padding = "same",
    dilation = dilation,
    groups = groups,
    bias = True,
    device="cuda"
)

x = torch.rand(batch_size, in_channels, length,requires_grad=True).cuda()
print(x.shape)

output_length = x.shape[-1]-dilation*(kernel_size-1)
offsets = nn.Parameter(torch.ones(batch_size, 1, output_length, kernel_size, requires_grad=True, device="cuda"))

start = time.time()
y = model(x, offsets)
end = time.time()
print(y.shape)
print("Deformable runtime =",end-start)
```
---
For more detailed examples, the ```nn``` and ```ops``` modules have example usage scripts appended to the bottom of the file inside their ```if __name__ == "__main__":``` clauses. For example one could run 
```
python dc1d.nn.py
```
to compare the runtime of our ```DeformConv1d``` layer against ```torch.nn.Conv1d```.

# Some notes for those interested
I experimented with retooling the ```torchvision.ops.deform_conv2d``` function for 1d convolution. This seemed to work but it is beyond my current capability to verify whether it is operating as intended. I can provide this code for anyone interested as it is faster than the code here but should be tested thoroughly before using. The tensor shapes are correct but the offset shape is designed implicitly in the ```deform_conv2d`` function so it is unknown how the data is being reshaped exactly inside the function unless you are able to find and understand the raw cuda and c++ in the kernel code.

The implementation here is written entirely in python. Interestingly it operates faster when you have offsets for every position in a sequence, every channel and every kernel position as opposed to just every kernel position. This is most likely due to unintuitive Tensor operations on $\mathbb{R}^{\ldots \times 1 \times \ldots}$ shaped Tensors where it is looping rather than copying data and performing a multiprocessed operation (just my hunch). I'll look at this in future if PyTorch still haven't implemented their own 1D deformable convolution (sadly I have zero experience with CUDA programming myself and don't have the time to get proficient to do it competently right now).
