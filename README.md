# DeformConv1D
An 1D implementation of a deformable convolutional layer implemented in pure Python in PyTorch. The code style is designed to imitate similar classes in PyTorch such as ```torch.nn.Conv1D``` and ```torchvision.ops.DeformConv2D```.

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
import time

# Import layer
from dc1d.nn import DeformConv1d

# Hyperparameters
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

# Construct layer
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

# Generate input sequence
x = torch.rand(batch_size, in_channels, length,requires_grad=True).cuda()
print(x.shape)

# Generate offsets by first computing the desired output length
output_length = x.shape[-1]-dilation*(kernel_size-1)
offsets = nn.Parameter(torch.ones(batch_size, 1, output_length, kernel_size, requires_grad=True, device="cuda"))

# Process the input sequence and time it
start = time.time()
y = model(x, offsets)
end = time.time()

# Print output shape and time taken
print(y.shape)
print("Deformable runtime =",end-start)
```
---
For more detailed examples, the ```nn``` and ```ops``` modules have example usage scripts appended to the bottom of the file inside their ```if __name__ == "__main__":``` clauses. For example one could run 
```
python dc1d.nn.py
```
to compare the runtime of our ```DeformConv1d``` layer against ```torch.nn.Conv1d```.

# Papers
I will soon publish a paper using this package in the mean time if you find this package useful please cite one of my other works below.
```
@article{ravenscroft2022receptive,
  title={Receptive Field Analysis of Temporal Convolutional Networks for Monaural Speech Dereverberation},
  author={Ravenscroft, William and Goetze, Stefan and Hain, Thomas},
  journal={arXiv preprint arXiv:2204.06439},
  year={2022}
}

@article{ravenscroft2022utterance,
  title={Utterance Weighted Multi-Dilation Temporal Convolutional Networks for Monaural Speech Dereverberation},
  author={Ravenscroft, William and Goetze, Stefan and Hain, Thomas},
  journal={arXiv preprint arXiv:2205.08455},
  year={2022}
}
```

# Some notes for those interested
I experimented with retooling the ```torchvision.ops.deform_conv2d``` function for 1d convolution. This seemed to work but it is beyond my current capability to verify whether it is operating as intended. I can provide this code for anyone interested as it is faster than the code here but should be tested thoroughly before using. The tensor shapes are correct but the offset shape is designed implicitly in the ```deform_conv2d`` function so it is unknown how the data is being reshaped exactly inside the function unless you are able to find and understand the raw cuda and c++ in the kernel code. A similar project tvdcn also exists which has adapted the raw 2D CUDA kernel operations to 1D and 3D operations: https://github.com/inspiros/tvdcn. I decided to pursue this alternate implementation anyway for the aforementioned implict tensor reshaping mentioned above.

The implementation here is written entirely in python. Interestingly it operates faster when you have offsets for every position in a sequence, every channel and every kernel position as opposed to just every kernel position. This is most likely due to unintuitive Tensor operations on $\mathbb{R}^{\ldots \times 1 \times \ldots}$ shaped Tensors where it is looping rather than copying data and performing a multiprocessed operation (just my hunch). I'll look at this in future if PyTorch still haven't implemented their own 1D deformable convolution.
