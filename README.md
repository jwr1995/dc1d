# dc1d (DeformConv1d)
A 1D implementation of a deformable convolutional layer implemented in pure Python in PyTorch. The code style is designed to imitate similar classes in PyTorch such as ```torch.nn.Conv1D``` and ```torchvision.ops.DeformConv2D```.

The motivation for creating this toolkit is as of 19/10/2022 there is no native 1D implementation of deformable convolution in the PyTorch library and no alternate library which is simple to install (requiring only a basic PyTorch installation with no additional compilation of c++ or cuda libraries). The implementation here is written entirely in Python and makes use of ```torch.autograd``` for backpropagation.

# Requirements
You must install PyTorch. Follow the details on the website to install properly: https://pytorch.org/get-started/locally/.

This package was most thoroughly tested on PyTorch 1.12.1 running CUDA 11.6 on Windows with Python version 3.8. It has also been tested on Ubuntu, Zorin OS and CentOS all running Python 3.8.

# Installation
Please use the following ```pip``` command to install for now.

Most thoroughly tested version (only "reflect" padding mode is implemented on this version, please do not try to use any other):
```
pip install dc1d==0.0.4
```
Alternatively to install the latest version do
```
pip install dc1d
```
The current version on this branch has not been tested for numerical accuracy but probably works fine. Gradient computation has been tested and seems okay.
To download direct from source do
```
git clone https://github.com/jwr1995/dc1d.git
cd dc1d
pip install .
```


# Usage
## Example of how to use the deformable convolutional layer ```DeformConv1d()``` with timing information.
```DeformConv1d``` is the deformable convolution layer designed to imitate ```torch.nn.Conv1d```.
Note: ```DeformConv1d``` does not compute the offset values used in its ```forward(...)``` call. These most be computed outside the layer.

```
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
)

# Generate input sequence
x = torch.rand(batch_size, in_channels, length,requires_grad=True)
print(x.shape)

# Generate offsets by first computing the desired output length
offsets = nn.Parameter(torch.ones(batch_size, 1, length, kernel_size, requires_grad=True))

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
python dc1d/nn.py
```
to compare the runtime of our ```DeformConv1d``` layer against ```torch.nn.Conv1d```.

A class called ```PackedConv1d``` also exists in ```dc1d.nn``` which computes the offsets using a depthwise-separable convolutional block as detailed in our paper below.

# Papers
Please cite the following if you use this package
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
