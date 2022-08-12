# DeformConv1D
An 1D implementation of a deformable convolutional layer in PyTorch. The code style is designed to imitate similar classes in PyTorch such as ```torch.nnet.Conv1D``` and ```torchvision.ops.DeformConv2D```.

# Requirements
You must install PyTorch. Follow the details on the website to install properly: https://pytorch.org/get-started/locally/

This package was tested on PyTorch 1.12.1 running CUDA 11.6 on Windows using Python version 3.8.

# Installation
```
pip install dc1d
```

# Usage
```
from dc1d import DeformConv1D
```

# Some notes for those interested
I experimented with retooling the ```torchvision.ops.deform_conv2d``` function for 1d convolution. This seemed to work but it is beyond my current capability to verify whether it is operating as intended. I can provide this code for anyone interested as it is faster than the code here but should be tested thoroughly before using. The tensor shapes are correct but the offset shape is designed implicitly in the ```deform_conv2d`` function so it is unknown how the data is being reshape exactly inside the function unless you are able to understand the raw cuda and c++ in the kernel code.

The implementation here is written entirely in python. Interestingly it operates faster when you have offsets for every position in a sequence, every channel and every kernel position as opposed to just every kernel position. This is most likely due to unintuitive Tensor operations on $\mathbb{R}^{\ldots \times 1 \times \ldots}$ shaped Tensors where it is looping rather than copying data and performing a multiprocessed operation (just my hunch). I'll look at this in future if PyTorch still haven't implemented their own 1D deformable convolution (sadly I have zero experience with CUDA programming myself and don't have the time to get proficient to do it competently right now).
