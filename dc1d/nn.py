""" 
nn.py provides classes for deformable convolution built on PyTorch functionality.

gLN and cLN layers are copied from the SpeechBrain framework:
https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/lobes/models/conv_tasnet.html
See licence here: https://github.com/speechbrain/speechbrain/blob/develop/LICENSE
Copyright SpeechBrain 2022.

The reset_paramters functions were adapted from the PyTorch ConvNd classes:
https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d
See licence here: https://github.com/pytorch/pytorch/blob/master/LICENSE
Copyright 2022, PyTorch Contributors.

The remainder of this module is original code belonging to the dc1d project.
Author: William Ravenscroft, August 2022
Copyright William Ravenscroft 2022.
"""

# Generic
import math, time
from turtle import forward
from typing import Optional, Tuple

# PyTorch
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _reverse_repeat_tuple
from typing import Callable

# dc1d
from dc1d.ops import kernel_width_linterpolate, full_seq_linterpolate, efficient_linterpolate

class DeformConv1d(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = "valid",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "reflect",
        device: str = "cpu",
        interpolation_function: Callable = efficient_linterpolate,
        unconstrained: str = None, # default None to maintain backwards compatibility
        *args,
        **kwargs
        ) -> None:
        """
        1D Deformable convolution kernel layer
        Args:
            in_channels (int): Value of convolution kernel size
            out_channels (int): Value of convolution kernel dilation factor
            kernel_size (int): Value of convolution kernel size
            stride (int): Value convolution kernel stride
            padding (int): See torch.nn.Conv1d for details. Default "valid". Still experimental beware of unexpected behaviour.
            dilation (int): Value of convolution kernel dilation factor
            groups (int) = 1
            bias (bool) = True
            padding_mode: See torch.nn.Conv1d for details. Default "reflect". Still experimental beware of unexpected behaviour.
            device: Device to operate function on. Default: torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        """

        self.device = device
        self.interpolation_function = interpolation_function
        padding_ = padding if isinstance(padding, str) else _single(padding)
        
        super(DeformConv1d, self).__init__(*args,**kwargs)
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
                
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding_ # note this is tuple-like for compatibility
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight = Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )

        self.dilated_positions = torch.linspace(0,
            dilation*kernel_size-dilation,
            kernel_size,
            ) # automatically store dilation offsets

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        if not unconstrained==None:
            self.unconstrained=unconstrained

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
    
    def __setstate__(self, state):
        super(DeformConv1d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


    def forward(
        self, 
        input: Tensor, 
        offsets: Tensor, 
        mask: Optional[Tensor] = None # TODO
        ) -> Tensor:
        """
        Forward pass of 1D deformable convolution layer
        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor
            offset (Tensor[batch_size, offset_groups, output length, kernel_size]):
                offsets to be applied for each position in the convolution kernel. Offset groups can be 1 or such that (in_channels%offset_groups == 0) is satisfied.
            mask (Tensor[batch_size, offset_groups, kernel_width, 1, out_width]): To be implemented

        Returns:
            output (Tensor[batch_size, in_channels, length]): output tensor
        """
        
        if self.padding_mode != 'zeros':
            input = F.pad(
                input, 
                self._reversed_padding_repeated_twice, 
                mode=self.padding_mode
                )
        if not self.device == offsets.device: # naive assumption
            self.device = offsets.device
        if self.dilated_positions.device != self.device:
            self.dilated_positions = self.dilated_positions.to(offsets.device)

        if "unconstrained" in self.__dict__.keys():
            input = self.interpolation_function(
                input, 
                kernel_size=self.kernel_size, 
                dilation=self.dilation,
                offsets=offsets, 
                stride=self.stride,
                dilated_positions=self.dilated_positions,
                device=self.device,
                unconstrained=self.unconstrained
                )
        else:
            input = self.interpolation_function(
                input, 
                kernel_size=self.kernel_size, 
                dilation=self.dilation,
                offsets=offsets, 
                stride=self.stride,
                dilated_positions=self.dilated_positions,
                device=self.device
                ) 
        input = input.flatten(-2,-1)
        output=F.conv1d(input, 
            self.weight, 
            self.bias, 
            stride=self.kernel_size, 
            groups=self.groups
            )

        return output

class PackedDeformConv1d(DeformConv1d):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = "valid",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "reflect",
        offset_groups: int = 1,
        device: str = "cpu",
        interpolation_function: Callable = efficient_linterpolate,
        unconstrained: str = None, # default None to maintain backwards compatibility
        *args,
        **kwargs
        ) -> None:
        """
        Packed 1D Deformable convolution class. Depthwise-Separable convolution is used to compute offsets.
        Args:
            in_channels (int): Value of convolution kernel size
            out_channels (int): Value of convolution kernel dilation factor
            kernel_size (int): Value of convolution kernel size
            stride (int): Value convolution kernel stride
            padding (int): See torch.nn.Conv1d for details. Default "valid". Still experimental beware of unexpected behaviour.
            dilation (int): Value of convolution kernel dilation factor
            groups (int): 1 or in_channels
            bias (bool): Whether to use bias. Default = True
            padding_mode (str): See torch.nn.Conv1d for details. Default "reflect". Still experimental beware of unexpected behaviour.
            offset_groups (int): 1 or in_channels
            device: Device to operate function on. Default: torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        """
        assert offset_groups in [1,in_channels], "offset_groups only implemented for offset_groups in {1,in_channels}"
        
        super(PackedDeformConv1d,self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            interpolation_function = interpolation_function,
            unconstrained=unconstrained,
            *args,
            **kwargs
            )
        self.offset_groups = offset_groups

        self.offset_dconv = nn.Conv1d(in_channels,in_channels,kernel_size,stride=1,groups=in_channels,padding=padding,padding_mode=padding_mode,bias=False)
        self.odc_norm = gLN(in_channels)
        self.odc_prelu = nn.PReLU()
        
        self.offset_pconv = nn.Conv1d(in_channels,kernel_size*offset_groups,1,stride=1,bias=False)
        self.odp_norm = gLN(kernel_size*offset_groups)
        self.odp_prelu = nn.PReLU()

        self.device=device
        self.to(device)
    
    def forward(self, input, with_offsets=False):
        """
        Forward pass of 1D deformable convolution layer
        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor
            
        Returns:
            output (Tensor[batch_size, in_channels, length]): output tensor
        """
        offsets = self.offset_dconv(input)
        offsets = self.odc_norm(self.odc_prelu(offsets).moveaxis(1,2)).moveaxis(2,1)

        self.device = offsets.device # naive assumption to fix errors

        assert str(input.device) == str(self.device), f"Input is on {input.device} but self is on {self.device}"
        assert str(input.device) == str(offsets.device), f"Input is on {input.device} but self is on {self.device}"

        offsets = self.offset_pconv(offsets)
        offsets = self.odp_norm(self.odp_prelu(offsets).moveaxis(1,2)).moveaxis(2,1) # batch_size x (kernel_size*offset_groups) x length
        offsets = offsets.unsqueeze(0).chunk(self.offset_groups,dim=2)# batch_size x offset_groups x length x kernel_size
        offsets = torch.vstack(offsets).moveaxis((0,2),(1,3))# batch_size x offset_groups x length x kernel_size

        if with_offsets:
            return super().forward(input,offsets), offsets
        else:
            return super().forward(input,offsets)

EPS=1e-9

class gLN(nn.Module):
    """Global Layer Normalization (gLN).

    Copyright SpeechBrain 2022

    Arguments
    ---------
    channel_size : int
        Number of channels in the third dimension.

    Example
    -------
    >>> x = torch.randn(2, 3, 3)
    >>> norm_func = GlobalLayerNorm(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super(gLN, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters."""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()


    def forward(self, y):
        """
        Arguments
        ---------
        y : Tensor
            Tensor shape [M, K, N]. M is batch size, N is channel size, and K is length.

        Returns
        -------
        gLN_y : Tensor
            Tensor shape [M, K. N]
        """
        mean = y.mean(dim=1, keepdim=True).mean(
            dim=2, keepdim=True
        )  # [M, 1, 1]
        var = (
            (torch.pow(y - mean, 2))
            .mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
        )
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class cLN(nn.Module):
    """Channel-wise Layer Normalization (cLN).

    Arguments
    ---------
    channel_size : int
        Number of channels in the normalization dimension (the third dimension).

    Example
    -------
    >>> x = torch.randn(2, 3, 3)
    >>> norm_func = ChannelwiseLayerNorm(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super(cLN, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters."""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()


    def forward(self, y):
        """
        Args:
            y: [M, K, N], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, K, N]
        """
        mean = torch.mean(y, dim=2, keepdim=True)  # [M, K, 1]
        var = torch.var(y, dim=2, keepdim=True, unbiased=False)  # [M, K, 1]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y

if __name__ == '__main__':
    # import time
    batch_size = 4
    in_channels = 512
    out_channels = 512
    kernel_size = 3
    stride = 1
    padding = "same"
    dilation = 2^7
    groups = 512
    bias = True
    length = 1998
    packed = True

    if packed:
        model = PackedDeformConv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = (kernel_size - 1) // 2, # "same",
            dilation = dilation,
            groups = groups,
            bias = True,
            offset_groups=in_channels,
            unconstrained=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        model = DeformConv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = "same",
            dilation = dilation,
            groups = groups,
            bias = True,
            unconstrained=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    x = torch.rand(batch_size, in_channels, length,requires_grad=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    print("Input shape",x.shape)
    
    output_length = x.shape[-1]-dilation*(kernel_size-1)
    
    offsets = nn.Parameter(torch.ones(batch_size, 1, 1998, kernel_size, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True))
    
    # Time DeformConv1d
    for i in range(3):
        start = time.time()  
        if type(model)==PackedDeformConv1d:
            y = model(x)
        else:
            y = model(x,offsets)
        end = time.time()
        print(f"Deformable runtime #{i} = {end-start}s")
    print("Output shape",y.shape)
    
    z = torch.mean(y)
    z.backward(retain_graph=True)

    if not packed:
        assert not offsets == None, "Offsets equal None... something has gone wrong"
        print(offsets.grad) # check gradients are not none

    vanilla_model = nn.Conv1d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        stride = stride,
        # padding = "zeros",
        dilation = dilation,
        groups = groups,
        bias = True,
        padding_mode = "reflect",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    start = time.time()
    y = vanilla_model(x)
    end = time.time()
    print("Vanilla shape",y.shape)
    print("Vanilla runtime =",end-start)

   
