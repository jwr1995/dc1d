""" 
nn.py provides classes for deformable convolution built on PyTorch functionality.

Author: William Ravenscroft, August 2022
Copyright William Ravenscroft 2022
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
from torch.nn.modules.utils import _reverse_repeat_tuple
from typing import Callable

# Speechbrain
from speechbrain.lobes.models.conv_tasnet import GlobalLayerNorm as gLN

# dc1d
from dc1d.ops import kernel_width_linterpolate, full_seq_linterpolate

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
        device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        interpolation_function: Callable = kernel_width_linterpolate,
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
        
        super(DeformConv1d, self).__init__(*args,**kwargs)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0]
            if padding == 'same':
                for d, k, i in zip([dilation], [kernel_size], range( 0, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight = Parameter(
            torch.empty(out_channels, in_channels // groups, self.kernel_size)
        )

        self.dilated_positions = torch.linspace(0,
            dilation*kernel_size-dilation,
            kernel_size,
            # requires_grad=True, 
            device=device
            ) # automatically store dilation offsets

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

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
                offsets to be applied for each position in the convolution kernel. Offset groups can be 1 or such that (in_channels%offset_grous == 0) is satisfied.
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

        input = self.interpolation_function(
            input, 
            kernel_size=self.kernel_size, 
            dilation=self.dilation,
            offsets=offsets, 
            stride=stride,
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
        device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        interpolation_function: Callable = kernel_width_linterpolate,
        *args,
        **kwargs
        ) -> None:
        assert offset_groups in [1,in_channels], "offset_groups only implemented for offset_gorups in {1,in_channels}"
        
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
        
        # self.offset_pconv_test = nn.Conv1d(in_channels,kernel_size*offset_groups,1,stride=1,bias=False)

        self.device=device
        self.to(device)
        print(self.offset_dconv.weight.shape)
    
    def forward(self, x):
        offsets = self.offset_dconv(x)
        offsets = self.odc_norm(self.odc_prelu(offsets).moveaxis(1,2)).moveaxis(2,1)


        offsets = self.offset_pconv(offsets)
        offsets = self.odp_norm(self.odp_prelu(offsets).moveaxis(1,2)).moveaxis(2,1) # batch_size x (kernel_size*offset_groups) x length

        offsets = offsets.unsqueeze(0).chunk(self.offset_groups,dim=2)# batch_size x offset_groups x length x kernel_size
        
        offsets = torch.vstack(offsets).moveaxis((0,2),(1,3))# batch_size x offset_groups x length x kernel_size
        
        return super().forward(x,offsets)

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

    model = PackedDeformConv1d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        stride = stride,
        padding = "same",
        dilation = dilation,
        groups = groups,
        bias = True,
        offset_groups=in_channels,
        device="cuda"
    )

    x = torch.rand(batch_size, in_channels, length,requires_grad=True).cuda()
    print("Input shape",x.shape)
    
    output_length = x.shape[-1]-dilation*(kernel_size-1)
    
    offsets = torch.ones(batch_size, 1, 1998, kernel_size, device="cuda", requires_grad=True)
    
    # Time DeformConv1d
    for i in range(3):
        start = time.time()  
        y = model(x)
        end = time.time()
        print(f"Deformable runtime #{i} = {end-start}s")
    print("Output shape",y.shape)
    
    z = torch.mean(y)
    z.backward(retain_graph=True)

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
        device="cuda"
    )
    
    start = time.time()
    y = vanilla_model(x)
    end = time.time()
    print("Vanilla shape",y.shape)
    print("Vanilla runtime =",end-start)

   