import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _reverse_repeat_tuple

from .ops import linterpolate

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
        *args,
        **kwargs
        ) -> None:

        self.device = device
        
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
            )

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
        # mask: Optional[Tensor] = None # TODO
        ) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor
            offset (Tensor[batch_size, offset_groups * kernel_width, otuput length]):
                offsets to be applied for each position in the convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_width, 1, out_width]):
                masks to be applied for each position in the convolution kernel.
        """
        
        if self.padding_mode != 'zeros':
            input = F.pad(
                input, 
                self._reversed_padding_repeated_twice, 
                mode=self.padding_mode
                )

        x_offset = linterpolate(
            x, 
            kernel_size=self.kernel_size, 
            dilation=self.dilation,
            offsets=offsets, 
            stride=stride,
            dilated_positions=self.dilated_positions,
            device=self.device
            ) 

        output=F.conv1d(x_offset.flatten(-2,-1), 
            self.weight, 
            self.bias, 
            stride=self.kernel_size, 
            groups=self.groups
            )

        return output

if __name__ == '__main__':
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
    
    x = torch.rand(batch_size, in_channels, length,requires_grad=True).cuda()
    print(x.shape)
    
    output_length = x.shape[-1]-dilation*(kernel_size-1)
    offsets = nn.Parameter(torch.ones(batch_size, 1, output_length, kernel_size, requires_grad=True, device="cuda"))

    start = time.time()
    y = model(x, offsets)
    end = time.time()
    print(y.shape)
    print("Deformable runtime =",end-start)
    z = torch.mean(y)
    z.backward(retain_graph=True)
    print(type(offsets.grad) == type(None))
    # print(offsets.grad)

    start = time.time()
    y = vanilla_model(x)
    end = time.time()
    print(y.shape)
    print("Vanilla runtime =",end-start)

   