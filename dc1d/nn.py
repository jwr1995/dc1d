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

from __future__ import annotations

# Generic
import math
from collections.abc import Callable

# PyTorch
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init
from torch.nn.modules.utils import _reverse_repeat_tuple, _single
from torch.nn.parameter import Parameter

# dc1d
from dc1d.ops import (
    efficient_linterpolate,
    full_seq_linterpolate,  # noqa: F401 -- re-exported for back-compat
    kernel_width_linterpolate,  # noqa: F401 -- re-exported for back-compat
    output_length,
)

__all__ = ["DeformConv1d", "PackedDeformConv1d", "gLN", "cLN"]


class DeformConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = "valid",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "reflect",
        device: torch.device | str | None = None,
        interpolation_function: Callable = efficient_linterpolate,
        unconstrained: bool | None = None,  # default None to maintain backwards compatibility
        *args,
        **kwargs,
    ) -> None:
        """
        1D Deformable convolution kernel layer

        Args:
            in_channels (int): Number of channels in the input signal
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int): Value of convolution kernel size
            stride (int): Value convolution kernel stride
            padding (int or str): See torch.nn.Conv1d for details. Default "valid".
                Still experimental, beware of unexpected behaviour.
            dilation (int): Value of convolution kernel dilation factor
            groups (int): Number of blocked connections from input to output channels
            bias (bool): Whether to add a learnable bias. Default True
            padding_mode (str): See torch.nn.Conv1d for details. Default "reflect".
                Still experimental, beware of unexpected behaviour.
            device: Optional device to move the layer to on construction. The forward
                pass always follows the device of its input; this is a convenience
                kwarg only.
            interpolation_function (Callable): Interpolation kernel from dc1d.ops.
            unconstrained (bool): If True, kernel taps may sample anywhere in the
                sequence rather than being confined to their own receptive field.
                Default None, treated as False.
        """
        super().__init__(*args, **kwargs)

        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        padding_ = padding if isinstance(padding, str) else _single(padding)
        stride_ = _single(stride)
        dilation_ = _single(dilation)
        kernel_size_ = _single(kernel_size)

        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride_):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, "
                f"but got padding_mode='{padding_mode}'"
            )

        self.interpolation_function = interpolation_function
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding_  # note this is tuple-like for compatibility
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        # Plain bool, set unconditionally. Previously this attribute was only
        # created when the caller passed a value, and `forward` dispatched on its
        # presence in `self.__dict__` -- which broke under any form of module
        # reconstruction and is invisible to torch.compile.
        self.unconstrained = bool(unconstrained)

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)
            if padding == "same":
                for d, k, i in zip(
                    dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1), strict=True
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight = Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))

        # Registered as a (non-persistent) buffer so that `.to()`/`.cuda()` move it
        # automatically. As a plain attribute it was invisible to both, which is
        # why `forward` used to hand-patch its device.
        self.register_buffer(
            "dilated_positions",
            torch.arange(kernel_size, dtype=torch.float32) * dilation,
            persistent=False,
        )

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        if device is not None:
            self.to(device)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
        s += f", stride={self.stride}"
        if self.padding != (0,) and self.padding != 0:
            s += f", padding={self.padding!r}"
        if self.dilation != 1:
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode}"
        if self.unconstrained:
            s += ", unconstrained=True"
        return s

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"
        if not hasattr(self, "unconstrained"):
            self.unconstrained = False

    def _pad(self, input: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            return F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        if any(p != 0 for p in self._reversed_padding_repeated_twice):
            return F.pad(input, self._reversed_padding_repeated_twice, mode="constant", value=0)
        return input

    def expected_offset_positions(self, length: int) -> int:
        """Number of offset positions required for an input of ``length`` samples."""
        padded = length + sum(self._reversed_padding_repeated_twice)
        return output_length(padded, self.kernel_size, self.dilation, self.stride)

    def forward(
        self,
        input: Tensor,
        offsets: Tensor,
        mask: Tensor | None = None,  # TODO
    ) -> Tensor:
        """
        Forward pass of 1D deformable convolution layer

        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor
            offsets (Tensor[batch_size, offset_groups, output_length, kernel_size]):
                offsets to be applied for each position in the convolution kernel.
                ``offset_groups`` may be 1 or any divisor of ``in_channels``.
            mask (Tensor[batch_size, offset_groups, kernel_width, 1, out_width]):
                To be implemented

        Returns:
            output (Tensor[batch_size, out_channels, output_length]): output tensor
        """
        if mask is not None:
            raise NotImplementedError("masked (deformable v2) convolution is not implemented")

        in_shape = input.shape
        input = self._pad(input)

        # Loud failure instead of silent corruption: an inconsistent number of
        # offset positions used to be absorbed by the index clamp in
        # `efficient_linterpolate`, producing a wrong-length output.
        expected = output_length(input.shape[-1], self.kernel_size, self.dilation, self.stride)
        if offsets.shape[-2] != expected:
            raise ValueError(
                f"offsets has {offsets.shape[-2]} positions but input of length "
                f"{in_shape[-1]} (padded to {input.shape[-1]}) with kernel_size="
                f"{self.kernel_size}, stride={self.stride}, dilation={self.dilation} "
                f"requires {expected}"
            )

        input = self.interpolation_function(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            offsets=offsets,
            stride=self.stride,
            dilated_positions=self.dilated_positions,
            device=input.device,
            unconstrained=self.unconstrained,
        )
        input = input.flatten(-2, -1)
        output = F.conv1d(
            input,
            self.weight,
            self.bias,
            stride=self.kernel_size,
            groups=self.groups,
        )
        if self.padding == "same" and in_shape[-1] != output.shape[-1]:
            raise RuntimeError(
                f"padding='same' but input length {in_shape[-1]} and output length "
                f"{output.shape[-1]} do not match."
            )
        return output


class PackedDeformConv1d(DeformConv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = "valid",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "reflect",
        offset_groups: int = 1,
        device: torch.device | str | None = None,
        interpolation_function: Callable = efficient_linterpolate,
        unconstrained: bool | None = None,  # default None to maintain backwards compatibility
        *args,
        **kwargs,
    ) -> None:
        """
        Packed 1D Deformable convolution class. Depthwise-separable convolution is
        used to compute the offsets.

        Args:
            in_channels (int): Number of channels in the input signal
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int): Value of convolution kernel size
            stride (int): Value convolution kernel stride
            padding (int or str): See torch.nn.Conv1d for details. Default "valid".
                Still experimental, beware of unexpected behaviour.
            dilation (int): Value of convolution kernel dilation factor
            groups (int): Number of blocked connections from input to output channels
            bias (bool): Whether to use bias. Default True
            padding_mode (str): See torch.nn.Conv1d for details. Default "reflect".
                Still experimental, beware of unexpected behaviour.
            offset_groups (int): Any divisor of in_channels. Default 1.
            device: Optional device to move the layer to on construction.
            interpolation_function (Callable): Interpolation kernel from dc1d.ops.
            unconstrained (bool): See DeformConv1d.
        """
        if offset_groups <= 0 or in_channels % offset_groups != 0:
            raise ValueError(
                f"offset_groups ({offset_groups}) must be a positive divisor of "
                f"in_channels ({in_channels})"
            )

        super().__init__(
            *args,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            interpolation_function=interpolation_function,
            unconstrained=unconstrained,
            **kwargs,
        )
        self.offset_groups = offset_groups

        # `stride` and `dilation` must match the deformable path, otherwise this
        # conv emits the wrong number of offset positions. They used to be
        # hardcoded to stride=1 / dilation=1.
        self.offset_dconv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )
        self.odc_norm = gLN(in_channels)
        self.odc_prelu = nn.PReLU()

        self.offset_pconv = nn.Conv1d(
            in_channels, kernel_size * offset_groups, 1, stride=1, bias=False
        )
        self.odp_norm = gLN(kernel_size * offset_groups)
        self.odp_prelu = nn.PReLU()

        if device is not None:
            self.to(device)

    def forward(self, input: Tensor, with_offsets: bool = False):
        """
        Forward pass of the packed 1D deformable convolution layer

        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor
            with_offsets (bool): also return the computed offsets

        Returns:
            output (Tensor[batch_size, out_channels, output_length]): output tensor
        """
        offsets = self.offset_dconv(input)
        offsets = self.odc_norm(self.odc_prelu(offsets).moveaxis(1, 2)).moveaxis(2, 1)

        offsets = self.offset_pconv(offsets)
        # batch_size x (kernel_size*offset_groups) x length
        offsets = self.odp_norm(self.odp_prelu(offsets).moveaxis(1, 2)).moveaxis(2, 1)
        offsets = offsets.unsqueeze(0).chunk(self.offset_groups, dim=2)
        # batch_size x offset_groups x length x kernel_size
        offsets = torch.vstack(offsets).moveaxis((0, 2), (1, 3))

        if with_offsets:
            return super().forward(input, offsets), offsets
        return super().forward(input, offsets)


EPS = 1e-9


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
    >>> norm_func = gLN(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.empty(1, 1, channel_size))  # [1, 1, N]
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
            Tensor shape [M, K, N]
        """
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
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
    >>> norm_func = cLN(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.empty(1, 1, channel_size))  # [1, 1, N]
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


if __name__ == "__main__":
    # Small smoke demo. For timing numbers use `python benchmarks/benchmark.py`,
    # which uses torch.utils.benchmark (correct CUDA synchronisation and warmup).
    batch_size = 4
    in_channels = 64
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = "same"
    dilation = 2**7  # NOTE: this used to read `2^7`, which is XOR and evaluates to 5
    groups = 64
    length = 133

    model = DeformConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
        unconstrained=True,
    )
    print(model)

    x = torch.rand(batch_size, in_channels, length, requires_grad=True)
    print("Input shape", x.shape)

    n_offsets = model.expected_offset_positions(length)
    offsets = nn.Parameter(torch.ones(batch_size, 1, n_offsets, kernel_size))

    y = model(x, offsets)
    print("Output shape", y.shape)

    torch.mean(y).backward()
    assert offsets.grad is not None, "Offsets have no gradient... something has gone wrong"
    print("Offset grad norm:", offsets.grad.norm().item())
