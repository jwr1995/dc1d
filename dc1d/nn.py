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
            interpolation_function (Callable): Interpolation kernel. Must accept
            the :func:`dc1d.ops.efficient_linterpolate` signature, including
            ``unconstrained``, which ``forward`` always passes. In practice
            that means ``efficient_linterpolate`` itself or a
            ``functools.partial`` of it, for example to set ``gather_lerp``.
            The reference kernels ``full_seq_linterpolate`` and
            ``kernel_width_linterpolate`` take no ``unconstrained`` argument
            and will raise ``TypeError`` here.
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
        if not hasattr(self, "modulated"):
            self.modulated = False

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
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass of 1D deformable convolution layer

        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor
            offsets (Tensor[batch_size, offset_groups, output_length, kernel_size]):
                offsets to be applied for each position in the convolution kernel.
                ``offset_groups`` may be 1 or any divisor of ``in_channels``.
            mask (Tensor[batch_size, offset_groups, output_length, kernel_size]):
                Optional modulation scalars, one per sampled position: this is the
                *v2* of Zhu et al. 2019 (DCNv2), which weights each tap by a
                learned scalar as well as moving it. Same shape as ``offsets``.
                Applied **as given**, exactly as ``torchvision.ops.deform_conv2d``
                does: if you want the ``[0, 1]`` modulation of the paper, pass
                ``mask.sigmoid()``. ``None`` (default) is plain DCNv1.

        Returns:
            output (Tensor[batch_size, out_channels, output_length]): output tensor
        """
        if mask is not None and mask.shape != offsets.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match offsets shape "
                f"{tuple(offsets.shape)} (batch, offset_groups, output_length, kernel_size)"
            )

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

        # Modulation (DCNv2). The interpolated tensor is
        # (batch, in_channels, output_length, kernel_size) and its channel axis
        # runs as (offset_groups, channels_per_group) -- the same split the
        # interpolation kernel gathers under -- so unflattening it lines each
        # group's channels up against that group's mask.
        if mask is not None:
            input = (
                input.unflatten(1, (mask.shape[1], -1)) * mask.unsqueeze(2).to(input.dtype)
            ).flatten(1, 2)

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
        modulated: bool = False,
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
            interpolation_function (Callable): Interpolation kernel. Must accept
            the :func:`dc1d.ops.efficient_linterpolate` signature, including
            ``unconstrained``, which ``forward`` always passes. In practice
            that means ``efficient_linterpolate`` itself or a
            ``functools.partial`` of it, for example to set ``gather_lerp``.
            The reference kernels ``full_seq_linterpolate`` and
            ``kernel_width_linterpolate`` take no ``unconstrained`` argument
            and will raise ``TypeError`` here.
            unconstrained (bool): See DeformConv1d.
            modulated (bool): Predict a DCNv2 modulation mask alongside the
                offsets. Default False, which is plain DCNv1 and leaves the
                parameter count unchanged. The mask head is a second pointwise
                branch off the shared depthwise trunk, structurally identical to
                the offset head, ending in a sigmoid so the mask lands in
                ``(0, 1)`` as in Zhu et al. 2019.
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

        self.modulated = bool(modulated)
        if self.modulated:
            self.mask_pconv = nn.Conv1d(
                in_channels, kernel_size * offset_groups, 1, stride=1, bias=False
            )
            self.mdp_norm = gLN(kernel_size * offset_groups)
            self.mdp_prelu = nn.PReLU()
            # Zero the last projection so every tap starts at sigmoid(0) == 0.5,
            # uniformly. This is the DCNv2 reference initialisation: the mask
            # starts uninformative and the layer has to learn to gate. Note it
            # halves the output scale at initialisation relative to modulated=False.
            init.zeros_(self.mask_pconv.weight)

        if device is not None:
            self.to(device)

    def _to_offset_layout(self, y: Tensor) -> Tensor:
        """(B, kernel_size*offset_groups, L) -> (B, offset_groups, L, kernel_size)."""
        chunks = y.unsqueeze(0).chunk(self.offset_groups, dim=2)
        return torch.vstack(chunks).moveaxis((0, 2), (1, 3))

    def forward(self, input: Tensor, with_offsets: bool = False):
        """
        Forward pass of the packed 1D deformable convolution layer

        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor
            with_offsets (bool): also return the computed offsets. When
                ``modulated=True`` the return is
                ``(output, (offsets, mask))`` rather than ``(output, offsets)``.

        Returns:
            output (Tensor[batch_size, out_channels, output_length]): output tensor
        """
        trunk = self.offset_dconv(input)
        trunk = self.odc_norm(self.odc_prelu(trunk).moveaxis(1, 2)).moveaxis(2, 1)

        offsets = self.offset_pconv(trunk)
        # batch_size x (kernel_size*offset_groups) x length
        offsets = self.odp_norm(self.odp_prelu(offsets).moveaxis(1, 2)).moveaxis(2, 1)
        # batch_size x offset_groups x length x kernel_size
        offsets = self._to_offset_layout(offsets)

        mask = None
        if self.modulated:
            mask = self.mask_pconv(trunk)
            mask = self.mdp_norm(self.mdp_prelu(mask).moveaxis(1, 2)).moveaxis(2, 1)
            # Sigmoid here, not in DeformConv1d.forward: the base layer applies a
            # caller-supplied mask verbatim (torchvision's contract), so the
            # squashing belongs to whoever predicts the mask.
            mask = self._to_offset_layout(mask).sigmoid()

        output = super().forward(input, offsets, mask)
        if with_offsets:
            return output, (offsets, mask) if self.modulated else offsets
        return output


EPS = 1e-9


def _rms(var: Tensor) -> Tensor:
    """
    ``sqrt(var)``, floored so that a zero-variance input cannot divide by zero.

    Written as ``clamp_min`` rather than the usual ``var + EPS`` because the
    ONNX exporter **deletes** a sufficiently small additive constant: measured
    2026-08, ``Add(var, 1e-9)`` and ``Add(var, 1e-8)`` are both folded out of
    the graph (``1e-5`` survives), leaving ``Div(x, Pow(var, 0.5))``. On a
    constant input that is ``0 / 0``, so the exported model returns NaN where
    eager PyTorch returns zeros, silently. Rewriting the square root as ``sqrt``
    or ``rsqrt`` does not help; the additive term is what gets dropped.

    ``clamp_min`` survives export at any magnitude, and for any non-degenerate
    input it is a closer match to eager than the addition was: it is a no-op
    wherever ``var > EPS``, whereas the addition perturbs every value.
    """
    return torch.pow(torch.clamp_min(var, EPS), 0.5)


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
        gLN_y = self.gamma * (y - mean) / _rms(var) + self.beta
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
        cLN_y = self.gamma * (y - mean) / _rms(var) + self.beta
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
