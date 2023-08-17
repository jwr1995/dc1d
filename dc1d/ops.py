""" 
ops.py provides linear interpolation operation functions for deformable convolution

Author: William Ravenscroft, August 2022
Copyright William Ravenscroft 2022
"""

# PyTorch
from multiprocessing.dummy import Pool
import torch
from torch import Tensor
import torch.multiprocessing as mp

# Torchvision
from torchvision.extension import _assert_has_ops
from torchvision.utils import _log_api_usage_once


# from torch.cuda.amp import autocast
from functools import partial
from typing import Optional, Tuple

def full_seq_linterpolate(
    x, 
    offsets,
    kernel_size, 
    dilation,
    stride,
    dilated_positions=None,
    device="cpu",
    _test=False
    ):
    """
    Full sequence linear interpolation function for 1D deformable convolution. This should only be used for short sequence lengths else the user will be likely to run into memory issues.
    Args:
        x (Tensor): Input Data Tensor of shape batch size x channels x length
        offsets (Tensor): Deforming offset Tensor of shape batch size x offset groups x number of offset positions x kernel size
        kernel_size (int): Value of convolution kernel size
        dilation (int): Value of convolution kernel dilation factor
        stride (int): Value convolution kernel stride
        dilated_positions (Tensor): Allows user to save computation by using precomputed dilation offset positions. If not these can be computed from dilation kernel_size for each function call
        device: Device to operate function on. Default: "cpu".
    """
    # Every index in x we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(0, dilation*kernel_size-dilation,kernel_size,device=device) # kernel_size
    max_t0 = (offsets.shape[-2]-1)*stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2],device=device).unsqueeze(-1) # out_length x 1
    dilated_offsets_repeated = dilated_positions+offsets
    T = t0s + dilated_offsets_repeated # batch_size x groups x out_length x kernel_size

    if _test:
        print("x:",x.shape) # batch_size x in_channels x input_length
        print("offsets:",offsets.shape) # batch_size x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:",t0s.shape) # out_lengths x 1
        print("dilated positions:",dilated_positions.shape) # kernel_size
        print("dilated_offsets_repeated:",dilated_offsets_repeated.shape)
    
    
    max_U = x.shape[-1]-1
    U = torch.linspace(0,max_U,max_U+1,device=device).repeat(1,1,1,1,1) # 1 x 1 x 1 x 1 x_length
    abs_sub = 1-torch.abs(U-T.unsqueeze(-1)) # batch_size x groups x out_length x kernel_size x_length
    _zeros = torch.zeros(abs_sub.shape,device=device)
    G = torch.max(_zeros, abs_sub) # batch_size x groups x out_length x kernel_size x_length
    
    if _test:
        print("T:",T.shape) # batch_size x groups x out_length x kernel_size
        print("U:",U.shape); 
        print("abs_sub:", abs_sub.shape)
        print("G:",G.shape)
    
    mx = torch.multiply(G.moveaxis((0,1),(2,3)),x)
    x_offset = torch.sum(mx, axis=-1).moveaxis((0,1),(-2,-1))   # batch_size x channels x output_length x kernel size

    if _test:
        print("mx:",mx.shape)
        print("x_offset:", x_offset.shape)
        print(
            "Desired shape:",
            (batch_size, x.shape[1], offsets.shape[-2], kernel_size),
            "(batch_size, in_channels, output_length, kernel_size)")
        assert x_offset.shape == (batch_size, x.shape[1],offsets.shape[-2], kernel_size)
    return x_offset

def _interpolate(i,x,t0s,T,kernel_rfield,x_offset,device):
    t0 = int(t0s[i,0].item())
    max_U = int(t0+kernel_rfield-1)
    U = torch.linspace(t0,max_U,kernel_rfield,device=device) #  kernel_size*max_dilation_factor
    abs_sub = 1-torch.abs(U.repeat(1,1,T.shape[-1],1)-T[:,:,i,:].unsqueeze(-1)) # batch_size x groups x kernel_size
    _zeros = torch.zeros(abs_sub.shape,device=device)
    G = torch.max(_zeros, abs_sub) # batch_size x channels x out_length x kernel_size x input_length
    mx = torch.multiply(G,x[:,:,t0:max_U+1].unsqueeze(-2))
    x_offset[:,:,i,:,] = torch.sum(mx, axis=-1)   # batch_size x channels x output_length x kernel size

def kernel_width_linterpolate(
    x, 
    offsets,
    kernel_size, 
    dilation,
    stride,
    dilated_positions=None,
    device="cpu",
    _test=False,
    _multiprocess=False,
    _max_memory=True
):
    assert x.device == offsets.device, "x and offsets must be on same device"
    kernel_rfield=dilation*(kernel_size-1)+1
    # Every index in x we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(0, kernel_rfield-1,kernel_size,device=device) # kernel_size
    
    max_t0 = (offsets.shape[-2]-1)*stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2],device=device).unsqueeze(-1) # out_length x 1
    dilated_offsets_repeated = dilated_positions+offsets
    
    T = t0s + dilated_offsets_repeated # batch_size x channels x out_length x kernel_size
    T = torch.max(T, t0s)
    T = torch.min(T, t0s+torch.max(dilated_positions))


    if _test:
        print("x:",x.shape) # batch_size x in_channels x input_length
        print("offsets:",offsets.shape) # batch_size x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:",t0s.shape) # out_lengths x 1
        print("dilated positions:",dilated_positions.shape) # kernel_size
        print("dilated_offsets_repeated:",dilated_offsets_repeated.shape)
        print("T:",T.shape) # batch_size x groups x out_length x kernel_rfield

    if _max_memory:
        U = t0s+torch.linspace(0,kernel_rfield-1,kernel_rfield,device=device).repeat(1,1,1,1) # 1 x 1 x 1 x length x kernel_rfield
        if _test:
            print("U:",U.shape)

        abs_sub = 1-torch.abs(U.unsqueeze(-1)-T.unsqueeze(-2)) # batch_size x groups x out_length x kernel_size x_length
        if _test:
            print("abs_sub:", abs_sub.shape)

        _zeros = torch.zeros(abs_sub.shape,device=device)
        x = x.unfold(dimension=2, size=kernel_rfield, step=stride).unsqueeze(-1)
        if _test:
            print("x unfolded:",x.shape)

        G = torch.max(_zeros, abs_sub) # batch_size x groups x out_length x kernel_rfield x kernel_size
        if _test:
            print("G:",G.shape)

        mx = torch.multiply(G,x)
        x_offset = torch.sum(mx, axis=-2)  # batch_size x channels x output_length x kernel size
        return x_offset

    elif not _multiprocess: 
        x_offset = torch.zeros((x.shape[0], x.shape[1], offsets.shape[-2], kernel_size),device=x.device)
        for i in range(t0s.shape[0]):
            t0 = int(t0s[i,0].item())
            max_U = int(t0+kernel_rfield-1)
            U = torch.linspace(t0,max_U,kernel_rfield,device=device) #  kernel_size*max_dilation_factor
            abs_sub = 1-torch.abs(U.repeat(1,1,T.shape[-1],1)-T[:,:,i,:].unsqueeze(-1)) # batch_size x groups x kernel_size
            _zeros = torch.zeros(abs_sub.shape,device=device)
            G = torch.max(_zeros, abs_sub) # batch_size x channels x out_length x kernel_size x input_length
            mx = torch.multiply(G,x[:,:,t0:max_U+1].unsqueeze(-2))
            x_offset[:,:,i,:,] = torch.sum(mx, axis=-1)   # batch_size x channels x output_length x kernel size
        return x_offset

    else:
        x_offset = torch.zeros((x.shape[0], x.shape[1], offsets.shape[-2], kernel_size),device=x.device)
        T.share_memory_()
        x.share_memory_()
        t0s.share_memory_()
        x_offset.share_memory_()
        with mp.Pool() as p:
            p.map(
                partial(_interpolate,t0s=t0s,T=T,x=x,x_offset=x_offset,kernel_rfield=kernel_rfield,device=x.device),
                range(t0s.shape[0])
                )
        return x_offset

# @torch.jit.script
def efficient_linterpolate(
    x, 
    offsets,
    kernel_size, 
    dilation,
    stride,
    dilated_positions=None,
    device="cpu",
    _test=False,
    unconstrained=False
):  

    assert x.device == offsets.device, "x and offsets must be on same device"
    kernel_rfield=dilation*(kernel_size-1)+1
    # Every index in x we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(0, kernel_rfield-1,kernel_size,device=offsets.device,dtype=offsets.dtype) # kernel_size

    max_t0 = (offsets.shape[-2]-1)*stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2],device=offsets.device,dtype=offsets.dtype).unsqueeze(-1) # out_length x 1
    dilated_offsets_repeated = dilated_positions+offsets
    
    T = t0s + dilated_offsets_repeated # batch_size x channels x out_length x kernel_size
    if not unconstrained:
        T = torch.max(T, t0s)
        T = torch.min(T, t0s+torch.max(dilated_positions))
    else:
        T = torch.clamp(T, 0.0, float(x.shape[-1]))

    if _test:
        print("x:",x.shape) # batch_size x in_channels x input_length
        print("offsets:",offsets.shape) # batch_size x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:",t0s.shape) # out_lengths x 1
        print("dilated positions:",dilated_positions.shape) # kernel_size
        print("dilated_offsets_repeated:",dilated_offsets_repeated.shape)
        print("T:",T.shape) # batch_size x groups x out_length x kernel_rfield

    with torch.no_grad():
        U = torch.floor(T).to(torch.long) # 1 x 1 x length x kernel_rfield
        U = torch.clamp(U,min=0,max=x.shape[2]-2)

        if _test:
            print("U:",U.shape)

        U = torch.stack([U,U+1],dim=-1)
        if U.shape[1] < x.shape[1]:
            U=U.repeat(1,x.shape[1],1,1,1)
        if _test:
            print("U:", U.shape)

    x=x.unsqueeze(-1).repeat(1,1,1,U.shape[-1])
    x = torch.stack([x.gather(index=U[:,:,:,i,:],dim=-2) for i in range(U.shape[-2])],dim=-1)
    
    G = torch.max(torch.zeros(U.shape,device=device), 1-torch.abs(U-T.unsqueeze(-1))) # batch_size x groups x out_length x kernel_rfield x kernel_size
    
    if _test:
        print("G:",G.shape)

    mx = torch.multiply(G,x.moveaxis(-2,-1))
    
    return torch.sum(mx, axis=-1) # .float()  # batch_size x channels x output_length x kernel size


@torch.jit.script
def _jit_efficient_linterpolate(
    x: Tensor, 
    offsets: Tensor,
    kernel_size: int, 
    dilation: int,
    stride: int,
    dilated_positions: Tensor = None,
    create_dilated_positions: bool = False,
    device: str = "cpu",
    _test: bool = False,
    unconstrained: bool= False
):  

    assert x.device == offsets.device, "x and offsets must be on same device"
    kernel_rfield=dilation*(kernel_size-1)+1
    # Every index in x we need to consider
    # if dilated_positions == None:
    # if not type(dilated_positions) == torch.Tensor:
    if create_dilated_positions:
        dilated_positions = torch.linspace(0, kernel_rfield-1,kernel_size,device=offsets.device,dtype=offsets.dtype) # kernel_size

    max_t0 = (offsets.shape[-2]-1)*stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2],device=offsets.device,dtype=offsets.dtype).unsqueeze(-1) # out_length x 1
    dilated_offsets_repeated = dilated_positions+offsets
    
    T = t0s + dilated_offsets_repeated # batch_size x channels x out_length x kernel_size
    if not unconstrained:
        T = torch.max(T, t0s)
        T = torch.min(T, t0s+torch.max(dilated_positions))
    else:
        T = torch.clamp(T, 0.0, float(x.shape[-1]))

    if _test:
        print("x:",x.shape) # batch_size x in_channels x input_length
        print("offsets:",offsets.shape) # batch_size x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:",t0s.shape) # out_lengths x 1
        print("dilated positions:",dilated_positions.shape) # kernel_size
        print("dilated_offsets_repeated:",dilated_offsets_repeated.shape)
        print("T:",T.shape) # batch_size x groups x out_length x kernel_rfield

    with torch.no_grad():
        U = torch.floor(T).to(torch.long) # 1 x 1 x length x kernel_rfield
        U = torch.clamp(U,min=0,max=x.shape[2]-2)

        if _test:
            print("U:",U.shape)

        U = torch.stack([U,U+1],dim=-1)
        if U.shape[1] < x.shape[1]:
            U=U.repeat(1,x.shape[1],1,1,1)
        if _test:
            print("U:", U.shape)

    x=x.unsqueeze(-1).repeat(1,1,1,U.shape[-1])
    x = torch.stack([x.gather(index=U[:,:,:,i,:],dim=-2) for i in range(U.shape[-2])],dim=-1)
    G = torch.max(torch.zeros_like(U, device=device), 1-torch.abs(U-T.unsqueeze(-1))) # batch_size x groups x out_length x kernel_rfield x kernel_size
    
    if _test:
        print("G:",G.shape)

    mx = torch.multiply(G,x.moveaxis(-2,-1))
    
    return torch.sum(mx, dim=-1) # .float()  # batch_size x channels x output_length x kernel size


def deform_conv1d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: Tuple[int, int] = (0, 0),
    dilation: int = 1,
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Performs Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

    Args:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
            offsets to be applied for each position in the convolution kernel.
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]): convolution weights,
            split into groups of size (in_channels // groups)
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
            masks to be applied for each position in the convolution kernel. Default: None

    Returns:
        Tensor[batch_sz, out_channels, out_h, out_w]: result of convolution

    Examples::
        >>> input = torch.rand(4, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
        >>> mask = torch.rand(4, kh * kw, 8, 8)
        >>> out = deform_conv2d(input, offset, weight, mask=mask)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([4, 5, 8, 8])
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(deform_conv1d)
    _assert_has_ops()

    out_channels = weight.shape[0]

    use_mask = mask is not None

    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_w = (stride)
    # pad_w = (padding)
    dil_w = (dilation)
    weights_w = weight.shape[-1]
    _, n_in_channels, _, _ = input.shape

    n_offset_grps = offset.shape[1] // (2  * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] .\n"
            f"Got offset.shape[1]={offset.shape[1]}, while 2 * weight.size[2] * weight.size[3]={2 * weights_w}"
        )

    return torch.ops.torchvision.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        1,
        stride_w,
        0,
        0,
        1,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    )

if __name__ == '__main__':
    import time
    # Use small values to easily observe effects
    batch_size = 1
    length = 150
    channels=12
    kernel_size = 3
    dilation = 3
    groups = 12
    stride = 2
    _test = True # Leave as false unless you want to see all intermediate shapes of linterpolate(...)
    torch.random.manual_seed(1234)

    # Input tensor
    x = torch.rand(batch_size, channels, length,requires_grad=True,device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute required number of offsets in the temporal direction (sequential length)
    # Uses delay/offset of 1 to test sample offset. Just edit to "2*torch.ones(..." for shift of 2 and so on
    num_samples = (x.shape[-1]-dilation*(kernel_size-1))//stride
    final_idx = (num_samples-1)*stride
    offsets = -0.5*torch.ones(batch_size, groups, num_samples, kernel_size,device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute interpolated offset positions of x
    # x_offset = full_seq_linterpolate(x, offsets, kernel_size, dilation, stride, _test=_test) # batch_size, in channels,output seq. length, kernel size
    # exit()
    # start=time.time()
    # x_offset = kernel_width_linterpolate(x, offsets, kernel_size, dilation, stride, _test=_test) # batch_size, in channels,output seq. length, kernel size
    # stop=time.time()
    # print("Ellapsed:",stop-start,"s")
    # # View results
    # if _test:
    #     print(f"Input {x.shape}:",x)
    #     print(f"Output {x_offset.shape}:",x_offset)

    start=time.time()
    x_offset = efficient_linterpolate(x, offsets, 
                kernel_size, dilation, stride,
                unconstrained=False,
                _test=_test,device='cuda' if torch.cuda.is_available() else 'cpu') # batch_size, in channels,output seq. length, kernel size
    stop=time.time()
    print("Ellapsed:",stop-start,"s")
    # View results
    if _test:
        print(f"Input {x.shape}:",x)
        print(f"Output {x_offset.shape}:",x_offset)
