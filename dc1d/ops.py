""" 
ops.py provides operation functions for defomrable convolution

Author: William Ravenscroft, August 2022
Copyright William Ravenscroft 2022
"""

# PyTorch
import torch

def linterpolate(
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
    Linear interpolation function for 1D deformable convolution. 
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
    
    if _test:
        print("x:",x.shape) # batch_size x in_channels x input_length
        print("offsets:",offsets.shape) # batch_size x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:",t0s.shape) # out_lengths x 1
        print("dilated positions:",dilated_positions.shape) # kernel_size
        print("dilated_offsets_repeated:",dilated_offsets_repeated.shape)
    
    T = t0s + dilated_offsets_repeated # batch_size x groups x out_length x kernel_size
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


if __name__ == '__main__':

    # Use small values to easily observe effects
    batch_size = 1
    length = 10
    channels=2
    kernel_size = 3
    dilation = 2
    groups = channels
    stride = 2
    _test = False # Leave as false unless you want to see all intermediate shapes of linterpolate(...)

    # Input tensor
    x = torch.rand(batch_size, channels, length,requires_grad=True)
    
    # Compute required number of offsets in the temporal direction (sequential length)
    # Uses delay/offset of 1 to test sample offset. Just edit to "2*torch.ones(..." for shift of 2 and so on
    num_samples = (x.shape[-1]-dilation*(kernel_size-1))//stride
    final_idx = (num_samples-1)*stride
    offsets = torch.ones(batch_size, groups, num_samples, kernel_size)
    
    # Compute interpolated offset positions of x
    x_offset = linterpolate(x, offsets, kernel_size, dilation, stride, _test=_test) # batch_size, in channels,output seq. length, kernel size
    
    # View results
    print("Input:",x)
    print("Output:",x_offset)
