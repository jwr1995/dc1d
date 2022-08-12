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
    Inputs:
        x, 
        offsets,
        kernel_size, 
        dilation,
        stride,
        dilated_positions=None,
        device="cpu"
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