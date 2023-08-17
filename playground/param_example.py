import time
import torch
from torch import nn

from dc1d.nn import PackedDeformConv1d, DeformConv1d

batch_size = 4
in_channels = 512
out_channels = 512
kernel_size = 3
stride = 1
padding = "same"
dilation = 2^7
groups = 512
bias = True
length = 133
packed = False

if packed:
    model = PackedDeformConv1d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        stride = stride,
        padding =  padding, # (kernel_size - 1) // 2,
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
        padding = padding,
        dilation = dilation,
        groups = groups,
        bias = True,
        unconstrained=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

x = torch.rand(batch_size, in_channels, length,requires_grad=True, device='cuda' if torch.cuda.is_available() else 'cpu')
print("Input shape",x.shape)

output_length = x.shape[-1]-dilation*(kernel_size-1)

offsets = nn.Parameter(torch.ones(1, 1, 1, kernel_size, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True))

offsets = offsets.repeat(batch_size,1,length,1)

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
    padding = padding,
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


