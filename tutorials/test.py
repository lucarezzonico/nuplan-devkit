import torch
from torch import nn

x = torch.ones(2, 2, requires_grad=True)
y = torch.ones(2, 2, requires_grad=True)

z = x + y                       # AddBackward
z = x @ y                       # MmBackward
z = torch.cat((x, x, x), 0)     # ACatBackward
z = x[:,0:2]                    # SliceBackward ??
z = x[0,0:2]                    # SelectBackward ??
z = x.view(4,-1)                # ViewBackward
z = torch.sum(x)
z = torch.mean(x)
z = torch.clone(x)

print(z)


input = torch.ones(5, 3, requires_grad=True)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, requires_grad=True)
index = torch.tensor([0, 4, 2])
z = input.index_add(0, index, t, alpha=2) # IndexAddBackward

print(z)

lin = nn.Linear(16,2)
z = lin(torch.ones(16,16, requires_grad=True)) # AddmmBackward

print(z)

conv = nn.Conv2d(1,4,2)
z = conv(torch.ones(4,1,2,2, requires_grad=True)) # MkldnnConvolutionBackward

print(z)