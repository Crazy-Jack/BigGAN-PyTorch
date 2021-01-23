import torch
from torch.autograd import Variable

torch.manual_seed(0)
x = Variable(torch.randn(3,3), requires_grad=True)
print(f"x{x}")
idx = Variable(torch.FloatTensor([0,1]), requires_grad=True)
print("idx", idx)
i0 = idx.floor().detach()
i1 = i0 + 1
print(f"i0{i0}; i1{i1};")
y0 = x.index_select(0, i0.long())
y1 = x.index_select(0, i1.long())

print(f"y0{y0}")
print(f"y1{y1}")
Wa = (i1 - idx).unsqueeze(1).expand_as(y0)
Wb = (idx - i0).unsqueeze(1).expand_as(y1)

print(f"Wa{Wa}")
print(f"Wb{Wb}")
out = Wa * y0 + Wb * y1

print(out)
out.sum().backward()
print(idx.grad)