import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np 



class ModularNet(nn.Module):
    """test modular network"""
    def __init__(self):
        super(ModularNet, self).__init__()
        self.list_of_nn = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

        # print(self.modules)
    def forward(self, x):
        print("x.train", self.list_of_nn[0].training)
        list_of_out = [i(x) for i in self.list_of_nn]
        return list_of_out


model = ModularNet().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


x = torch.rand(20, 10).cuda()
model.train()
model(x)

# for _ in range(100):
#     optimizer.zero_grad()
#     out_list = model(x)
#     # print([i.mean().item() for i in out_list])
#     loss = torch.mean(torch.cat(out_list))
#     loss.backward()
#     optimizer.step()
