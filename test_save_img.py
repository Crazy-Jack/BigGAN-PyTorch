import torch
import torchvision 



a = torch.rand(5, 9, 200, 200)
a[:,0,:,:]=0.3
a[:,1,:,:]=0.5
a[:,2,:,:]=0.7

a1 = a.view(5, 9, 1, 200, 200).repeat(1,1,3,1,1) # [5, 9, 3, 200, 200]
print(a1.shape)
b = torch.ones(5, 3, 200,200).unsqueeze(1) # [5, 1, 3, 200, 200]
print(b.shape)
c = torch.cat([a1, b], dim=1).view(5*10,3, 200, 200)
print(c.shape)
torchvision.utils.save_image(c, "test.jpg", nrow=10)