

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable





#################################################
#     Reconstruct convolutionally/or not        #
#################################################
class NeuralConvRecon(nn.Module):
    """conv reconstruct by mlp"""
    def __init__(self, ch, resolution, kernel_size, mode="mlp"):
        super(NeuralConvRecon, self).__init__()
        self.kernel_size = kernel_size
        self.linear1_out = (ch // kernel_size) * (kernel_size * kernel_size) 
        self.conv_ch_in = ch // kernel_size
        self.interm_ch = self.linear1_out // (kernel_size * kernel_size)
        self.mode = mode
        # define architecture
        if self.mode == "mlp":
            self.linear1 = nn.Linear(ch, self.linear1_out // 2)
            self.relu1 = nn.ReLU()
            self.linear2 = nn.Linear(self.linear1_out // 2, self.linear1_out)
            self.conv = nn.Conv2d(self.conv_ch_in, ch, kernel_size=1, padding=0, bias=False)
            self.relu2 = nn.ReLU()
        elif self.mode == "deconv":
            # deconvolutionally fill in the holes
            # upsample

            # 
            pass

        elif self.mode == "Intepolation":
            # TODO: reconstruct by intepolate
            pass 
        elif self.mode == "noconv":
            # don't have convolution to avoid blur results?
            pass

        else:
            raise NotImplementedError(f"mode {self.mode} is not supported in NeuralConvRecon")


        # fold
        self.fold = nn.Fold(output_size=(resolution, resolution), kernel_size=(kernel_size, kernel_size), stride=3)

    def forward(self, x):
        """
        x: [n, L, c]
        [n, L, c] -> [n, L, c * kernel * kernel] -> [n, c * kernel * kernel, L] -> fold => [n, c, h, w]
        output: [n, c, h, w]
        """
        n, L, c = x.shape
        x = self.linear1(x.view(-1, c)) # [n * L, c] -> [n * L, self.linear1_out // 2]
        x = self.relu1(x)
        x = self.linear2(x) # [n * L, self.linear1_out]
        x = x.view(n * L, -1, self.kernel_size, self.kernel_size) # [n * L, ch // kernel_size, kernel_size, kernel_size]
        x = self.conv(x) # [n * L, ch, kernel_size, kernel_size]
        x = x.view(n, L, c * self.kernel_size * self.kernel_size)
        x = torch.transpose(x, 1, 2) # [n, c * kernel * kernel, L]
        x = self.fold(x) # n, c, h, w

        return x


class NeuralNormalRecon(nn.Module):
    """reconstruct by without convolution to avoid blur results"""
    def __init__(self, ch, resolution, kernel_size, mode="mlp"):
        super(NeuralNormalRecon, self).__init__()
        self.kernel_size = kernel_size

        


    def forward(self, x, keypoints):
        """
        x: [n, L, c], 
        keypoints: [n, L, 1]
        [n, L, c] -> [n, L, c * kernel * kernel] -> [n, c * kernel * kernel, L] -> fold => [n, c, h, w]
        output: [n, c, h, w]
        """

        # convert x and keypoint to a map
        keypoints

        # break
        