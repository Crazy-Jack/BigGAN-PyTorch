''' Layers
    This file contains various layers for the BigGAN models.
'''
import functools

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable
import faiss

from sync_batchnorm import SynchronizedBatchNorm2d as SyncBN2d

from torch.distributions import Categorical
from layers import SNConv2d



# the original attention was self-attention
# here is an attention module to be used by memory bank
class ConceptAttention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d):
        super(ConceptAttention, self).__init__()
        self.myid = "atten"
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(
            self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y):
        # x from encoder (N x 128 x 128), 64 (64 is the feature map size of that layer
        # y from memory bank 512, 64
        # Apply convs
        theta = self.theta(x)
        phi = self.phi(y)
        g = self.g(y)
        # phi = F.max_pool2d(self.phi(x), [2, 2])
        # g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 2, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 2, y.shape[0])
        g = g.view(-1, self.ch // 2, y.shape[0])
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                           self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


class MemoryClusterAttention(nn.Module):
    def __init__(self, dim=64, n_embed=512):
        super(MemoryClusterAttention, self).__init__()

        self.dim = dim   # set to 64 currently
        self.n_embed = n_embed   # vq vae use 512, maybe we can use the same?

        embed = torch.randn(n_embed, dim)
        self.register_parameter("embed", embed)
        self.attention_module = ConceptAttention(dim)

        # currently dont need them as we don't have multiple clusters
        # self.register_buffer("cluster_size", torch.zeros(n_embed))
        # self.register_buffer("embed_avg", embed.clone())

    def attend_to_memory_bank(self, x):
        return self.attention_module.forward(x, self.embed)

    def forward(self, input: torch.Tensor):
        # input_dim [batch_size, emb_dim, h, w]
        return self.attend_to_memory_bank(input)



if __name__ == '__main__':
    concept_atten_proto = ConceptAttentionProto(pool_size_per_cluster=100, num_k=20, feature_dim=128)
    results_kmean = concept_atten_proto.pool_kmean_init()