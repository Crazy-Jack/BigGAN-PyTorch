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


from sync_batchnorm import SynchronizedBatchNorm2d as SyncBN2d

from layer_conv_select import SparseNeuralConv
from layer_vc_linear_comb import LinearCombineVC
from layer_conv_select_multiple_path import SparseNeuralConvMulti
from torch.distributions import Categorical
from layers import SNConv2d


class ConceptAttentionProto(nn.Module):
    """concept attention"""
    def __init__(self, pool_size_per_cluster, num_k, feature_dim, warmup_total_iter=1000, device='cuda'):
        self.device = device
        self.pool_size_per_cluster = pool_size_per_cluster
        self.num_k = num_k
        self.feature_dim = feature_dim
        self.total_pool_size = self.num_k * self.pool_size_per_cluster
        self.register_buffer('concept_pool', torch.zeros(self.feature_dim, self.total_pool_size))
        self.register_buffer('concept_proto', torch.zeros(self.feature_dim, self.num_k))
        # concept pool is arranged as memory cell, i.e. linearly arranged as a 2D tensor, use get_cluster_ptr to get starting pointer for each cluster
        

        # states that indicating the warmup
        self.warmup_state = 1 # 1 means during warm up, will switch to 0 after warmup
        self.warmup_iter_counter = 0
        self.warmup_total_iter = warmup_total_iter
        self.pool_structured = 0 # 0 means pool is un clustered, 1 mean pool is structured as clusters arrays

        # register attention module
        self.attention_module = ConceptAttention(self.feature_dim)


    def get_cluster_ptr(self, cluster_num):
        """get starting pointer for cluster_num"""
        assert cluster_num < self.num_k, f"cluster_num {cluster_num} out of bound (totally has {self.num_k} clusters)"
        return self.pool_size_per_cluster * cluster_num
    
    def _update_pool(self, index, content):
        """update concept pool according to the content"""
        assert (index >= self.total_pool_size).sum() == 0, f"index contains index that larger/equal to pool size"
        assert len(index.shape) = 1
        assert content.shape[1] == index.shape[0]
        assert content.shape[0] == self.feature_dim
        
        self.concept_pool[:, index] = content 
    
    
    def pool_kmean_init(self):
        """perform kmeans for cluster concept pool initialization"""
        



    def _check_warmup_state(self):
        """check if need to switch warup_state to 0; when turn off warmup state, trigger k-means init for clustering"""
        if self.warmup_state:
            if self.warmup_iter_counter > self.warmup_total_iter:
                self.warmup_state = 0
                # trigger kmean concept pool init

        else:
            raise Exception("Calling _check_warmup_state when self.warmup_state is 0")


    def warmup_sampling(self, x):
        """
        linearly sample input x to make it 
        x: [n, c, h, w]"""
        n, c, h, w = shape
        assert self.warmup_state, "calling warmup sampling when warmup state is 0"
        
        # evenly distributed across space
        sample_per_instance = max(int(self.total_pool_size / n), 1)
        
        # sample index
        index = torch.randint(h * w, size=(n, 1, sample_per_instance)).repeat(1, c, 1).to(self.device) # n, c, sample_per_instance
        sampled_columns = torch.gather(x.reshape(n, c, h * w), 2, index) # n, c, sample_per_instance 
        sampled_columns = torch.transpose(sampled_columns, 1, 0).reshape(c, -1).contiguous() # c, n * sample_per_instance
        
        # calculate percentage to populate into pool, as the later the better, use linear intepolation from 1% to 50% according to self.warmup_iter_couunter
        percentage = (self.warmup_iter_counter + 1) / self.warmup_total_iter * 0.5 # max percent is 50%
        sample_column_num = max(1, int(percentage * sampled_columns.shape[1]))
        sampled_columns_idx = torch.randint(sampled_columns.shape[1], size=(sample_column_num,))
        sampled_columns = sampled_columns[:, sampled_columns_idx]  # [c, sample_column_num]

        # random select pool idx to update
        update_idx = torch.randperm(self.concept_pool.shape[1])[:sample_column_num]
        self._update_pool(update_idx, sampled_columns)

        # update number
        self.warmup_iter_counter += 1


    def forward(self, x, device="cuda"):
        n, c, h, w = x.shape 

        # warmup
        if self.warmup_state:
            # if still in warmup, skip attention
            self.warmup_sampling(x)
            self._check_warmup_state()
            return x

        else:
            # attend to concepts
            # selecting 

            

        
         



# the original attention was self-attention
# here is an attention module to be used by memory bank
class ConceptAttention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d):
        super(Attention, self).__init__()
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
        super().__init__()

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
