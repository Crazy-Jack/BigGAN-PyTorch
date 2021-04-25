import torch 
import torch.nn as nn 
import torchvision 
import os


class ConceptAttentionProto(nn.Module):
    """concept attention"""
    def __init__(self, ):
        self.pool_size_per_cluster = pool_size_per_cluster
        self.num_k = num_k
        self.feature_dim = feature_dim
        self.register_buffer('concept_pool', torch.zeros(self.num_k * self.pool_size_per_cluster, self.feature_dim))
        # concept pool is arranged as memory cell, i.e. linearly arranged as a 2D tensor

        # states that indicating the warmup
        self.warmup_state = 1
    

    def get_cluster_ptr(self, cluster_num):
        """get starting pointer for cluster_num"""
        assert cluster_num < self.num_k, f"cluster_num {cluster_num} out of bound (totally has {self.num_k} clusters)"
        

        
        
    def forward(self, x, device="cuda"):
        n, c, h, w = x.shape 

        # 





